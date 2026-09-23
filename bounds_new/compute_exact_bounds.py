"""CLI to compute exact local and global bounds for a trainer checkpoint.

Usage (example):
  python -m bounds_new.compute_exact_bounds --model model_final.pth --height 20 --width 20 --goal 399 --sonar-number 8 --unknown-number 2 --layer both
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import numpy as np
import concurrent.futures
import time

import torch

from . import io as io_mod
from . import precompute as premod
from . import solver as solver_mod
import dataset.static_cache as sc
from environment.utils import detection_probability
# use local io.save_bounds to avoid depending on bounds/


def _find_state_param(state: dict, name: str):
    if name in state:
        return state[name]
    mname = "module." + name
    if mname in state:
        return state[mname]
    for k in state.keys():
        if k.endswith(name) or name in k:
            return state[k]
    raise KeyError(f"Parameter '{name}' not found in checkpoint state_dict keys")


def _local_feature_slices(R: int) -> dict:
    # replicate ordering used in builder.build_local_input
    pos = 0
    slices = {}
    slices['cost'] = (pos, pos + 1); pos += 1
    slices['obstacle_placement'] = (pos, pos + 1); pos += 1
    slices['sonar_placement'] = (pos, pos + 1); pos += 1
    slices['axial_coordinates'] = (pos, pos + 2); pos += 2
    slices['cartesian_coordinates'] = (pos, pos + 2); pos += 2
    slices['distance_to_goal'] = (pos, pos + 1); pos += 1
    slices['known_sonar_mask'] = (pos, pos + 1); pos += 1
    slices['unknown_sonar_mask'] = (pos, pos + 1); pos += 1
    slices['neighbor_cost'] = (pos, pos + 6); pos += 6
    slices['neighbor_obstacle'] = (pos, pos + 6); pos += 6
    slices['neighbor_sonar'] = (pos, pos + 6); pos += 6
    slices['neighbor_mask'] = (pos, pos + 6); pos += 6
    # rings: 5 chunks of R each
    slices['ring_cost'] = (pos, pos + R); pos += R
    slices['ring_obstacle_count'] = (pos, pos + R); pos += R
    slices['ring_obstacle_density'] = (pos, pos + R); pos += R
    slices['ring_sonar_count'] = (pos, pos + R); pos += R
    slices['ring_sonar_density'] = (pos, pos + R); pos += R
    return slices


def _add_relu_bigM(m, z_var, a_var, L: float, U: float, name_prefix: str | None = None):
    try:
        from gurobipy import GRB
    except Exception as e:
        raise RuntimeError("Gurobi (gurobipy) is required for exact global tightening") from e

    if not np.isfinite(L):
        L = -1e6
    if not np.isfinite(U):
        U = 1e6

    m.addConstr(z_var >= float(L))
    m.addConstr(z_var <= float(U))

    if U <= 0.0:
        m.addConstr(a_var == 0.0)
        return None
    if L >= 0.0:
        m.addConstr(a_var == z_var)
        return None

    y = m.addVar(vtype=GRB.BINARY, name=(f"{name_prefix}_y" if name_prefix else None))
    m.addConstr(a_var >= z_var)
    m.addConstr(a_var >= 0.0)
    m.addConstr(a_var <= float(U) * y)
    m.addConstr(a_var <= z_var - float(L) * (1 - y))
    return y


def compute_local_bounds(
    checkpoint_path: str | Path,
    height: int,
    width: int,
    goal_idx: int,
    obstacles: list[int] | None,
    detection_probability = detection_probability,
    sonar_number: int | None = None,
    unknown_number: int | None = None,
    prune_eps: float | None = None ,
    time_limit: float | None = None,
    jobs: int | None = None,
    out_dir: Path | None = None,
):
    # ensure c_z, const_z precompute
    _, c_z, const_z = premod.precompute_model_cz(checkpoint_path, height=height, width=width, goal_idx=goal_idx, obstacles=obstacles, detection_probability=detection_probability)

    # load model state for weight access
    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    if isinstance(ckpt, dict) and ("state_dict" in ckpt or "model_state_dict" in ckpt):
        state = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
    elif isinstance(ckpt, dict):
        state = ckpt
    else:
        state = ckpt.state_dict()

    W_loc_t = _find_state_param(state, "local_hidden_weight")
    B_loc_t = _find_state_param(state, "local_hidden_bias")
    W_loc = W_loc_t.cpu().numpy()
    B_loc = B_loc_t.cpu().numpy()

    # static info
    static = sc.build_static(height=height, width=width, obstacle_indices=obstacles, detection_probability=detection_probability)
    N = int(static.raw_map.total_cells)
    R = int(static.ring_obstacle_count.shape[1])
    num_vars = c_z.shape[2]

    slices = _local_feature_slices(R)
    known_idx = slices['known_sonar_mask'][0]
    unknown_idx = slices['unknown_sonar_mask'][0]

    fixed_zero = set(int(x) for x in static.obstacles)

    cells = []
    neurons = []
    Ls = []
    Us = []

    Nn = W_loc.shape[1]

    total = int(N * Nn)
    start = time.time()
    idx = 0
    for i in range(N):
        for h in range(Nn):
            # coeff on placements
            coeff_x_full = np.asarray(c_z[i, h, :], dtype=float).reshape(-1)
            const = float(const_z[i, h])

            # add known/unknown feature contributions (only at placement index == i)
            w = np.asarray(W_loc[i, h, :], dtype=float)
            wk = float(w[known_idx])
            wu = float(w[unknown_idx])

            # ensure length
            if i < coeff_x_full.size:
                coeff_x_full[i] = float(coeff_x_full[i]) + wk
            else:
                # fallback if shapes mismatch
                pass

            coeff_u_full = None
            if unknown_number is not None and unknown_number > 0:
                coeff_u_full = np.zeros_like(coeff_x_full)
                if i < coeff_u_full.size:
                    coeff_u_full[i] = wu - wk

            # pick var indices that matter
            var_idx = np.nonzero(np.abs(coeff_x_full) > prune_eps)[0].tolist()
            # ensure feasibility for cardinality
            if sonar_number is not None and len(var_idx) < int(sonar_number):
                var_idx = list(range(num_vars))

            try:
                L, U = solver_mod.solve_min_max(
                    coeff_x_full[var_idx],
                    coeff_u_full[var_idx] if coeff_u_full is not None else None,
                    const,
                    var_idx,
                    num_vars,
                    sonar_number=sonar_number,
                    fixed_zero=fixed_zero,
                    unknown_count=unknown_number,
                    time_limit=time_limit,
                    threads=jobs,
                    sign_stop_at_zero=True,
                    verbose=False,
                )
            except Exception as e:
                raise RuntimeError(f"Solver failed for neuron (cell={i}, h={h}): {e}") from e

            cells.append(int(i))
            neurons.append(int(h))
            Ls.append(float(L))
            Us.append(float(U))

            idx += 1
            if idx % 100 == 0:
                now = time.time()
                print(f"processed {idx}/{total} neurons ({now-start:.1f}s)")

    arrays = {"cell": np.asarray(cells, dtype=np.int32), "neuron": np.asarray(neurons, dtype=np.int32), "L": np.asarray(Ls, dtype=np.float64), "U": np.asarray(Us, dtype=np.float64)}
    out_path = out_dir / "local_bounds.npz"
    meta = {"layer": "local", "checkpoint": str(checkpoint_path), "created": time.asctime(), "sonar_number": sonar_number} #"unknown_number": unknown_number
    io_mod.atomic_save_npz(out_path, arrays, meta=meta)
    print(f"saved local bounds to {out_path}")
    return out_path


def compute_global_bounds(
    checkpoint_path: str | Path,
    height: int,
    width: int,
    goal_idx: int,
    obstacles: list[int] | None,
    detection_probability,
    sonar_number: int | None,
    unknown_number: int | None,
    prune_eps: float,
    jobs: int | None,
    time_limit: float | None,
    out_dir: Path,
    local_bounds_npz: str,
    verbose: bool = False,
):
    # load precomputes and model params
    model_dir = out_dir
    c_z = np.load(model_dir / "c_z.npy", mmap_mode="r")
    const_z = np.load(model_dir / "const_z.npy", mmap_mode="r")

    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    if isinstance(ckpt, dict) and ("state_dict" in ckpt or "model_state_dict" in ckpt):
        state = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
    elif isinstance(ckpt, dict):
        state = ckpt
    else:
        state = ckpt.state_dict()

    W_loc = _find_state_param(state, "local_hidden_weight").cpu().numpy()
    B_loc = _find_state_param(state, "local_hidden_bias").cpu().numpy()
    W_out_t = _find_state_param(state, "local_output_weight").cpu().numpy()
    b_out_t = _find_state_param(state, "local_output_bias").cpu().numpy()

    # global layer
    Wg = _find_state_param(state, "global_hidden.weight").cpu().numpy()
    bg = _find_state_param(state, "global_hidden.bias").cpu().numpy()

    N = int(c_z.shape[0])
    H = int(W_loc.shape[1])
    O = int(W_out_t.shape[1])
    G = int(Wg.shape[0])

    if verbose:
        print(f"Computing global bounds: checkpoint={checkpoint_path}, height={height}, width={width}, goal={goal_idx}")
        print(f"  N={N}, H={H}, O={O}, G={G}, prune_eps={prune_eps}, sonar_number={sonar_number}, unknown_number={unknown_number}, jobs={jobs}, time_limit={time_limit}")

    # load local bounds
    data = np.load(local_bounds_npz, allow_pickle=True)
    L_arr = np.full((N, H), -1e6, dtype=float)
    U_arr = np.full((N, H), 1e6, dtype=float)
    if 'cell' in data and 'neuron' in data and 'L' in data and 'U' in data:
        for c, n, l, u in zip(data['cell'], data['neuron'], data['L'], data['U']):
            ci = int(c); hi = int(n)
            if 0 <= ci < N and 0 <= hi < H:
                L_arr[ci, hi] = float(l)
                U_arr[ci, hi] = float(u)
    else:
        raise RuntimeError("local_bounds_npz missing required arrays")

    # static obstacles -> fixed_zero
    static = sc.build_static(height=height, width=width, obstacle_indices=obstacles, detection_probability=detection_probability)
    fixed_zero = set(int(x) for x in static.obstacles)

    # prepare local feature slices to locate known/unknown mask indices
    R = int(static.ring_obstacle_count.shape[1])
    slices = _local_feature_slices(R)
    known_idx = slices['known_sonar_mask'][0]
    unknown_idx = slices['unknown_sonar_mask'][0]

    # global input partition expected by trainer:
    # [local_out_flat (N * O), goal_onehot (N), obstacle_number (1), sonar_number (1),
    #  start_as_known_number (1), start_as_unknown_number (1)]
    local_flat_dim = N * O
    expected_global_in = local_flat_dim + N + 4
    if Wg.shape[1] != expected_global_in:
        raise ValueError(
            f"Unexpected global_hidden input dimension; got {Wg.shape[1]}, expected {expected_global_in} "
            f"for N={N}, local_output_dimension={O}"
        )

    Wg_local = Wg[:, :local_flat_dim].reshape(G, N, O)
    Wg_goal = Wg[:, local_flat_dim:local_flat_dim + N]
    Wg_obst = Wg[:, local_flat_dim + N + 0]
    Wg_sonar = Wg[:, local_flat_dim + N + 1]
    Wg_start_known = Wg[:, local_flat_dim + N + 2]
    Wg_start_unknown = Wg[:, local_flat_dim + N + 3]

    out = {}
    # optionally load existing out_path to merge
    out_path = out_dir / "global_bounds.npz"
    existing = {}
    if out_path.exists():
        try:
            ex = np.load(out_path, allow_pickle=True)
            if 'cell' in ex and 'neuron' in ex and 'L' in ex and 'U' in ex:
                for c, n, l, u in zip(ex['cell'], ex['neuron'], ex['L'], ex['U']):
                    existing[(int(c), int(n))] = (float(l), float(u))
        except Exception:
            existing = {}

    # ensure c_z and const_z shapes (they were precomputed)
    c_z = np.asarray(c_z, dtype=float)
    const_z = np.asarray(const_z, dtype=float)

    num_vars = c_z.shape[2]

    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception as e:
        raise RuntimeError("Gurobi (gurobipy) is required for exact global tightening") from e

    for g in range(G):
        key = (-1, int(g))
        if key in existing:
            out[key] = existing[key]
            continue

        # form per-(i,h) coefficient on a_{i,h} by summing the contribution of
        # every local output channel that is flattened into the global input.
        coef_ih = np.einsum('io,ioh->ih', Wg_local[g], W_out_t)

        # When unknowns are present we must avoid pruning that could drop
        # placement variables needed by adversarial `u` — use a relaxed
        # prune tolerance (0.0) in that case to preserve provable tightness.
        p_eps = float(prune_eps) if prune_eps is not None else 0.0
        if unknown_number is not None and int(unknown_number) > 0:
            p_eps = 0.0
        included_local = [(i, h) for i in range(N) for h in range(H) if abs(coef_ih[i, h]) > p_eps]

        # constant contributions: global bias, Wg*b_out, goal onehot, and scalar globals
        const_global = float(bg[g])
        const_global += float(np.sum(Wg_local[g] * b_out_t))
        # goal one-hot contribution
        const_global += float(Wg_goal[g, goal_idx])
        # scalar globals: obstacle_number, sonar_number, start_as_known_number, start_as_unknown_number
        obst_num = float(len(static.obstacles))
        snum = float(0 if sonar_number is None else int(sonar_number))
        unk = float(0 if unknown_number is None else int(unknown_number))
        known = float(snum - unk)
        const_global += float(Wg_obst[g] * obst_num)
        const_global += float(Wg_sonar[g] * snum)
        const_global += float(Wg_start_known[g] * known)
        const_global += float(Wg_start_unknown[g] * unk)

        if len(included_local) == 0:
            out[key] = (const_global, const_global)
            # save incremental
            existing[key] = out[key]
            io_mod.save_bounds(str(out_path), existing, meta=dict(layer='global', sonar_number=sonar_number, fixed_zero=list(fixed_zero), time_limit=time_limit))
            continue

        # compute placement mask including known-mask (wk) contributions and
        # potential unknown (u) contributions (wu - wk) when unknowns are present.
        is_idx = np.array([int(i) for i, _ in included_local], dtype=int)
        hs_idx = np.array([int(h) for _, h in included_local], dtype=int)
        # base_mask rows: shape (len(included_local), num_vars)
        if is_idx.size > 0:
            # use same prune tolerance as above
            mask_rows = np.abs(c_z[is_idx, hs_idx, :]) > p_eps
            # include the effect of known_sonar_mask weight wk at column == i,
            # and also include the unknown(u) delta (wu - wk) if unknowns are considered
            wks = np.asarray([float(W_loc[i, h, known_idx]) for i, h in included_local], dtype=float)
            wus = np.asarray([float(W_loc[i, h, unknown_idx]) for i, h in included_local], dtype=float)
            for r in range(is_idx.size):
                i_cell = int(is_idx[r])
                # consider adjusted coefficient at the placement index i_cell
                #if abs(float(c_z[i_cell, hs_idx[r], i_cell]) + float(wks[r])) > p_eps:
                if abs(float(c_z[i_cell, hs_idx[r], i_cell])) > p_eps:
                    mask_rows[r, i_cell] = True
                # if unknowns are relevant and the delta is non-negligible, include that placement var
                if unknown_number is not None and int(unknown_number) > 0:
                    delta = float(wus[r] - wks[r])
                    if abs(delta) > p_eps:
                        mask_rows[r, i_cell] = True
            placement_mask = np.any(mask_rows, axis=0)
        else:
            placement_mask = np.zeros((num_vars,), dtype=bool)

        vars_idx = [j for j in range(num_vars) if placement_mask[j]]

        # For correctness with adversarial unknowns we must allow `u` to be
        # placed anywhere the local solver could place them. The simplest
        # safe policy is to include all placement vars when unknowns > 0.
        if unknown_number is not None and int(unknown_number) > 0:
            if len(vars_idx) != num_vars:
                vars_idx = list(range(num_vars))
        else:
            if sonar_number is not None and len(vars_idx) < int(sonar_number):
                vars_idx = list(range(num_vars))
        var_index_map = {orig_j: idx for idx, orig_j in enumerate(vars_idx)}
        num_included_vars = len(vars_idx)

        # build Gurobi model
        if verbose:
            print(f"[g={g}] building MILP (included_local={len(included_local)})")
        m = gp.Model(f'global_bound_g{g}')
        # enable Gurobi logging when verbose
        m.setParam('OutputFlag', 1 if verbose else 0)
        m.setParam("NumericFocus", 3)
        if time_limit is not None:
            m.setParam('TimeLimit', float(time_limit))
        if jobs is not None and int(jobs) > 0:
            m.setParam('Threads', int(jobs))

        # x variables for included placements
        xs = [m.addVar(vtype=GRB.BINARY, name=f'x_{j}') for j in vars_idx]
        # optional u variables for unknown sonars (one per included placement)
        us = None
        if unknown_number is not None and int(unknown_number) > 0:
            us = [m.addVar(vtype=GRB.BINARY, name=f'u_{j}') for j in vars_idx]
        # fix obstacles among included vars
        for orig_j, var_j in var_index_map.items():
            if orig_j in fixed_zero:
                m.addConstr(xs[var_j] == 0)
                if us is not None:
                    m.addConstr(us[var_j] == 0)
        
        # cardinality constraint for sonars
        if sonar_number is not None:
            m.addConstr(gp.quicksum(xs) == int(sonar_number))
        # cardinality constraint for unknowns
        if us is not None:
            m.addConstr(gp.quicksum(us) == int(unknown_number))
        # link u_j <= x_j
        if us is not None:
            for ridx in range(len(vars_idx)):
                m.addConstr(us[ridx] <= xs[ridx])

        # create z, a, s vars for included local neurons
        z_vars = {}
        a_vars = {}
        for (i, h) in included_local:
            z = m.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name=f'z_{i}_{h}')
            a = m.addVar(lb=0.0, ub=gp.GRB.INFINITY, name=f'a_{i}_{h}')
            z_vars[(i, h)] = z
            a_vars[(i, h)] = a
            # z = const + sum_j c_j * x_j  (only included placement vars)
            # include known-sonar-mask weight (wk) and unknown (wu) contributions at placement index i
            coeffs = np.asarray(c_z[i, h, :], dtype=float).copy()
            try:
                wk = float(W_loc[i, h, known_idx])
            except Exception:
                wk = 0.0
            try:
                wu = float(W_loc[i, h, unknown_idx])
            except Exception:
                wu = 0.0
            # add wk contribution at placement index i
            if i < coeffs.size:
                coeffs[i] = float(coeffs[i]) + wk
            if num_included_vars > 0:
                # sum over x contributions
                x_term = gp.quicksum(float(coeffs[j]) * xs[var_index_map[j]] for j in vars_idx)
                # include u contributions if present (delta = wu - wk)
                if us is not None:
                    delta = float(wu - wk)
                    if i in var_index_map and abs(delta) > prune_eps:
                        u_idx = var_index_map[i]
                        u_term = delta * us[u_idx]
                    else:
                        u_term = 0.0
                    m.addConstr(z == float(const_z[i, h]) + x_term + u_term)
                else:
                    m.addConstr(z == float(const_z[i, h]) + x_term)
            else:
                m.addConstr(z == float(const_z[i, h]))
            # ReLU big-M using L_arr/U_arr
            L = float(L_arr[i, h])
            U = float(U_arr[i, h])
            if not np.isfinite(L):
                L = -1e6
            if not np.isfinite(U):
                U = 1e6
            _add_relu_bigM(m, z, a, L, U, name_prefix=f's_{i}_{h}')

        # build objective coefficients on a_vars
        obj_terms = []
        for (i, h) in included_local:
            coef = float(coef_ih[i, h])
            obj_terms.append(coef * a_vars[(i, h)])

        obj_expr = gp.quicksum(obj_terms) + float(const_global)

        def _solve_maximize(maximize: bool):
            if maximize:
                if verbose:
                    print(f"[g={g}] optimizing (maximize)")
                m.setObjective(obj_expr, GRB.MAXIMIZE)
            else:
                if verbose:
                    print(f"[g={g}] optimizing (minimize)")
                m.setObjective(obj_expr, GRB.MINIMIZE)
            sign_proved = {"done": False}

            def _sign_callback(model, where):
                if where != GRB.Callback.MIP:
                    return
                try:
                    best_bd = float(model.cbGet(GRB.Callback.MIP_OBJBND))
                except Exception:
                    return
                if maximize:
                    if best_bd <= 0.0:
                        sign_proved["done"] = True
                        model.terminate()
                else:
                    if best_bd >= 0.0:
                        sign_proved["done"] = True
                        model.terminate()

            m.optimize(_sign_callback)
            if m.status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED):
                if verbose:
                    print(f"[g={g}] Gurobi status: {m.status}")
                raise RuntimeError(f'Gurobi failed to solve global MILP (status {m.status})')
            if sign_proved["done"]:
                return float(m.ObjBound)
            if m.status == GRB.OPTIMAL:
                try:
                    return float(m.objVal)
                except Exception:
                    return float(m.ObjBound)
            return float(m.ObjBound)

        start_g = time.time()
        max_val = _solve_maximize(True)
        min_val = _solve_maximize(False)
        elapsed_g = time.time() - start_g
        if verbose:
            print(f"[g={g}] result min={min_val:.6f} max={max_val:.6f} (elapsed={elapsed_g:.2f}s)")

        out[key] = (min_val, max_val)

        # merge and save incremental
        existing[key] = out[key]
        io_mod.save_bounds(str(out_path), existing, meta=dict(layer='global', sonar_number=sonar_number, fixed_zero=list(fixed_zero), time_limit=time_limit))

    # final save of global results
    out_path = out_dir / "global_bounds.npz"
    # existing dict contains mapping (cell,neuron)->(L,U)
    io_mod.save_bounds(str(out_path), existing if existing else {k: v for k, v in out.items()}, meta=dict(layer='global', checkpoint=str(checkpoint_path), created=time.asctime(), sonar_number=sonar_number))
    print(f"saved global bounds to {out_path}")
    return out_path


def main():
    p = argparse.ArgumentParser(description="Compute exact local/global bounds for a trainer checkpoint")
    p.add_argument("--model", required=True, help="Path to .pt checkpoint (state_dict or full model)")
    p.add_argument("--height", type=int, default=20)
    p.add_argument("--width", type=int, default=20)
    p.add_argument("--goal", type=int, default=399)
    p.add_argument("--sonar-number", type=int, default=None)
    p.add_argument("--unknown-number", type=int, default=0)
    p.add_argument("--prune-eps", type=float, default=0.0, help="Prune coefficients with abs < prune_eps to reduce MILP size")
    p.add_argument("--jobs", type=int, default=32)
    p.add_argument("--time-limit", type=float, default=60)
    p.add_argument("--layer", choices=["local", "global", "both"], default="both")
    p.add_argument("--obstacles", nargs="+", type=int, default=[
            43, 44, 24, 183, 184, 203, 204, 205,
            224, 225, 244, 245, 262, 263, 264, 265,
            285, 286, 287, 267, 132, 133, 152, 190,
            191, 211, 212, 213, 355, 376, 113, 112,
            94, 92, 91, 63, 64, 19, 18, 17, 16, 39,
            38, 37, 59, 231, 232, 233, 234, 251, 252,
            253, 271, 290, 291, 292, 309, 310, 329, 351
        ])
    p.add_argument("--verbose", action="store_true", help="Enable verbose logging for global stage")
    args = p.parse_args()

    checkpoint = Path(args.model)
    model_hash = io_mod.model_hash_from_file(checkpoint)
    out_dir = io_mod.model_output_dir(args.height, args.width, args.obstacles, None, model_hash)

    if args.layer in ("local", "both"):
        compute_local_bounds(checkpoint, args.height, args.width, args.goal, args.obstacles, detection_probability, args.sonar_number, args.unknown_number, args.prune_eps, args.time_limit, args.jobs, out_dir)
    if args.layer in ("global", "both"):
        # requires local bounds produced earlier
        local_bounds = out_dir / "local_bounds.npz"
        if not local_bounds.exists():
            raise FileNotFoundError(f"local bounds not found at {local_bounds}; run --layer local first")
        compute_global_bounds(checkpoint, args.height, args.width, args.goal, args.obstacles, detection_probability, args.sonar_number, args.unknown_number, args.prune_eps, args.jobs, args.time_limit, out_dir, str(local_bounds), args.verbose)


if __name__ == "__main__":
    main()

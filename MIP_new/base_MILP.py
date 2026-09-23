"""Shared MILP construction helpers for MIP_new.

Provides a builder for the full joint MILP (x and optional u) using
precomputed `c_z`/`const_z` and model weights.
"""
from __future__ import annotations

from typing import Dict, Tuple
import numpy as np


def _add_relu_bigM(m, z_var, a_var, L: float, U: float, name_prefix: str = None):
    # add ReLU encoding with Big-M binary when needed
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception:
        raise RuntimeError('Gurobi (gurobipy) is required')

    if not np.isfinite(L):
        L = -1e6
    if not np.isfinite(U):
        U = 1e6

    m.addConstr(z_var >= float(L))
    m.addConstr(z_var <= float(U))

    if U <= 0:
        m.addConstr(a_var == 0)
        return None
    if L >= 0:
        m.addConstr(a_var == z_var)
        return None

    y = m.addVar(vtype=GRB.BINARY, name=(f'{name_prefix}_y' if name_prefix else None))
    m.addConstr(a_var >= z_var)
    m.addConstr(a_var >= 0.0)
    m.addConstr(a_var <= float(U) * y)
    m.addConstr(a_var <= z_var - float(L) * (1 - y))
    return y


def _expand_bounds(L: float, U: float, scale: float = 1.01) -> Tuple[float, float]:
    if not np.isfinite(L):
        L = -1e6
    if not np.isfinite(U):
        U = 1e6
    center = 0.5 * (float(L) + float(U))
    half = 0.5 * (float(U) - float(L)) * float(scale)
    return center - half, center + half


def _map_output_index(Wg_out: np.ndarray, N: int, start_index: int):
    # map full grid index to model output index (handles reduced output sizes)
    model_out_size = int(Wg_out.shape[0])
    if model_out_size == N:
        return int(start_index)
    # boundary mapping fallback
    # create boundary list (square grid) conservative approach: map to 0 if mismatch
    try:
        # attempt a simple fallback mapping (map to 0)
        return 0
    except Exception:
        return 0


def build_joint_mip(data: Dict, *, start_index: int = 0, goal_idx: int = 0,
                    sonar_number: int = 8, unknown_number: int = 0,
                    fixed_zero=None, forbidden_radius: int = 6,
                    rows: int = 20, columns: int = 20,
                    prune_eps: float = 0.0, time_limit: float | None = None,
                    use_known_unknown_distinctions: bool = True,
                    mip_gap: float | None = None, verbose: bool = False):
    """Build a joint MILP with `x` (placements) and optional `u` (unknowns).

    `data` is the dict returned by `load_model_cz_bounds.load` and must
    contain at least `c_z` and `const_z`. Other entries (weights, local bounds)
    are used when available to tighten the model.

    Returns (model, x_vars, u_vars, z_out_var)
    """
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception:
        raise RuntimeError('Gurobi (gurobipy) is required')

    cz = np.asarray(data['c_z'])
    constz = np.asarray(data['const_z'])
    N = int(cz.shape[0]); H = int(cz.shape[1]); num_vars = int(cz.shape[2])

    W_out = data.get('W_out', None)
    b_out = data.get('b_out', None)
    if W_out is None and data.get('W_out_flat', None) is not None:
        W_out_flat = np.asarray(data.get('W_out_flat'))
        b_out_flat = np.asarray(data.get('b_out_flat')) if data.get('b_out_flat', None) is not None else np.zeros((W_out_flat.shape[0],), dtype=float)
        W_out = W_out_flat[:, np.newaxis, :]
        b_out = b_out_flat[:, np.newaxis]
    W_out_arr = np.asarray(W_out) if W_out is not None else None
    b_out_arr = np.asarray(b_out) if b_out is not None else None
    O = int(W_out_arr.shape[1]) if W_out_arr is not None and W_out_arr.ndim == 3 else 1
    Wg = data.get('Wg', None)
    bg = data.get('bg', None)
    Wg_out = data.get('Wg_out', None)
    b_gout = data.get('b_gout', None)

    L_arr = data.get('L_arr', None)
    U_arr = data.get('U_arr', None)

    fixed_zero = set(fixed_zero or [])

    m = gp.Model('MIP_joint_x_u')
    m.setParam('OutputFlag', 1 if verbose else 0)
    if time_limit is not None:
        m.setParam('TimeLimit', float(time_limit))
    if mip_gap is not None:
        m.setParam('MIPGap', float(mip_gap))

    # placement and unknown binaries
    x = m.addVars(num_vars, vtype=GRB.BINARY, name='x')
    for j in fixed_zero:
        if 0 <= int(j) < num_vars:
            m.addConstr(x[int(j)] == 0)

    # forbid placements within hex radius around start and goal (if requested)
    if forbidden_radius is not None and int(forbidden_radius) >= 0:
        def idx_to_cube(idx: int) -> np.ndarray:
            r = idx // columns
            c = idx % columns
            q = c - ((r - (r & 1)) // 2)
            x_c = int(q)
            z_c = int(r)
            y_c = -x_c - z_c
            return np.array((x_c, y_c, z_c), dtype=int)

        cube_coords = np.vstack([idx_to_cube(i) for i in range(N)])
        centers = [int(start_index), int(goal_idx)]
        forbidden_cells = set()
        for center in centers:
            if 0 <= center < N:
                cc = idx_to_cube(center)
                dists = np.max(np.abs(cube_coords - cc), axis=1)
                for i_cell, dist in enumerate(dists):
                    if int(dist) <= int(forbidden_radius):
                        forbidden_cells.add(int(i_cell))
        for cell in sorted(forbidden_cells):
            if 0 <= int(cell) < num_vars:
                m.addConstr(x[int(cell)] == 0)

    u = None
    use_unknown_vars = bool(use_known_unknown_distinctions) and unknown_number is not None and int(unknown_number) > 0
    if use_unknown_vars:
        u = m.addVars(num_vars, vtype=GRB.BINARY, name='u')
        for j in fixed_zero:
            if 0 <= int(j) < num_vars:
                m.addConstr(u[int(j)] == 0)
        # u <= x
        for j in range(num_vars):
            m.addConstr(u[j] <= x[j])
        m.addConstr(gp.quicksum(u[j] for j in range(num_vars)) == int(unknown_number))

    # cardinality for placements
    m.addConstr(gp.quicksum(x[j] for j in range(num_vars)) == int(sonar_number))

    # identify included local neurons by several signals to avoid pruning neurons
    # that still impact the final output via local_out and global layers.
    c_u_full = data.get('c_u', None) if use_known_unknown_distinctions else None
    c_u_arr = np.asarray(c_u_full) if c_u_full is not None else None

    include_by_cz = np.abs(np.sum(cz, axis=2)) > prune_eps
    include_by_cu = np.zeros((N, H), dtype=bool)
    if c_u_arr is not None:
        include_by_cu = np.abs(np.sum(c_u_arr, axis=2)) > prune_eps

    include_by_wout = np.zeros((N, H), dtype=bool)
    if W_out_arr is not None and W_out_arr.shape == (N, O, H):
        include_by_wout = np.any(np.abs(W_out_arr) > prune_eps, axis=1)

    include_by_global = np.zeros((N, H), dtype=bool)
    try:
        if Wg is not None and Wg_out is not None and W_out_arr is not None:
            Wg_arr = np.asarray(Wg)
            if Wg_arr.shape[1] < N * O:
                raise ValueError('global_hidden input dimension is smaller than local output block')
            Wg_local = Wg_arr[:, :N * O].reshape(Wg_arr.shape[0], N, O)
            Wg_out_arr = np.asarray(Wg_out)
            target_idx = _map_output_index(Wg_out_arr, N, int(start_index))
            # local_out coefficient per (cell i, output o) is:
            # sum_k Wg_out[target,k] * Wg_local[k,i,o]
            wgo_row = np.asarray(Wg_out_arr)[int(target_idx), :]
            local_out_coeffs = np.tensordot(wgo_row, Wg_local, axes=([0], [0]))
            if local_out_coeffs.shape == (N, O):
                # final coefficient on a_{i,h} = sum_o local_out_coeffs[i,o] * W_out[i,o,h]
                final_coef = np.einsum('io,ioh->ih', local_out_coeffs, W_out_arr)
                include_by_global = np.abs(final_coef) > prune_eps
    except Exception:
        include_by_global = np.zeros((N, H), dtype=bool)

    mask = include_by_cz | include_by_cu | include_by_wout | include_by_global
    included_local = [(int(i), int(h)) for i in range(N) for h in range(H) if mask[i, h]]
    # Prepare dictionaries
    z_vars = {}
    a_vars = {}

    # local activations
    for (i, h) in included_local:
        z = m.addVar(lb=-GRB.INFINITY, name=f'z_{i}_{h}')
        a = m.addVar(lb=0.0, name=f'a_{i}_{h}')
        # build linear expression for z = const + sum_j c_z * x_j + sum_j c_u * u_j
        coeffs_x = cz[i, h, :]
        nz_x = [j for j in range(num_vars) if abs(float(coeffs_x[j])) > prune_eps]
        expr = float(constz[i, h])
        if nz_x:
            expr = expr + gp.quicksum(float(coeffs_x[j]) * x[j] for j in nz_x)
        # optional coeffs for u
        c_u_full = data.get('c_u', None) if use_known_unknown_distinctions else None
        if c_u_full is not None and u is not None:
            coeffs_u = np.asarray(c_u_full)[i, h, :]
            nz_u = [j for j in range(num_vars) if abs(float(coeffs_u[j])) > prune_eps]
            if nz_u:
                expr = expr + gp.quicksum(float(coeffs_u[j]) * u[j] for j in nz_u)
        m.addConstr(z == expr)

        # ReLU encoding using local L/U when available
        L_orig = float(L_arr[i, h]) if L_arr is not None else -1e6
        U_orig = float(U_arr[i, h]) if U_arr is not None else 1e6
        L, U = _expand_bounds(L_orig, U_orig)

        # check warm start value
        warm_start = {178, 215, 256, 259, 295, 349, 369, 373}
        x_test = np.zeros(num_vars)
        for j in warm_start:
            x_test[j] = 1.0

        z_test = float(constz[i,h] + np.dot(cz[i,h,:], x_test))

        if z_test < L - 1e-8 or z_test > U + 1e-8:
            print(
                "BAD BOUND:",
                "cell=", i,
                "neuron=", h,
                "z=", z_test,
                "L=", L_orig,
                "U=", U_orig
            )

        _add_relu_bigM(m, z, a, L, U, name_prefix=f'loc_{i}_{h}')

        z_vars[(i, h)] = z
        a_vars[(i, h)] = a

    # local_out per cell
    local_out = {}
    if W_out_arr is not None and b_out_arr is not None:
        for i in range(N):
            for o in range(O):
                lo = m.addVar(lb=-GRB.INFINITY, name=f'local_out_{i}_{o}')
                # sum only over h that were included
                terms = []
                for h in range(H):
                    if (i, h) in a_vars:
                        coef = float(W_out_arr[i, o, h])
                        terms.append(coef * a_vars[(i, h)])
                if terms:
                    m.addConstr(lo == gp.quicksum(terms) + float(b_out_arr[i, o]))
                else:
                    m.addConstr(lo == float(b_out_arr[i, o]))
                local_out[(i, o)] = lo

    # global hidden units
    a_glob = {}
    z_glob = {}
    constant_term_total = 0.0
    if Wg is not None and bg is not None and Wg_out is not None and b_gout is not None:
        Wg_arr = np.asarray(Wg)
        if Wg_arr.shape[1] < N * O:
            raise ValueError('global_hidden input dimension is smaller than local output block')
        Wg_local = Wg_arr[:, :N * O].reshape(Wg_arr.shape[0], N, O)
        obst_num = float(len(fixed_zero))
        snum = float(sonar_number)
        if use_known_unknown_distinctions:
            unk = float(unknown_number)
            known = float(max(0.0, snum - unk))
        else:
            unk = 0.0
            known = 0.0

        # Compute per-neuron constant offset from global context features.
        # Wg layout: [:, :N*O] = local_out_flat, [:, N*O:N*O+N] = goal_one_hot,
        # [:, N*O+N:] = scalar globals.
        # Supported scalar tails:
        #   - [obstacle_number, sonar_number] (N+2 context)
        #   - [obstacle_number, sonar_number, start_as_known, start_as_unknown] (N+4 context)
        global_ctx_dim = int(Wg_arr.shape[1]) - N * O
        global_const_offset = np.zeros(int(Wg_arr.shape[0]), dtype=float)
        if global_ctx_dim >= N:
            goal_col = N * O + int(goal_idx)
            if 0 <= goal_col < int(Wg_arr.shape[1]):
                global_const_offset += Wg_arr[:, goal_col]
        scalar_count = max(0, int(global_ctx_dim - N))
        if scalar_count >= 2:
            scalar_start = N * O + N
            global_const_offset += Wg_arr[:, scalar_start + 0] * obst_num
            global_const_offset += Wg_arr[:, scalar_start + 1] * snum
        if scalar_count >= 4:
            scalar_start = N * O + N
            global_const_offset += Wg_arr[:, scalar_start + 2] * known
            global_const_offset += Wg_arr[:, scalar_start + 3] * unk

        target_out_idx = _map_output_index(np.asarray(Wg_out), N, int(start_index))

        for k in range(Wg_arr.shape[0]):
            # determine which local a_vars appear
            coef_ih = np.einsum('io,ioh->ih', Wg_local[k], W_out_arr) if W_out_arr is not None else np.zeros((N, H), dtype=float)
            local_terms = []
            for (i, h) in a_vars.keys():
                coef = float(coef_ih[i, h])
                if abs(coef) > prune_eps:
                    local_terms.append(((i, h), coef))
            if len(local_terms) == 0:
                # purely-constant contribution
                const_bout = float(np.sum(Wg_local[k] * b_out_arr)) if (Wg_local is not None and b_out_arr is not None) else 0.0
                zkg_const = float(bg[k]) + float(global_const_offset[k]) + const_bout
                act_const = float(max(0.0, zkg_const))
                constant_term_total += float(Wg_out[target_out_idx, k]) * act_const
                continue

            const_bout_k = float(np.sum(Wg_local[k] * b_out_arr)) if b_out_arr is not None else 0.0
            zkg = m.addVar(lb=-GRB.INFINITY, name=f'zg_{k}')
            ag = m.addVar(lb=0.0, name=f'ag_{k}')
            expr = gp.quicksum(float(coef) * a_vars[(i, h)] for ((i, h), coef) in local_terms) + float(bg[k]) + float(global_const_offset[k]) + const_bout_k
            m.addConstr(zkg == expr)

            # estimate bounds Lk/Uk conservatively from local bounds
            smin = 0.0; smax = 0.0
            for ((i, h), coef) in local_terms:
                if coef >= 0:
                    smin += coef * (L_arr[i, h] if L_arr is not None else -1e6)
                    smax += coef * (U_arr[i, h] if U_arr is not None else 1e6)
                else:
                    smin += coef * (U_arr[i, h] if U_arr is not None else 1e6)
                    smax += coef * (L_arr[i, h] if L_arr is not None else -1e6)

            # prefer precomputed global bounds when available
            G_L = data.get('G_L', None)
            G_U = data.get('G_U', None)
            if G_L is not None and G_U is not None and np.isfinite(G_L[k]) and np.isfinite(G_U[k]):
                Lk_orig = float(G_L[k])
                Uk_orig = float(G_U[k])
            else:
                Lk_orig = float(bg[k]) + float(global_const_offset[k]) + const_bout_k + smin
                Uk_orig = float(bg[k]) + float(global_const_offset[k]) + const_bout_k + smax
            Lk, Uk = _expand_bounds(Lk_orig, Uk_orig)
            _add_relu_bigM(m, zkg, ag, Lk, Uk, name_prefix=f'glob_{k}')

            z_glob[k] = zkg
            a_glob[k] = ag

        # final output
        terms = []
        for k, ag in a_glob.items():
            terms.append(float(Wg_out[target_out_idx, k]) * ag)
        z_out = m.addVar(lb=-GRB.INFINITY, name='z_out')
        if terms:
            m.addConstr(z_out == gp.quicksum(terms) + float(b_gout[target_out_idx]) + float(constant_term_total))
        else:
            m.addConstr(z_out == float(b_gout[target_out_idx]) + float(constant_term_total))
    else:
        # no global weights -> build simple aggregate of local outs if possible
        z_out = m.addVar(lb=-GRB.INFINITY, name='z_out')
        # fallback: sum local outputs at the start cell across output channels
        start_terms = [local_out[(int(start_index), o)] for o in range(O) if (int(start_index), o) in local_out]
        if start_terms:
            m.addConstr(z_out == gp.quicksum(start_terms))
        else:
            m.addConstr(z_out == 0.0)

    m.setObjective(z_out, GRB.MAXIMIZE)

    #"""
    # set warm start
    warm_start = {178, 215, 256, 259, 295, 349, 369, 373}
    for j in range(num_vars):
        if j in warm_start:
            x[j].start = 1.0
        else:
            x[j].start = 0.0
    #"""

    # set artificial upper bound on z_out to help Gurobi prune early (if known)
    #z_out_upper_bound = 13.32
    #m.addConstr(z_out <= z_out_upper_bound)

    return m, x, u, z_out


def make_incumbent_callback(data: Dict, xvars, uvars, z_out_var, start_index: int = 0, goal_idx: int = 0,
                            sonar_number: int = 8, unknown_number: int = 0, obstacle_count: int = 0,
                            use_known_unknown_distinctions: bool = True,
                            checkpoint_path: str | None = None,
                            rows: int = 20, columns: int = 20,
                            obstacle_indices=None,
                            verbose: bool = False):
    """Return a Gurobi callback that evaluates incumbents.

    The callback computes the model's output for the incumbent solution by
    reconstructing local preactivations/activations and the global layers
    using the precomputed `c_z/const_z` and checkpoint weights available
    in `data`. When `verbose` is True the callback prints placements,
    unknowns, model z_out (from Gurobi) and a recomputed z_out plus a few
    intermediate feature values.
    """
    import gurobipy as gp
    from gurobipy import GRB

    cz = np.asarray(data.get('c_z')) if 'c_z' in data else None
    constz = np.asarray(data.get('const_z')) if 'const_z' in data else None
    c_u = np.asarray(data.get('c_u')) if (use_known_unknown_distinctions and 'c_u' in data) else None
    W_out = np.asarray(data.get('W_out')) if 'W_out' in data else None
    b_out = np.asarray(data.get('b_out')) if 'b_out' in data else None
    if W_out is None and 'W_out_flat' in data:
        Wof_legacy = np.asarray(data.get('W_out_flat'))
        bof_legacy = np.asarray(data.get('b_out_flat')) if 'b_out_flat' in data else np.zeros((Wof_legacy.shape[0],), dtype=float)
        W_out = Wof_legacy[:, np.newaxis, :]
        b_out = bof_legacy[:, np.newaxis]
    Wg = np.asarray(data.get('Wg')) if 'Wg' in data else None
    bg = np.asarray(data.get('bg')) if 'bg' in data else None
    Wg_out = np.asarray(data.get('Wg_out')) if 'Wg_out' in data else None
    b_gout = np.asarray(data.get('b_gout')) if 'b_gout' in data else None

    num_vars = int(data.get('num_vars', 0))

    # Exact validation context matching trainer/test/validate_positioning.py.
    exact_ctx = {
        'initialized': False,
        'enabled': False,
        'reason': None,
        'device': None,
        'model': None,
        'meta': None,
        'static': None,
        'extract_local_features': None,
        'extract_global_context': None,
    }

    def _init_exact_validation():
        if exact_ctx['initialized']:
            return
        exact_ctx['initialized'] = True

        if checkpoint_path is None:
            exact_ctx['reason'] = 'checkpoint_path not provided'
            return

        try:
            import torch
            import dataset.static_cache as sc
            from environment.cost import CostMap
            from environment.features import Features
            from environment.utils import detection_probability
            from trainer.test.validate_positioning import (
                _build_model_from_checkpoint,
                _extract_local_features,
                _extract_global_context,
            )

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model_obj, meta = _build_model_from_checkpoint(checkpoint_path, int(columns), int(rows), device)
            static = sc.build_static(
                height=int(rows),
                width=int(columns),
                obstacle_indices=list(obstacle_indices or []),
                detection_probability=detection_probability,
            )

            exact_ctx['device'] = device
            exact_ctx['model'] = model_obj
            exact_ctx['meta'] = meta
            exact_ctx['static'] = static
            exact_ctx['CostMap'] = CostMap
            exact_ctx['Features'] = Features
            exact_ctx['detection_probability'] = detection_probability
            exact_ctx['extract_local_features'] = _extract_local_features
            exact_ctx['extract_global_context'] = _extract_global_context
            exact_ctx['enabled'] = True
        except Exception as ex:
            exact_ctx['reason'] = str(ex)

    def _compute_exact_validation_prediction(placed_cells, unknown_cells):
        _init_exact_validation()
        if not exact_ctx['enabled']:
            return None, None

        try:
            import numpy as _np
            import torch

            placed_sorted = [int(i) for i in sorted(placed_cells)]
            unknown_set = set(int(i) for i in unknown_cells)
            unknown_indices = [idx for idx, cell in enumerate(placed_sorted) if int(cell) in unknown_set]

            cost_map = exact_ctx['CostMap'](
                exact_ctx['static'].obstacle_map,
                exact_ctx['detection_probability'],
                sonar_number=len(placed_sorted),
                sonar_indices=placed_sorted,
                base_contributions=exact_ctx['static'].base_contributions,
            )
            features = exact_ctx['Features'](
                static=exact_ctx['static'],
                cost_map=cost_map,
                goal_index=int(goal_idx),
                start_as_unknown_number=len(unknown_indices),
                single_value=True,
                start_index=int(start_index),
                best_sonar_split=True,
            )

            # Match validate_positioning mask handling.
            features.known_sonar_mask.fill(False)
            features.unknown_sonar_mask.fill(False)
            sonar_array = _np.asarray(cost_map.sonar_indices, dtype=_np.int32)
            known_indices = [i for i in range(len(placed_sorted)) if i not in unknown_indices]
            if len(known_indices) > 0:
                features.known_sonar_mask[sonar_array[known_indices]] = True
            if len(unknown_indices) > 0:
                features.unknown_sonar_mask[sonar_array[unknown_indices]] = True

            features_dict = features.to_dict()
            meta = exact_ctx['meta']

            local_np = exact_ctx['extract_local_features'](
                features=features,
                features_dict=features_dict,
                total_cells=int(meta['total_cells']),
                local_input_dimension=int(meta['local_input_dimension']),
            )
            global_np = exact_ctx['extract_global_context'](
                features=features,
                features_dict=features_dict,
                cost_map=cost_map,
                goal_index=int(goal_idx),
                total_cells=int(meta['total_cells']),
                global_context_dimension=int(meta['global_context_dimension']),
            )

            local_t = torch.tensor(local_np, dtype=torch.float32, device=exact_ctx['device']).unsqueeze(0)
            global_t = torch.tensor(global_np, dtype=torch.float32, device=exact_ctx['device']).unsqueeze(0)

            with torch.no_grad():
                pred = exact_ctx['model'](local_t, global_t)

            pred_scalar = float(pred.detach().cpu().numpy().reshape(-1)[0])
            env_scalar = float(features.outputs[int(start_index)])
            return pred_scalar, env_scalar
        except Exception:
            return None, None

    def cb(model, where):
        if where != GRB.Callback.MIPSOL:
            return
        try:
            # extract incumbent x and u values
            x_vals = np.zeros(num_vars, dtype=float)
            for j in range(num_vars):
                try:
                    x_vals[j] = float(model.cbGetSolution(xvars[j]))
                except Exception:
                    x_vals[j] = 0.0

            u_vals = np.zeros(num_vars, dtype=float)
            if uvars is not None:
                for j in range(num_vars):
                    try:
                        u_vals[j] = float(model.cbGetSolution(uvars[j]))
                    except Exception:
                        u_vals[j] = 0.0

            placed = [int(i) for i in np.where(x_vals > 0.5)[0]]
            unknowns = [int(i) for i in np.where(u_vals > 0.5)[0]] if uvars is not None else []

            # attempt numeric z_out from model
            try:
                z_out_model = float(model.cbGetSolution(z_out_var))
            except Exception:
                z_out_model = None

            # recompute network quantities when possible
            z_out_calc = None
            z_local = None
            a_local = None
            local_out = None
            zkg = None
            a_glob = None

            if cz is not None and constz is not None:
                try:
                    # cz: (N, H, K)
                    z_local = np.asarray(constz, dtype=float) + np.tensordot(np.asarray(cz, dtype=float), x_vals, axes=([2], [0]))
                    if c_u is not None and uvars is not None:
                        z_local = z_local + np.tensordot(np.asarray(c_u, dtype=float), u_vals, axes=([2], [0]))
                    a_local = np.maximum(z_local, 0.0)

                    # local_out per cell
                    if W_out is not None and b_out is not None:
                        W_out_cb = np.asarray(W_out, dtype=float)
                        b_out_cb = np.asarray(b_out, dtype=float)
                        local_out = np.einsum('ioh,ih->io', W_out_cb, a_local) + b_out_cb
                    else:
                        # fallback: local_out as zeros
                        local_out = np.zeros((z_local.shape[0], 1), dtype=float)

                    # global
                    if Wg is not None and bg is not None:
                        Wg_cb = np.asarray(Wg, dtype=float)
                        N_cb = int(local_out.shape[0])
                        O_cb = int(local_out.shape[1])
                        Wg_local = Wg_cb[:, :N_cb * O_cb].reshape(Wg_cb.shape[0], N_cb, O_cb)
                        zkg_pre = np.asarray(bg, dtype=float) + np.einsum('kio,io->k', Wg_local, local_out)
                        # add global context contributions (goal one-hot + scalar globals)
                        ctx_dim = int(Wg_cb.shape[1]) - N_cb * O_cb
                        if ctx_dim >= N_cb:
                            goal_col = N_cb * O_cb + int(goal_idx)
                            if 0 <= goal_col < int(Wg_cb.shape[1]):
                                zkg_pre = zkg_pre + Wg_cb[:, goal_col]
                        scalar_count = max(0, int(ctx_dim - N_cb))
                        if scalar_count >= 2:
                            sc = N_cb * O_cb + N_cb
                            snum_cb = float(sonar_number)
                            if use_known_unknown_distinctions:
                                unk_cb = float(unknown_number)
                                known_cb = float(max(0.0, snum_cb - unk_cb))
                            else:
                                unk_cb = 0.0
                                known_cb = 0.0
                            zkg_pre = zkg_pre + Wg_cb[:, sc + 0] * float(obstacle_count)
                            zkg_pre = zkg_pre + Wg_cb[:, sc + 1] * snum_cb
                        if scalar_count >= 4:
                            sc = N_cb * O_cb + N_cb
                            zkg_pre = zkg_pre + Wg_cb[:, sc + 2] * known_cb
                            zkg_pre = zkg_pre + Wg_cb[:, sc + 3] * unk_cb
                        zkg = zkg_pre
                        a_glob = np.maximum(zkg, 0.0)
                        if Wg_out is not None and b_gout is not None:
                            target_idx = _map_output_index(np.asarray(Wg_out, dtype=float), N_cb, int(start_index))
                            z_out_calc = float(np.dot(np.asarray(Wg_out, dtype=float)[target_idx, :], a_glob) + float(np.asarray(b_gout, dtype=float)[target_idx]))
                        else:
                            # fallback
                            if 0 <= int(start_index) < local_out.shape[0]:
                                z_out_calc = float(np.sum(local_out[int(start_index), :]))
                    else:
                        if 0 <= int(start_index) < local_out.shape[0]:
                            z_out_calc = float(np.sum(local_out[int(start_index), :]))
                except Exception as ex:
                    if verbose:
                        print('Incumbent callback compute error:', ex)

            if verbose:
                if z_out_calc is not None:
                    print('z_out (c_z_recalc) =', z_out_calc)
                else:
                    print('z_out (c_z_recalc) = unavailable')
                exact_pred, env_output = _compute_exact_validation_prediction(placed, unknowns)
                if exact_pred is not None:
                    print('z_out (pt_forward) =', exact_pred)
                elif exact_ctx.get('reason'):
                    print('z_out (pt_forward) = unavailable:', exact_ctx['reason'])
                else:
                    print('z_out (pt_forward) = unavailable')

        except Exception as e:
            if verbose:
                print('Incumbent callback unexpected error:', e)

    return cb
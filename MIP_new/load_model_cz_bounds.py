"""Load model parameters and precomputed c_z/const_z and local bounds.

This helper provides a minimal, robust loader used by the MIP_new builders.
"""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np


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


def load(checkpoint_path: str | Path | None = None, *,
         precompute_dir: str | Path | None = None,
         local_bounds_npz: str | Path | None = None,
         include_mask_features: bool = True,
         verbose: bool = False):
    """Return a dict with keys used by the MILP builders.

    Keys returned (when available):
      - c_z, const_z (numpy arrays)
      - L_arr, U_arr (local bounds arrays)
            - W_out, b_out, Wg, bg, Wg_out, b_gout
      - N, H, G, num_vars
    """
    out = {}
    # load precomputes if provided
    if precompute_dir is not None:
        p = Path(precompute_dir)
        czp = p / "c_z.npy"
        constp = p / "const_z.npy"
        if czp.exists() and constp.exists():
            out['c_z'] = np.load(czp, mmap_mode='r')
            out['const_z'] = np.load(constp, mmap_mode='r')
            if verbose:
                print(f"Loaded c_z/const_z from {p}")

        # try to read precompute meta to find checkpoint if none provided
        meta_p = p / 'precompute.meta.json'
        if checkpoint_path is None and meta_p.exists():
            try:
                with open(meta_p, 'r') as mf:
                    meta = json.load(mf)
                ck = meta.get('checkpoint', None)
                if ck is not None:
                    # resolve relative path to workspace root
                    ck_path = Path(ck)
                    if not ck_path.exists():
                        # try relative to precompute dir parent or workspace
                        alt = p.parent / ck
                        if alt.exists():
                            ck_path = alt
                    if ck_path.exists():
                        checkpoint_path = str(ck_path)
                        if verbose:
                            print(f"Using checkpoint from precompute.meta.json: {checkpoint_path}")
            except Exception:
                pass
        # try load global bounds saved by bounds_new
        gbp = p / 'global_bounds.npz'
        if gbp.exists():
            try:
                out['global_bounds_data'] = np.load(gbp, allow_pickle=True)
                if verbose:
                    print(f"Loaded global_bounds from {gbp}")
            except Exception:
                if verbose:
                    print('Failed to load global_bounds.npz')

    # load local bounds if provided
    if local_bounds_npz is not None:
        d = np.load(local_bounds_npz, allow_pickle=True)
        out['local_bounds_data'] = d
        # try to shape into L/U arrays later when N/H known

    # load model parameters if checkpoint provided
    if checkpoint_path is not None:
        try:
            import torch
        except Exception:
            torch = None
        if torch is None:
            raise RuntimeError('PyTorch is required to load model checkpoint')
        ck = torch.load(str(checkpoint_path), map_location='cpu')
        if isinstance(ck, dict) and ("state_dict" in ck or "model_state_dict" in ck):
            state = ck.get('state_dict', ck.get('model_state_dict', ck))
        elif isinstance(ck, dict):
            state = ck
        else:
            try:
                state = ck.state_dict()
            except Exception:
                state = {}

        # attempt to extract local_out weights and global weights
        try:
            W_out_t = _find_state_param(state, 'local_output_weight')
            b_out_t = _find_state_param(state, 'local_output_bias')
            W_out = np.asarray(W_out_t.cpu().numpy())
            b_out = np.asarray(b_out_t.cpu().numpy())
            out['W_out'] = W_out
            out['b_out'] = b_out
            out['local_output_dim'] = int(W_out.shape[1])
            # Backward-compatible aliases used by older code paths.
            out['W_out_flat'] = W_out[:, 0, :]
            out['b_out_flat'] = b_out[:, 0]
        except Exception:
            # try legacy names
            try:
                W_out_t = _find_state_param(state, 'local_out_weight')
                b_out_t = _find_state_param(state, 'local_out_bias')
                W_out = np.asarray(W_out_t.cpu().numpy())
                b_out = np.asarray(b_out_t.cpu().numpy())
                out['W_out'] = W_out
                out['b_out'] = b_out
                out['local_output_dim'] = int(W_out.shape[1])
                # Backward-compatible aliases used by older code paths.
                out['W_out_flat'] = W_out[:, 0, :]
                out['b_out_flat'] = b_out[:, 0]
            except Exception:
                if verbose:
                    print('Warning: failed to load local_out weights from checkpoint')

        # global layer
        try:
            Wg = _find_state_param(state, 'global_hidden.weight')
            bg = _find_state_param(state, 'global_hidden.bias')
            out['Wg'] = np.asarray(Wg.cpu().numpy())
            out['bg'] = np.asarray(bg.cpu().numpy())
        except Exception:
            # legacy names
            try:
                Wg = _find_state_param(state, 'global_fc1.weight')
                bg = _find_state_param(state, 'global_fc1.bias')
                out['Wg'] = np.asarray(Wg.cpu().numpy())
                out['bg'] = np.asarray(bg.cpu().numpy())
            except Exception:
                if verbose:
                    print('Warning: failed to load global hidden weights from checkpoint')

        try:
            Wg_out = _find_state_param(state, 'global_output.weight')
            b_gout = _find_state_param(state, 'global_output.bias')
            out['Wg_out'] = np.asarray(Wg_out.cpu().numpy())
            out['b_gout'] = np.asarray(b_gout.cpu().numpy())
        except Exception:
            # legacy names
            try:
                Wg_out = _find_state_param(state, 'global_out.weight')
                b_gout = _find_state_param(state, 'global_out.bias')
                out['Wg_out'] = np.asarray(Wg_out.cpu().numpy())
                out['b_gout'] = np.asarray(b_gout.cpu().numpy())
            except Exception:
                if verbose:
                    print('Warning: failed to load global output weights from checkpoint')

        # try to extract local hidden weights (used to compute known/unknown mask contributions)
        try:
            W_loc_t = _find_state_param(state, 'local_hidden_weight')
            B_loc_t = _find_state_param(state, 'local_hidden_bias')
            out['W_loc'] = np.asarray(W_loc_t.cpu().numpy())
            out['B_loc'] = np.asarray(B_loc_t.cpu().numpy())
        except Exception:
            # legacy name fallback
            try:
                W_loc_t = _find_state_param(state, 'local_fc1_weight')
                B_loc_t = _find_state_param(state, 'local_fc1_bias')
                out['W_loc'] = np.asarray(W_loc_t.cpu().numpy())
                out['B_loc'] = np.asarray(B_loc_t.cpu().numpy())
            except Exception:
                if verbose:
                    print('Warning: failed to load local hidden weights from checkpoint')

    # derive shapes if possible
    if 'c_z' in out:
        cz = np.asarray(out['c_z'])
        out['N'] = int(cz.shape[0])
        out['H'] = int(cz.shape[1])
        out['num_vars'] = int(cz.shape[2])
    else:
        # try to derive from local output weights
        if 'W_out' in out:
            Wof = np.asarray(out['W_out'])
            out['N'] = int(Wof.shape[0])
            out['H'] = int(Wof.shape[2])
            out['local_output_dim'] = int(Wof.shape[1])
        elif 'W_out_flat' in out:
            Wof = np.asarray(out['W_out_flat'])
            out['N'] = int(Wof.shape[0])
            out['H'] = int(Wof.shape[1])
            out['local_output_dim'] = 1
    # materialize local bounds arrays if local_bounds supplied
    if 'local_bounds_data' in out and 'N' in out and 'H' in out:
        d = out['local_bounds_data']
        L_arr = np.full((out['N'], out['H']), -1e6, dtype=float)
        U_arr = np.full((out['N'], out['H']), 1e6, dtype=float)
        if 'cell' in d and 'neuron' in d and 'L' in d and 'U' in d:
            for c, n, l, u in zip(d['cell'], d['neuron'], d['L'], d['U']):
                ci = int(c); hi = int(n)
                if 0 <= ci < out['N'] and 0 <= hi < out['H']:
                    L_arr[ci, hi] = float(l); U_arr[ci, hi] = float(u)
            out['L_arr'] = L_arr
            out['U_arr'] = U_arr

    # materialize global bounds arrays if global_bounds supplied and Wg present
    if 'global_bounds_data' in out and ('Wg' in out or 'Wg_out' in out):
        try:
            gd = out['global_bounds_data']
            # determine G from Wg or Wg_out
            if 'Wg' in out:
                G = int(out['Wg'].shape[0])
            elif 'Wg_out' in out:
                # if only Wg_out available, try to infer G from its second dim
                Wgo = np.asarray(out['Wg_out'])
                G = int(Wgo.shape[1]) if Wgo.ndim > 1 else int(Wgo.shape[0])
            else:
                G = None
            if G is not None:
                G_L = np.full((G,), -1e6, dtype=float)
                G_U = np.full((G,), 1e6, dtype=float)
                if 'cell' in gd and 'neuron' in gd and 'L' in gd and 'U' in gd:
                    for c, n, l, u in zip(gd['cell'], gd['neuron'], gd['L'], gd['U']):
                        if int(c) == -1:
                            ni = int(n)
                            if 0 <= ni < G:
                                G_L[ni] = float(l); G_U[ni] = float(u)
                out['G_L'] = G_L
                out['G_U'] = G_U
                if verbose:
                    print('Prepared G_L/G_U from global_bounds.npz')
        except Exception:
            if verbose:
                print('Failed to prepare global bounds arrays')

    # If requested, derive known/unknown mask contributions from local layer weights.
    # When include_mask_features=False, c_z is left unchanged and c_u is omitted.
    if include_mask_features and 'W_loc' in out and 'c_z' in out:
        try:
            W_loc = np.asarray(out['W_loc'])  # shape (N, H, K)
            cz_arr = np.asarray(out['c_z']).copy()
            N, H, Kx = W_loc.shape[0], W_loc.shape[1], W_loc.shape[2]
            # infer R from local input length K where K = 34 + 5*R
            K = int(Kx)
            R = None
            if K >= 34 and (K - 34) % 5 == 0:
                R = (K - 34) // 5
            # compute local feature slices to locate known/unknown mask indices
            def _local_feature_slices(R_val: int):
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
                slices['ring_cost'] = (pos, pos + R_val); pos += R_val
                slices['ring_obstacle_count'] = (pos, pos + R_val); pos += R_val
                slices['ring_obstacle_density'] = (pos, pos + R_val); pos += R_val
                slices['ring_sonar_count'] = (pos, pos + R_val); pos += R_val
                slices['ring_sonar_density'] = (pos, pos + R_val); pos += R_val
                return slices

            if R is not None:
                slices = _local_feature_slices(int(R))
                known_idx = slices['known_sonar_mask'][0]
                unknown_idx = slices['unknown_sonar_mask'][0]
                # prepare c_u array (same shape as c_z)
                c_u = np.zeros_like(cz_arr, dtype=float)
                # add wk to placement self-effect and compute c_u self-effect = wu - wk
                for i in range(min(cz_arr.shape[0], W_loc.shape[0])):
                    for h in range(min(cz_arr.shape[1], W_loc.shape[1])):
                        wk = float(W_loc[i, h, known_idx])
                        wu = float(W_loc[i, h, unknown_idx])
                        # add wk to placement column i
                        if i < cz_arr.shape[2]:
                            cz_arr[i, h, i] = float(cz_arr[i, h, i]) + wk
                            c_u[i, h, i] = float(wu - wk)
                out['c_z'] = cz_arr
                out['c_u'] = c_u
                if verbose:
                    print('Computed c_u and adjusted c_z with known-mask weights')
        except Exception:
            if verbose:
                print('Failed to compute c_u from W_loc')

    return out

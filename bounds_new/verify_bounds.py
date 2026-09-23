# bounds_new/verify_bounds.py
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from bounds_new import solver as solver_mod
import dataset.static_cache as sc
from environment.utils import detection_probability

def find_param(state, name):
    if name in state: return state[name]
    m = "module." + name
    if m in state: return state[m]
    for k in state.keys():
        if k.endswith(name) or name in k:
            return state[k]
    raise KeyError(name)

def local_feature_slices(R):
    pos = 0; s = {}
    s['cost'] = (pos,pos+1); pos+=1
    s['obstacle_placement'] = (pos,pos+1); pos+=1
    s['sonar_placement'] = (pos,pos+1); pos+=1
    s['axial_coordinates'] = (pos,pos+2); pos+=2
    s['cartesian_coordinates'] = (pos,pos+2); pos+=2
    s['distance_to_goal'] = (pos,pos+1); pos+=1
    s['known_sonar_mask'] = (pos,pos+1); pos+=1
    s['unknown_sonar_mask'] = (pos,pos+1); pos+=1
    s['neighbor_cost'] = (pos,pos+6); pos+=6
    s['neighbor_obstacle'] = (pos,pos+6); pos+=6
    s['neighbor_sonar'] = (pos,pos+6); pos+=6
    s['neighbor_mask'] = (pos,pos+6); pos+=6
    # rings (5 groups of R)
    for name in ('ring_cost','ring_obstacle_count','ring_obstacle_density','ring_sonar_count','ring_sonar_density'):
        s[name] = (pos,pos+R); pos += R
    return s

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--model-dir", required=True, help="tight_bounds/model_<hash> directory")
    p.add_argument("--height", type=int, default=20)
    p.add_argument("--width", type=int, default=20)
    p.add_argument("--goal", type=int, default=399)
    p.add_argument("--sonar-number", type=int, default=8)
    p.add_argument("--unknown-number", type=int, default=0)
    p.add_argument("--prune-eps", type=float, default=0.0)
    p.add_argument("--n-local-checks", type=int, default=2)
    p.add_argument("--n-global-samples", type=int, default=5)
    p.add_argument("--time-limit", type=float, default=10.0)
    p.add_argument("--dataset-dir", type=str, default=None, help="optional dataset parts directory to fully validate bounds")
    p.add_argument("--max-samples", type=int, default=0, help="maximum samples to check from the dataset (0 => all)")
    p.add_argument("--tol", type=float, default=1e-6, help="tolerance when checking bounds")
    p.add_argument("--bounds-scale", type=float, default=1.01, help="Multiply stored bounds (L and U) by this factor before checking")
    args = p.parse_args()

    model_dir = Path(args.model_dir)
    ckpt = Path(args.model)
    # load precomputes and bounds
    c_z = np.load(model_dir/"c_z.npy")
    const_z = np.load(model_dir/"const_z.npy")
    local_npz = np.load(model_dir/"local_bounds.npz", allow_pickle=True)
    print("Loaded local_bounds:", list(local_npz.keys()))
    # optionally scale loaded bounds
    if float(args.bounds_scale) != 1.0:
        local_npz = dict(local_npz)
        L_orig = np.asarray(local_npz['L'], dtype=float)
        U_orig = np.asarray(local_npz['U'], dtype=float)
        local_npz['L'] = ((1 + float(args.bounds_scale)) * L_orig + (1 - float(args.bounds_scale)) * U_orig) / 2.0
        local_npz['U'] = ((1 - float(args.bounds_scale)) * L_orig + (1 + float(args.bounds_scale)) * U_orig) / 2.0
        print(f"Scaled local bounds by {args.bounds_scale}")
    # basic sanity
    L = local_npz['L']; U = local_npz['U']
    print("sanity: L<=U violations:", int(np.sum(L>U)), "NaNs L/U:", int(np.sum(np.isnan(L))+np.sum(np.isnan(U))))
    # load model state
    raw = torch.load(str(ckpt), map_location="cpu")
    if isinstance(raw, dict) and ("state_dict" in raw or "model_state_dict" in raw):
        state = raw.get("state_dict", raw.get("model_state_dict", raw))
    elif isinstance(raw, dict):
        state = raw
    else:
        state = raw.state_dict()
    W_loc = find_param(state, "local_hidden_weight").cpu().numpy()
    # static
    static = sc.build_static(height=args.height, width=args.width, obstacle_indices=None, detection_probability=detection_probability)
    N = int(static.raw_map.total_cells)
    R = int(static.ring_obstacle_count.shape[1])
    slices = local_feature_slices(R)
    known_idx = slices['known_sonar_mask'][0]
    unknown_idx = slices['unknown_sonar_mask'][0]
    fixed_zero = set(int(x) for x in static.obstacles)
    num_vars = int(c_z.shape[2])

    # exact re-solve a random sample of local neurons
    idxs = np.arange(len(local_npz['cell']))
    np.random.shuffle(idxs)
    print("Running exact MILP re-solves for", args.n_local_checks, "local neurons (may call Gurobi)...")
    nbad = 0
    for ii in idxs[:args.n_local_checks]:
        i = int(local_npz['cell'][ii]); h = int(local_npz['neuron'][ii])
        stored_L, stored_U = float(local_npz['L'][ii]), float(local_npz['U'][ii])
        coeff = np.asarray(c_z[i,h,:], dtype=float).copy()
        const = float(const_z[i,h])
        w = np.asarray(W_loc[i,h,:], dtype=float)
        wk = float(w[known_idx]); wu = float(w[unknown_idx])
        if i < coeff.size:
            coeff[i] = coeff[i] + wk
        coeff_u = None
        if args.unknown_number and args.unknown_number > 0:
            coeff_u = np.zeros_like(coeff); coeff_u[i] = wu - wk
        var_idx = np.nonzero(np.abs(coeff) > args.prune_eps)[0].tolist()
        if args.sonar_number is not None and len(var_idx) < int(args.sonar_number):
            var_idx = list(range(num_vars))
        L2, U2 = solver_mod.solve_min_max(coeff[var_idx], coeff_u[var_idx] if coeff_u is not None else None,
                                          const, var_idx, num_vars,
                                          sonar_number=args.sonar_number,
                                          fixed_zero=fixed_zero,
                                          unknown_count=args.unknown_number,
                                          time_limit=args.time_limit,
                                          verbose=False)
        if abs(L2-stored_L) > 1e-6 or abs(U2-stored_U) > 1e-6:
            print("Mismatch local (cell,neuron)=", (i,h), "stored=({:.6g},{:.6g}) recomputed=({:.6g},{:.6g})".format(stored_L,stored_U,L2,U2))
            nbad += 1
    print("local re-solve mismatches:", nbad, "out of", min(args.n_local_checks, len(idxs)))

    # quick sample check for global neurons
    if (model_dir/"global_bounds.npz").exists():
        g_npz = np.load(model_dir/"global_bounds.npz", allow_pickle=True)
        print("Loaded global_bounds:", list(g_npz.keys()))
        # optionally scale stored global bounds
        if float(args.bounds_scale) != 1.0:
            g_npz = dict(g_npz)
            if 'L' in g_npz and 'U' in g_npz:
                g_npz['L'] = np.asarray(g_npz['L'], dtype=float) * float(args.bounds_scale)
                g_npz['U'] = np.asarray(g_npz['U'], dtype=float) * float(args.bounds_scale)
                print(f"Scaled global bounds by {args.bounds_scale}")
        # load model params needed
        W_out_t = find_param(state, "local_output_weight").cpu().numpy()
        b_out_t = find_param(state, "local_output_bias").cpu().numpy()
        N = int(c_z.shape[0])
        H = int(W_loc.shape[1])
        O = int(W_out_t.shape[1])
        Wout = W_out_t
        bout = b_out_t
        Wg = find_param(state, "global_hidden.weight").cpu().numpy()
        bg = find_param(state, "global_hidden.bias").cpu().numpy()
        N = int(c_z.shape[0]); H = int(W_loc.shape[1]); G = int(Wg.shape[0])
        # pick up stored global bounds indexes
        cell_arr = g_npz['cell']; neuron_arr = g_npz['neuron']; Lg_arr = g_npz['L']; Ug_arr = g_npz['U']
        # Determine global slices and helper arrays
        local_flat_dim = N * O

        Wg_local = Wg[:, :local_flat_dim].reshape(G, N, O)

        Wg_goal = Wg[:, local_flat_dim:local_flat_dim+N]
        Wg_obst = Wg[:, local_flat_dim+N+0]
        Wg_sonar = Wg[:, local_flat_dim+N+1]
        Wg_start_known = Wg[:, local_flat_dim+N+2]
        Wg_start_unknown = Wg[:, local_flat_dim+N+3]

        # If a dataset directory is provided, run a full check over dataset samples
        if args.dataset_dir:
            ddir = Path(args.dataset_dir)
            print("Loading dataset for full-bounds validation from:", ddir)
            sonar_placement = np.load(ddir / "sonar_placement.npy", allow_pickle=True)
            goal_onehot = np.load(ddir / "goal_onehot.npy", allow_pickle=True)
            known_mask_arr = np.load(ddir / "known_sonar_mask.npy", allow_pickle=True)
            unknown_mask_arr = np.load(ddir / "unknown_sonar_mask.npy", allow_pickle=True)
            obstacle_number_arr = np.load(ddir / "obstacle_number.npy", allow_pickle=True)
            sonar_number_arr = np.load(ddir / "sonar_number.npy", allow_pickle=True)
            start_known_arr = np.load(ddir / "start_as_known_number.npy", allow_pickle=True)
            start_unknown_arr = np.load(ddir / "start_as_unknown_number.npy", allow_pickle=True)

            nsamples = int(sonar_placement.shape[0])
            max_samples = int(args.max_samples) if args.max_samples and args.max_samples > 0 else nsamples
            max_samples = min(max_samples, nsamples)

            # local bound arrays
            local_cell = local_npz['cell']; local_neuron = local_npz['neuron']; local_L = local_npz['L']; local_U = local_npz['U']

            # global stored entries where cell == -1 correspond to global hidden neurons
            g_cell = cell_arr; g_neuron = neuron_arr; g_L = Lg_arr; g_U = Ug_arr
            g_mask = (g_cell == -1)
            g_indices = g_neuron[g_mask]
            g_stored_L = g_L[g_mask]; g_stored_U = g_U[g_mask]

            tot_local_viol = 0; tot_global_viol = 0
            samples_with_local_viol = 0; samples_with_global_viol = 0

            print(f"Checking {max_samples}/{nsamples} samples (tol={args.tol})...")
            for si in range(max_samples):
                x = sonar_placement[si].astype(float)
                # compute local preactivations
                z = const_z + np.tensordot(c_z, x, axes=(2,0))  # (N,H)
                # incorporate known/unknown mask feature contributions from model weights
                try:
                    km = known_mask_arr[si].astype(float).reshape((N, 1))
                    um = unknown_mask_arr[si].astype(float).reshape((N, 1))
                    wk_mat = W_loc[:, :, known_idx]
                    wu_mat = W_loc[:, :, unknown_idx]
                    z = z + wk_mat * km + wu_mat * um
                except Exception:
                    # if masks not available or shapes mismatch, skip this augmentation
                    pass

                # check local stored bounds (vectorized)
                z_entries = z[local_cell, local_neuron]
                viol_local = (z_entries < (local_L - args.tol)) | (z_entries > (local_U + args.tol))
                if np.any(viol_local):
                    cnt = int(np.sum(viol_local))
                    tot_local_viol += cnt
                    samples_with_local_viol += 1
                    # print up to first few violations for this sample
                    idxs = np.nonzero(viol_local)[0][:10]
                    for j in idxs:
                        ci = int(local_cell[j]); nh = int(local_neuron[j])
                        print(f"Sample {si} LOCAL violation cell={ci} neuron={nh} val={float(z[ci,nh]):.6g} stored=({float(local_L[j]):.6g},{float(local_U[j]):.6g})")

                # compute local outputs and global preactivations
                a = np.maximum(z, 0.0)
                local_out = np.einsum("nh,noh->no", a, Wout) + bout  # (N,O)
                local_out_flat = local_out.reshape(-1)
                goal_vec = goal_onehot[si].astype(float)
                goal_contrib = Wg_goal.dot(goal_vec)
                global_pre = bg + np.einsum("gno,no->g", Wg_local, local_out) + goal_contrib + Wg_obst * float(obstacle_number_arr[si])
                global_pre = global_pre + Wg_sonar * float(sonar_number_arr[si]) + Wg_start_known * float(start_known_arr[si]) + Wg_start_unknown * float(start_unknown_arr[si])

                if g_mask.any():
                    g_vals = global_pre[g_indices]
                    viol_g = (g_vals < (g_stored_L - args.tol)) | (g_vals > (g_stored_U + args.tol))
                    if np.any(viol_g):
                        cntg = int(np.sum(viol_g))
                        tot_global_viol += cntg
                        samples_with_global_viol += 1
                        idxs2 = np.nonzero(viol_g)[0][:10]
                        for jj in idxs2:
                            gg = int(g_indices[jj])
                            print(f"Sample {si} GLOBAL violation g={gg} val={float(global_pre[gg]):.6g} stored=({float(g_stored_L[jj]):.6g},{float(g_stored_U[jj]):.6g})")

                if si % 1000 == 0 and si > 0:
                    print(f"Processed {si}/{max_samples} samples...")

            print(f"Dataset check complete. samples checked={max_samples}, samples w/local_viol={samples_with_local_viol}, total_local_viol={tot_local_viol}, samples w/global_viol={samples_with_global_viol}, total_global_viol={tot_global_viol}")

        else:
            def find_global(g):
                mask = (cell_arr == -1) & (neuron_arr == g)
                if not np.any(mask): return None
                idx = np.nonzero(mask)[0][0]; return float(Lg_arr[idx]), float(Ug_arr[idx])
            # sample a few g's (first 5 or less)
            to_check = list(range(min(5, G)))
            allowed = [j for j in range(num_vars) if j not in fixed_zero]
            print("Running sample test for global neurons (samples per g):", args.n_global_samples)
            for g in to_check:
                found = find_global(g)
                if found is None:
                    print("no stored record for global neuron", g); continue
                Ls, Us = found
                # prepare coef_ih and const_global
                coef_ih = np.einsum("io,ioh->ih", Wg_local[g], W_out_t)  # shape (N,H)
                const_global += float(np.sum(Wg_local[g] * b_out_t))
                const_global += float(Wg_goal[g, args.goal])
                obst_num = float(len(static.obstacles)); snum = float(0 if args.sonar_number is None else int(args.sonar_number))
                unk = float(0 if args.unknown_number is None else int(args.unknown_number)); known = float(snum - unk)
                const_global += float(Wg_obst[g] * obst_num)
                const_global += float(Wg_sonar[g] * snum)
                const_global += float(Wg_start_known[g] * known)
                const_global += float(Wg_start_unknown[g] * unk)
                violations = 0
                for _ in range(args.n_global_samples):
                    if args.sonar_number is None:
                        k = np.random.randint(0, len(allowed))
                        sel = np.random.choice(allowed, k, replace=False)
                    else:
                        sel = np.random.choice(allowed, int(args.sonar_number), replace=False)
                    x = np.zeros(num_vars, dtype=float); x[sel] = 1.0
                    # compute local preacts and ReLU
                    z = const_z + np.tensordot(c_z, x, axes=(2,0))  # z shape (N,H)
                    a = np.maximum(z, 0.0)
                    global_val = const_global + float(np.sum(coef_ih * a))
                    if not (Ls - 1e-6 <= global_val <= Us + 1e-6):
                        violations += 1
                print(f"g={g} sampled violations {violations}/{args.n_global_samples} (stored L,U=({Ls:.6g},{Us:.6g}))")
    else:
        print("no global_bounds.npz found; skip global check")

if __name__ == "__main__":
    main()

    #usage: python bounds_new/verify_bounds.py --model best_model_mse.pt --model-dir tight_bounds/model_<hash> --height 20 --width 20 --goal 399 --sonar-number 8 --unknown-number 2
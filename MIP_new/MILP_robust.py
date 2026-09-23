"""Robust solve via alternating master/subproblem (heuristic iterate).

This implements a simple alternating optimization:
  - fix `u`, optimize `x` (maximization)
  - fix `x`, optimize `u` (minimization)
Repeat until convergence or timeout. Returns best robust (maximin) solution found.
"""
from __future__ import annotations

from pathlib import Path
import time
import numpy as np

from . import load_model_cz_bounds as loader
from . import base_MILP as base


def robust_solve(checkpoint: str | None, precompute_dir: str | None, local_bounds: str | None,
                 start_index: int, goal_idx: int, sonar_number: int, unknown_number: int,
                 fixed_zero=None, forbidden_radius: int = 3, rows: int = 20, columns: int = 20,
                 prune_eps: float = 1e-8,
                 max_iters: int = 10, time_limit: float | None = None,
                 per_solve_time_limit: float | None = None, verbose: bool = False):
    data = loader.load(checkpoint, precompute_dir=precompute_dir, local_bounds_npz=local_bounds, verbose=verbose)
    if 'c_z' not in data:
        raise RuntimeError('Precomputed c_z/const_z required')

    num_vars = int(data['num_vars'])
    # initial u_guess: choose a feasible mask with sum == unknown_number when possible
    if unknown_number is not None and int(unknown_number) > 0:
        k = int(unknown_number)
        u_fixed = [0] * num_vars
        fixed_zero_set = set(fixed_zero or [])
        cnt = 0
        for j in range(num_vars):
            if j in fixed_zero_set:
                continue
            u_fixed[j] = 1
            cnt += 1
            if cnt >= k:
                break
        if cnt < k and verbose:
            print(f'Warning: could only initialize {cnt}/{k} unknowns (fixed_zero too restrictive)')
    else:
        u_fixed = [0] * num_vars

    best_val = -1e99
    best_x = None
    best_u = None

    start_time = time.time()

    for it in range(max_iters):
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break

        # 1) Solve x given u_fixed (maximize)
        m_x, xvars, uvars, z_out = base.build_joint_mip(data,
                                   start_index=start_index,
                                   goal_idx=goal_idx,
                                   sonar_number=sonar_number,
                                   unknown_number=unknown_number,
                                   fixed_zero=fixed_zero,
                                   forbidden_radius=forbidden_radius,
                                   rows=rows, columns=columns,
                                   prune_eps=prune_eps,
                                   time_limit=per_solve_time_limit,
                                   verbose=verbose)
        # fix u variables to u_fixed if present
        if uvars is not None:
            for j, val in enumerate(u_fixed):
                try:
                    m_x.addConstr(uvars[j] == int(val))
                except Exception:
                    pass

        # attach incumbent callback when verbose
        if verbose:
            try:
                cb_x = base.make_incumbent_callback(data, xvars, uvars, z_out,
                                                    start_index=start_index,
                                                    goal_idx=goal_idx,
                                                    sonar_number=sonar_number,
                                                    unknown_number=unknown_number,
                                                    obstacle_count=len(list(fixed_zero or [])),
                                                    checkpoint_path=checkpoint,
                                                    rows=rows,
                                                    columns=columns,
                                                    obstacle_indices=list(fixed_zero or []),
                                                    verbose=verbose)
                m_x.optimize(cb_x)
            except Exception:
                m_x.optimize()
        else:
            m_x.optimize()
        if m_x.Status not in (2, 9):
            # 2 OPTIMAL, 9 TIME_LIMIT
            if verbose:
                print('x-solve failed or no solution')
            break

        # extract x solution
        x_sol = [j for j in range(num_vars) if getattr(xvars[j], 'X', 0) and xvars[j].X > 0.5]

        # 2) Solve u given x_sol (minimize)
        m_u, xvars2, uvars2, z_out2 = base.build_joint_mip(data,
                                  start_index=start_index,
                                  goal_idx=goal_idx,
                                  sonar_number=sonar_number,
                                  unknown_number=unknown_number,
                                  fixed_zero=fixed_zero,
                                  forbidden_radius=forbidden_radius,
                                  rows=rows, columns=columns,
                                  prune_eps=prune_eps,
                                  time_limit=per_solve_time_limit,
                                  verbose=verbose)
        # fix x to x_sol
        for j in range(num_vars):
            val = 1 if j in x_sol else 0
            try:
                m_u.addConstr(xvars2[j] == int(val))
            except Exception:
                pass

        # minimize worst-case
        try:
            import gurobipy as gp
            from gurobipy import GRB
            m_u.setObjective(z_out2, GRB.MINIMIZE)
        except Exception:
            # fallback: keep default
            pass

        # attach incumbent callback when verbose
        if verbose:
            try:
                cb_u = base.make_incumbent_callback(data, xvars2, uvars2, z_out2,
                                                    start_index=start_index,
                                                    goal_idx=goal_idx,
                                                    sonar_number=sonar_number,
                                                    unknown_number=unknown_number,
                                                    obstacle_count=len(list(fixed_zero or [])),
                                                    checkpoint_path=checkpoint,
                                                    rows=rows,
                                                    columns=columns,
                                                    obstacle_indices=list(fixed_zero or []),
                                                    verbose=verbose)
                m_u.optimize(cb_u)
            except Exception:
                m_u.optimize()
        else:
            m_u.optimize()
        if m_u.Status not in (2, 9):
            if verbose:
                print('u-solve failed or no solution')
            break

        # extract u solution and min value
        u_sol = []
        for j in range(num_vars):
            try:
                if getattr(uvars2[j], 'X', None) is not None and uvars2[j].X > 0.5:
                    u_sol.append(j)
            except Exception:
                pass

        try:
            val = float(m_u.objVal)
        except Exception:
            val = None

        if val is not None and val > best_val + 1e-9:
            best_val = val
            best_x = list(x_sol)
            best_u = list(u_sol)

        # convergence check
        if u_fixed == [1 if j in u_sol else 0 for j in range(num_vars)]:
            if verbose:
                print('u fixed repeated, converged')
            break

        # update u_fixed for next iteration
        u_fixed = [1 if j in u_sol else 0 for j in range(num_vars)]

    return dict(placements=best_x, unknowns=best_u, robust_objective=best_val)

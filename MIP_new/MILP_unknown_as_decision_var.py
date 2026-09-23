"""Build and solve MILP with `u` as a decision variable (joint formulation).

This module exposes a simple `solve` function that builds the MILP using
`base_MILP.build_joint_mip` and runs Gurobi to obtain placements and unknowns.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

from . import load_model_cz_bounds as loader
from . import base_MILP as base


def solve(checkpoint: str | None, precompute_dir: str | None, local_bounds: str | None,
          start_index: int, goal_idx: int, sonar_number: int, unknown_number: int,
          fixed_zero=None, forbidden_radius: int = 6, rows: int = 20, columns: int = 20,
          prune_eps: float = 1e-8, time_limit: float | None = None,
        mip_gap: float | None = None,
        use_known_unknown_distinctions: bool = True,
        verbose: bool = False):
    data = loader.load(checkpoint,
                 precompute_dir=precompute_dir,
                 local_bounds_npz=local_bounds,
                 include_mask_features=use_known_unknown_distinctions,
                 verbose=verbose)

    if 'c_z' not in data or 'const_z' not in data:
        raise RuntimeError('Precomputed c_z/const_z required (precompute_dir)')

    m, xvars, uvars, z_out = base.build_joint_mip(data,
                                                 start_index=start_index,
                                                 goal_idx=goal_idx,
                                                 sonar_number=sonar_number,
                                                 unknown_number=unknown_number,
                                                 fixed_zero=fixed_zero,
                                                 forbidden_radius=forbidden_radius,
                                                 rows=rows, columns=columns,
                                                 prune_eps=prune_eps,
                                                 time_limit=time_limit,
                                                 use_known_unknown_distinctions=use_known_unknown_distinctions,
                                                 mip_gap=mip_gap,
                                                 verbose=verbose)

    # optimize (attach incumbent callback when verbose)
    if verbose:
        try:
            cb = base.make_incumbent_callback(data, xvars, uvars, z_out, start_index=start_index,
                                                 goal_idx=goal_idx, sonar_number=sonar_number,
                                                 unknown_number=unknown_number,
                                                 obstacle_count=len(list(fixed_zero or [])),
                                                 use_known_unknown_distinctions=use_known_unknown_distinctions,
                                                 checkpoint_path=checkpoint,
                                                 rows=rows,
                                                 columns=columns,
                                                 obstacle_indices=list(fixed_zero or []),
                                                 verbose=verbose)
            m.optimize(cb)
        except Exception:
            m.optimize()
    else:
        m.optimize()
    status = int(getattr(m, 'Status', 0))
    try:
        obj = float(m.objVal) if status != 0 else None
    except Exception:
        obj = None

    sol_x = []
    sol_u = []
    # extract x
    for j in range(int(data['num_vars'])):
        try:
            xv = xvars[j]
            if getattr(xv, 'X', None) is not None and xv.X > 0.5:
                sol_x.append(int(j))
        except Exception:
            pass
    if uvars is not None:
        for j in range(int(data['num_vars'])):
            try:
                uv = uvars[j]
                if getattr(uv, 'X', None) is not None and uv.X > 0.5:
                    sol_u.append(int(j))
            except Exception:
                pass

    return dict(placements=sol_x, unknowns=sol_u, objective=obj, model=m)

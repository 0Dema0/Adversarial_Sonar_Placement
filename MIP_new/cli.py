"""CLI to run the new MIP solvers in `MIP_new`.

Usage examples:
    python -m MIP_new.cli --mode joint --checkpoint model.pth --bounds-dir tight_bounds/model_x
    python -m MIP_new.cli --mode robust --checkpoint model.pth --bounds-dir tight_bounds/model_x
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from . import MILP_unknown_as_decision_var as joint
from . import MILP_robust as robust


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['joint', 'robust'], default='joint')
    p.add_argument('--checkpoint', required=True, help='PyTorch model checkpoint')
    p.add_argument('--bounds-dir', required=True, help='Directory containing c_z.npy, const_z.npy, local_bounds.npz, and optional global_bounds.npz')
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--goal', type=int, default=399)
    p.add_argument('--sonar-number', type=int, default=8, help='Number of sonar placements to use')
    p.add_argument('--unknown-number', type=int, default=0)
    p.add_argument('--forbidden-radius', type=int, default=6, help='Hex radius around start/goal where placements are forbidden')
    p.add_argument('--rows', type=int, default=20, help='Grid rows')
    p.add_argument('--columns', type=int, default=20, help='Grid columns')
    p.add_argument("--obstacles", nargs="+", type=int, default=[
            43, 44, 24, 183, 184, 203, 204, 205,
            224, 225, 244, 245, 262, 263, 264, 265,
            285, 286, 287, 267, 132, 133, 152, 190,
            191, 211, 212, 213, 355, 376, 113, 112,
            94, 92, 91, 63, 64, 19, 18, 17, 16, 39,
            38, 37, 59, 231, 232, 233, 234, 251, 252,
            253, 271, 290, 291, 292, 309, 310, 329, 351
        ], help='List of obstacle cell indices (fixed zeroes)')
    p.add_argument('--time-limit', type=float, default=None)
    p.add_argument('--mip-gap', type=float, default=None)
    p.add_argument('--output', default=None, help='Optional NPZ file to save placements')
    p.add_argument('--verbose', action='store_true')

    args = p.parse_args(argv)
    bounds_dir = Path(args.bounds_dir)
    local_bounds = bounds_dir / 'local_bounds.npz'
    if not local_bounds.exists():
        raise FileNotFoundError(f'local bounds not found: {local_bounds}')

    if args.mode == 'joint':
        res = joint.solve(args.checkpoint, str(bounds_dir), str(local_bounds),
                          start_index=int(args.start), goal_idx=int(args.goal),
                          sonar_number=int(args.sonar_number), unknown_number=int(args.unknown_number),
                          fixed_zero=args.obstacles,
                          forbidden_radius=int(args.forbidden_radius), rows=int(args.rows), columns=int(args.columns),
                          time_limit=args.time_limit, mip_gap=args.mip_gap, verbose=args.verbose)
    else:
        res = robust.robust_solve(args.checkpoint, str(bounds_dir), str(local_bounds),
                                  start_index=int(args.start), goal_idx=int(args.goal),
                                  sonar_number=int(args.sonar_number), unknown_number=int(args.unknown_number),
                                  fixed_zero=args.obstacles,
                                  forbidden_radius=int(args.forbidden_radius), rows=int(args.rows), columns=int(args.columns),
                                  time_limit=args.time_limit, verbose=args.verbose)

    print('Result:', {k: v for k, v in res.items() if k != 'model'})
    if args.output and 'placements' in res:
        np.savez(args.output, placements=np.array(res.get('placements', []), dtype=int), unknowns=np.array(res.get('unknowns', []), dtype=int))
        print('Saved solution to', args.output)


if __name__ == '__main__':
    main()

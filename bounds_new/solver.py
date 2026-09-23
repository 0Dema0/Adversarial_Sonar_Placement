"""Gurobi-based MILP helpers used by bounds_new.

Provides a solve_min_max routine that supports adversarial unknowns (u_j).
"""
from __future__ import annotations

from typing import Iterable
import numpy as np


def _ensure_numpy(v):
    return np.asarray(v, dtype=float)


def solve_min_max(
    coeff_x: list[float],
    coeff_u: list[float] | None,
    const: float,
    vars_idx: list[int],
    num_vars: int,
    sonar_number: int | None = None,
    fixed_zero: Iterable[int] | None = None,
    unknown_count: int | None = None,
    time_limit: float | None = None,
    threads: int | None = None,
    sign_stop_at_zero: bool = False,
    verbose: bool = False,
) -> tuple[float, float]:
    """Solve min and max of `const + coeff_x @ x + coeff_u @ u` with:

    - x_j binary for j in `vars_idx` (mapped to original indices), sum(x)=sonar_number
    - optionally u_j binary with u_j <= x_j and sum(u)=unknown_count

    `coeff_u` may be None (no unknown variables).
    Returns (min, max).
    """
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception as e:
        raise RuntimeError("Gurobi (gurobipy) is required for exact tightening") from e

    c_x = _ensure_numpy(coeff_x).reshape(-1)
    if coeff_u is None:
        c_u = None
    else:
        c_u = _ensure_numpy(coeff_u).reshape(-1)

    if c_x.size != len(vars_idx):
        raise ValueError("coeff_x length must match vars_idx length")
    if c_u is not None and c_u.size != len(vars_idx):
        raise ValueError("coeff_u length must match vars_idx length")

    fixed_zero = set(fixed_zero or [])

    def _solve(maximize: bool) -> float:
        m = gp.Model("bound")
        m.setParam("OutputFlag", 1 if verbose else 0)
        if time_limit is not None:
            m.setParam("TimeLimit", float(time_limit))
        if threads is not None and int(threads) > 0:
            m.setParam("Threads", int(threads))

        xs = [m.addVar(vtype=GRB.BINARY, name=f"x_{j}") for j in range(len(vars_idx))]

        # fix obstacle vars to zero when included
        for ridx, orig_j in enumerate(vars_idx):
            if orig_j in fixed_zero:
                m.addConstr(xs[ridx] == 0)

        # cardinality constraint (assume vars_idx covers enough vars)
        if sonar_number is not None:
            m.addConstr(gp.quicksum(xs) == int(sonar_number))

        us = None
        if c_u is not None and unknown_count is not None and int(unknown_count) > 0:
            us = [m.addVar(vtype=GRB.BINARY, name=f"u_{j}") for j in range(len(vars_idx))]
            # u_j <= x_j
            for ridx in range(len(vars_idx)):
                m.addConstr(us[ridx] <= xs[ridx])
            # sum u == unknown_count
            m.addConstr(gp.quicksum(us) == int(unknown_count))

        # objective
        expr = gp.quicksum(float(c_x[r]) * xs[r] for r in range(len(vars_idx)))
        if c_u is not None and us is not None:
            expr = expr + gp.quicksum(float(c_u[r]) * us[r] for r in range(len(vars_idx)))
        expr = expr + float(const)

        if maximize:
            m.setObjective(expr, GRB.MAXIMIZE)
        else:
            m.setObjective(expr, GRB.MINIMIZE)

        sign_proved = {"done": False}

        def _sign_callback(model, where):
            if not sign_stop_at_zero or where != GRB.Callback.MIP:
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

        if sign_stop_at_zero:
            m.optimize(_sign_callback)
        else:
            m.optimize()
        if m.status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED):
            raise RuntimeError(f"Gurobi failed to solve MILP (status {m.status})")
        if sign_proved["done"]:
            return float(m.ObjBound)
        if m.status == GRB.OPTIMAL:
            return float(m.objVal)
        return float(m.ObjBound)

    max_v = _solve(True)
    min_v = _solve(False)
    return min_v, max_v

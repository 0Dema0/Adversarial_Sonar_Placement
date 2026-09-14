"""
Pathfinding functions.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import heapq

def a_star(
    cost_field: npt.NDArray[np.float64],
    neighbor_map: list[list[int]],
    positions: npt.NDArray[np.float64],
    obstacles: set[int],
    goal_index: int,
    start_index: int,
    return_path: bool = False,
) -> tuple[list[int] | None, npt.NDArray[np.float64] | float, int]:

    total_cells = cost_field.shape[0]

    if goal_index in obstacles:
        raise ValueError("Goal cell is an obstacle")
    if start_index in obstacles:
        raise ValueError("Start cell is an obstacle")

    def h(u: int) -> float:
        return np.linalg.norm(positions[u] - positions[goal_index])

    g_score = np.full(total_cells, np.inf, dtype=np.float64)
    f_score = np.full(total_cells, np.inf, dtype=np.float64)

    g_score[start_index] = 0.0
    f_score[start_index] = h(start_index)

    heap: list[tuple[float, int]] = [(f_score[start_index], start_index)]
    came_from: dict[int, int | None] = {start_index: None}
    visited: set[int] = set()
    nodes_visited = 0

    while heap:
        current_f, u = heapq.heappop(heap)

        if current_f > f_score[u]:
            continue
        if u in visited:
            continue

        visited.add(u)
        nodes_visited += 1

        if return_path and u == goal_index:
            path: list[int] = []
            cur: int | None = goal_index
            while cur is not None:
                path.append(cur)
                cur = came_from[cur]
            return path[::-1], float(g_score[goal_index]), nodes_visited

        for v in neighbor_map[u]:
            if v in obstacles:
                continue

            tentative_g = g_score[u] + cost_field[v]

            if tentative_g < g_score[v]:
                g_score[v] = tentative_g
                came_from[v] = u
                f_score[v] = tentative_g + h(v)
                heapq.heappush(heap, (f_score[v], v))

    if return_path:
        return [], float(np.inf), nodes_visited

    return None, g_score, nodes_visited


def dijkstra(
    cost_field: npt.NDArray[np.float64],
    neighbor_map: list[list[int]],
    positions: npt.NDArray[np.float64],
    obstacles: set[int],
    goal_index: int,
    start_index: int,
    return_path: bool = False,
) -> tuple[list[int] | None, npt.NDArray[np.float64] | float, int]:
    """
    Pure Dijkstra's algorithm (no heuristic). API mirrors `a_star` so it
    can be used interchangeably by the planner.
    """
    total_cells = cost_field.shape[0]

    if goal_index in obstacles:
        raise ValueError("Goal cell is an obstacle")
    if start_index in obstacles:
        raise ValueError("Start cell is an obstacle")

    g_score = np.full(total_cells, np.inf, dtype=np.float64)
    g_score[start_index] = 0.0

    heap: list[tuple[float, int]] = [(0.0, start_index)]
    came_from: dict[int, int | None] = {start_index: None}
    visited: set[int] = set()
    nodes_visited = 0

    while heap:
        current_g, u = heapq.heappop(heap)

        if current_g > g_score[u]:
            continue
        if u in visited:
            continue

        visited.add(u)
        nodes_visited += 1

        if return_path and u == goal_index:
            path: list[int] = []
            cur: int | None = goal_index
            while cur is not None:
                path.append(cur)
                cur = came_from[cur]
            return path[::-1], float(g_score[goal_index]), nodes_visited

        for v in neighbor_map[u]:
            if v in obstacles:
                continue

            tentative_g = g_score[u] + cost_field[v]

            if tentative_g < g_score[v]:
                g_score[v] = tentative_g
                came_from[v] = u
                heapq.heappush(heap, (tentative_g, v))

    if return_path:
        return [], float(np.inf), nodes_visited

    return None, g_score, nodes_visited
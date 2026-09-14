import numpy as np
import numpy.typing as npt
from itertools import combinations

from environment.cost import CostMap
from environment.map import RawMap

class SonarKnowledge:
    def __init__(self, start_as_known: npt.NDArray[np.int64], start_as_unknown: npt.NDArray[np.int64]):
        """
        Initialize the SonarKnowledge object with known and unknown sonar indices.
        They are NumPy arrays of dimension (N,) where N is the number of known or unknown sonars, respectively.
        """
        self.start_as_known = start_as_known
        self.start_as_unknown = start_as_unknown
        self.known = start_as_known
        self.unknown = start_as_unknown

def random_sonar_split(sonar_number: int, start_as_unknown_number: int) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    all_ids = np.arange(sonar_number, dtype=np.int64)
    unknown = np.random.choice(all_ids, size=start_as_unknown_number, replace=False)
    known = np.setdiff1d(all_ids, unknown)
    return known, unknown

def best_sonar_split(sonar_number: int, start_as_unknown_number: int, cost_map: CostMap, start_index: int, goal_index: int) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    from environment.planner import IncrementalPlanner
    from environment.state import MapState
    
    planner = IncrementalPlanner(cost_map)
    
    all_ids = np.arange(sonar_number)

    best_unknown = None
    best_score = np.inf

    for unknown in combinations(all_ids, start_as_unknown_number):
        unknown = np.array(unknown, dtype=np.int64)
        known = np.setdiff1d(all_ids, unknown)

        sonar_knowledge = SonarKnowledge(known, unknown)
        state = MapState(
            obstacle_map=cost_map.obstacle_map,
            sonar_knowledge=sonar_knowledge,
            agent_index=start_index,
            goal_index=goal_index
        )

        score = 0.0

        while state.agent_index != state.goal_index:
            update_sonar_visibility(
                raw_map=cost_map.obstacle_map.raw_map,
                agent_index=state.agent_index,
                sonar_knowledge=state.sonar,
                discover_distance=3,
                sonar_indices=cost_map.obstacle_map.raw_map.index_coordinates[cost_map.sonar_indices],
            )

            path, _, _ = planner.plan(state)

            if not path:
                score = np.inf
                break

            if len(path) < 2:
                break

            nxt = path[1]
            score += cost_map.static_cost_map[nxt]
            if score >= best_score:
                break

            state.agent_index = nxt

        if score < best_score:
            best_score = score
            best_unknown = unknown
    
    return np.setdiff1d(all_ids, best_unknown), best_unknown, best_score

def update_sonar_visibility(
    raw_map: RawMap,
    agent_index: int,
    sonar_knowledge: SonarKnowledge,
    discover_distance: int,
    sonar_indices: npt.NDArray[np.int64],
):

    unknown = sonar_knowledge.unknown
    if len(unknown) == 0:
        return np.array([], dtype=np.int64)

    distances = raw_map.hex_distance[
            agent_index,
            sonar_indices[unknown]
        ]

    newly_discovered = unknown[distances <= discover_distance]

    if newly_discovered.size > 0:
        sonar_knowledge.known = np.union1d(sonar_knowledge.known, newly_discovered)
        sonar_knowledge.unknown = np.setdiff1d(sonar_knowledge.unknown, newly_discovered)

    return newly_discovered
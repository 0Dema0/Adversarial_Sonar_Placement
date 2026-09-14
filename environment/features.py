import numpy as np
import numpy.typing as npt

from environment.cost import CostMap
from environment.perception import SonarKnowledge, best_sonar_split, random_sonar_split, update_sonar_visibility
from environment.planner import IncrementalPlanner
from environment.state import MapState
from environment.utils import detection_probability

class StaticFeatures:
    def __init__(
        self,
        raw_map,
        obstacle_map,
        max_radius: int = 3,
        detection_probability=detection_probability,
        visibility_map: npt.NDArray[np.bool_] | None = None,
        base_contributions: npt.NDArray[np.float32] | None = None,
    ):
        self.raw_map = raw_map
        self.obstacle_map = obstacle_map
        self.obstacles = np.asarray(obstacle_map.obstacle_indices, dtype=np.int32)
        self.obstacle_set = set(self.obstacles)

        self.goal_distance = raw_map.hex_distance
        self.goal_onehot_cache = None

        self.neighbor_map = raw_map.neighbor_map
        self.rings = raw_map.rings

        self._build_neighbor_features()
        self._build_ring_features(max_radius)

        # allow passing precomputed visibility/base arrays to avoid recomputation
        if visibility_map is not None:
            self.visibility_map = visibility_map
        else:
            self.visibility_map = self.obstacle_map.compute_visibility_map()

        self.detection_probability = detection_probability

        if base_contributions is not None:
            self.base_contributions = base_contributions
        else:
            self.base_contributions: npt.NDArray[np.float32] | None = None
            self._compute_base_contributions()

    def _build_neighbor_features(self):
        N = self.raw_map.total_cells
        self.neighbor_mask = np.zeros((N, 6), dtype=bool)
        self.neighbor_obstacle = np.zeros((N, 6), dtype=bool)

        for i, neigh in enumerate(self.neighbor_map):
            self.neighbor_mask[i, :len(neigh)] = True
            self.neighbor_obstacle[i, :len(neigh)] = [n in self.obstacle_set for n in neigh]

    def _build_ring_features(self, max_radius):
        N = self.raw_map.total_cells

        self.ring_obstacle_count = np.zeros((N, max_radius), dtype=np.int32)
        self.ring_obstacle_density = np.zeros((N, max_radius), dtype=np.float32)

        for i in range(N):
            for r in range(1, max_radius + 1):
                ring = self.rings[i][r]

                if len(ring) == 0:
                    continue

                obs_count = sum(n in self.obstacle_set for n in ring)

                self.ring_obstacle_count[i, r - 1] = obs_count
                self.ring_obstacle_density[i, r - 1] = obs_count / len(ring)

    def _compute_base_contributions(self):
        prob = self.detection_probability(self.raw_map.euclidean_distance)
        eps = 1e-12
        log_probability = -np.log1p(-np.clip(prob, 0.0, 1.0 - eps))
        self.base_contributions = log_probability * self.visibility_map

class Features:
    def __init__(
        self,
        static: StaticFeatures,
        cost_map: CostMap,
        goal_index: int,
        start_as_unknown_number: int,
        single_value: bool = False,
        start_index: int | None = None,
        best_sonar_split: bool = False,
    ):
        if single_value is False and best_sonar_split is True:
            raise ValueError("best_sonar_split can only be used with single_value=True")

        self.static = static
        self.cost_map = cost_map
        self.raw_map = static.raw_map
        self.goal_index = goal_index
        self.goal_onehot = np.zeros(self.raw_map.total_cells, dtype=np.bool_)
        self.goal_onehot[goal_index] = True
        self.single_value = single_value
        self.start_index = start_index
        self.best_sonar_split = best_sonar_split

        self.cost = cost_map.static_cost_map.astype(np.float32)
        self.obstacle_number = len(static.obstacles)
        self.obstacle_placement: npt.NDArray[np.bool_] = np.zeros(self.raw_map.total_cells, dtype=np.bool_)
        self.obstacle_placement[static.obstacles] = True

        self.sonar_number = cost_map.sonar_number
        self.sonar_placement: npt.NDArray[np.bool_] = np.zeros(self.raw_map.total_cells, dtype=np.bool_)
        self.sonar_placement[cost_map.sonar_indices] = True

        self.start_as_unknown_number = start_as_unknown_number
        self.start_as_known_number = self.sonar_number - self.start_as_unknown_number

        self.known_sonar_mask: npt.NDArray[np.bool_] = np.zeros(self.raw_map.total_cells, dtype=np.bool_)
        self.unknown_sonar_mask: npt.NDArray[np.bool_] = np.zeros(self.raw_map.total_cells, dtype=np.bool_)
        self.outputs = self._compute_outputs_and_known_unknown_sonar_masks()
        self.reachable_mask = np.isfinite(self.outputs)

        self.neighbor_cost, self.neighbor_sonar = self._compute_neighbor()
        self.neighbor_mask = static.neighbor_mask
        self.neighbor_obstacle = static.neighbor_obstacle

        self.ring_cost, self.ring_sonar_count, self.ring_sonar_density = self._compute_ring()
        self.ring_obstacle_count = static.ring_obstacle_count
        self.ring_obstacle_density = static.ring_obstacle_density

    def _compute_outputs_and_known_unknown_sonar_masks(self):
        planner = IncrementalPlanner(self.cost_map)

        obstacles = self.static.obstacles

        if self.best_sonar_split:
            start_as_known, start_as_unknown, out = best_sonar_split(
                sonar_number=self.sonar_number,
                start_as_unknown_number=self.start_as_unknown_number,
                cost_map=self.cost_map,
                start_index=self.start_index,
                goal_index=self.goal_index,
            )

            self.known_sonar_mask[np.asarray(self.cost_map.sonar_indices, dtype=np.int32)[start_as_known]] = True
            self.unknown_sonar_mask[np.asarray(self.cost_map.sonar_indices, dtype=np.int32)[start_as_unknown]] = True

            outputs = np.zeros(self.raw_map.total_cells, dtype=np.float32)
            outputs[self.start_index] = out
            return outputs
        
        else:
            start_as_known, start_as_unknown = random_sonar_split(
                sonar_number=self.sonar_number,
                start_as_unknown_number=self.start_as_unknown_number,
            )

            self.known_sonar_mask[np.asarray(self.cost_map.sonar_indices, dtype=np.int32)[start_as_known]] = True
            self.unknown_sonar_mask[np.asarray(self.cost_map.sonar_indices, dtype=np.int32)[start_as_unknown]] = True

            valid_nodes = (
                [self.start_index]
                if self.single_value
                else [i for i in self.raw_map.index_coordinates if i not in obstacles]
            )

            outputs = np.zeros(self.raw_map.total_cells, dtype=np.float32)

            for start in valid_nodes:
                sonar_knowledge = SonarKnowledge(start_as_known, start_as_unknown)

                state = MapState(
                    obstacle_map=self.static.obstacle_map,
                    sonar_knowledge=sonar_knowledge,
                    agent_index=start,
                    goal_index=self.goal_index,
                )

                while state.agent_index != state.goal_index:
                    update_sonar_visibility(
                        raw_map=self.raw_map,
                        agent_index=state.agent_index,
                        sonar_knowledge=state.sonar,
                        discover_distance=3,
                        sonar_indices=self.raw_map.index_coordinates[self.cost_map.sonar_indices],
                    )

                    path, _, _ = planner.plan(state)

                    if not path:
                        outputs[start] = np.inf
                        break

                    if len(path) < 2:
                        break

                    nxt = path[1]
                    outputs[start] += self.cost[nxt]
                    state.agent_index = nxt

            return outputs

    def _compute_neighbor(self):
        N = self.raw_map.total_cells

        cost = np.zeros((N, 6), dtype=np.float32)
        sonar = np.zeros((N, 6), dtype=np.bool_)
        mask = np.zeros((N, 6), dtype=np.bool_)

        sonar_set = set(self.cost_map.sonar_indices)

        for i, neigh in enumerate(self.raw_map.neighbor_map):
            cost[i, :len(neigh)] = self.cost[neigh]
            sonar[i, :len(neigh)] = [n in sonar_set for n in neigh]

        return cost, sonar

    def _compute_ring(self, max_radius=3):
        N = self.raw_map.total_cells

        ring_cost = np.zeros((N, max_radius))
        ring_sonar_count = np.zeros((N, max_radius), dtype=np.int32)
        ring_sonar_density = np.zeros((N, max_radius), dtype=np.float32)

        sonar_set = set(self.cost_map.sonar_indices)

        for i in range(N):
            for r in range(1, max_radius + 1):
                ring = self.raw_map.rings[i][r]
                if len(ring) == 0:
                    continue

                arr = np.asarray(ring, dtype=np.int32)

                # ring cost: mean cost of non-obstacle cells in the ring (ignore obstacles)
                try:
                    non_obs_mask = ~np.isin(arr, list(self.static.obstacles))
                except Exception:
                    non_obs_mask = ~np.isin(arr, list(self.static.obstacles))

                if non_obs_mask.any():
                    vals = self.cost[arr[non_obs_mask]]
                    # ignore non-finite values
                    finite_mask = np.isfinite(vals)
                    if finite_mask.any():
                        ring_cost[i, r - 1] = float(vals[finite_mask].mean())
                    else:
                        ring_cost[i, r - 1] = 0.0
                else:
                    ring_cost[i, r - 1] = 0.0

                ring_sonar_count[i, r - 1] = int(np.isin(arr, list(sonar_set)).sum())
                ring_sonar_density[i, r - 1] = ring_sonar_count[i, r - 1] / len(ring)

        return ring_cost, ring_sonar_count, ring_sonar_density
    
    def to_dict(self):
        return {
            "goal_index": self.goal_index,
            "obstacle_number": self.obstacle_number,
            "sonar_number": self.sonar_number,
            "start_as_unknown_number": self.start_as_unknown_number,
            "start_as_known_number": self.start_as_known_number,
            "best_sonar_split": self.best_sonar_split,

            "cost": self.cost,
            "outputs": self.outputs,
            "reachable_mask": self.reachable_mask,
            "goal_onehot": self.goal_onehot,
            "obstacle_placement": self.obstacle_placement,
            "sonar_placement": self.sonar_placement,
            "axial_coordinates": self.raw_map.axial_coordinates,
            "cartesian_coordinates": self.raw_map.cartesian_coordinates,
            "distance_to_goal": self.static.goal_distance[:, self.goal_index],
            "known_sonar_mask": self.known_sonar_mask,
            "unknown_sonar_mask": self.unknown_sonar_mask,

            "neighbor_cost": self.neighbor_cost,
            "neighbor_obstacle": self.neighbor_obstacle,
            "neighbor_sonar": self.neighbor_sonar,
            "neighbor_mask": self.neighbor_mask,

            "ring_cost": self.ring_cost,
            "ring_obstacle_count": self.ring_obstacle_count,
            "ring_obstacle_density": self.ring_obstacle_density,
            "ring_sonar_count": self.ring_sonar_count,
            "ring_sonar_density": self.ring_sonar_density,
        }
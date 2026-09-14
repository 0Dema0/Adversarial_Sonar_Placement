import numpy as np

from environment.cost import CostMap
from environment.features import StaticFeatures, Features
from environment.map import RawMap, ObstacleMap
from environment.utils import detection_probability

def demo_features(
    raw_map: RawMap,
    obstacle_map: ObstacleMap,
    sonar_indices: list[int],
    goal_index: int = 399,
    start_index: int = 0,
    start_as_unknown_number: int = 2,
    single_value: bool = True,
):
    """
    Compute and print features for a specific sonar placement.
    
    Args:
        raw_map: The hexagonal map
        obstacle_map: The obstacle map
        sonar_indices: List of sonar positions
        goal_index: Goal cell index
        start_index: Starting position index
        start_as_unknown_number: Number of sonars starting as unknown
        single_value: Whether to compute single value or all nodes
    """
    # Create cost map with specified sonar placement
    cost_map = CostMap(
        obstacle_map=obstacle_map,
        sonar_indices=sonar_indices,
        detection_probability=detection_probability,
        sonar_number=len(sonar_indices)
    )
    
    # Compute static features
    static_features = StaticFeatures(
        raw_map=raw_map,
        obstacle_map=obstacle_map,
        max_radius=3,
    )
    
    # Compute dynamic features
    features = Features(
        static=static_features,
        cost_map=cost_map,
        goal_index=goal_index,
        start_as_unknown_number=start_as_unknown_number,
        single_value=single_value,
        start_index=start_index if single_value else None,
        best_sonar_split=True,
    )
    
    # Print feature summary
    print(f"\n=== Features Summary ===")
    print(f"Goal index: {features.goal_index}")
    print(f"Start index: {start_index}")
    print(f"Obstacle count: {features.obstacle_number}")
    print(f"Sonar count: {features.sonar_number}")
    print(f"Sonar known: {features.start_as_known_number}")
    print(f"Sonar unknown: {features.start_as_unknown_number}")
    print(f"Reachable cells: {np.sum(features.reachable_mask)}/{raw_map.total_cells}")
    
    if single_value and start_index is not None:
        output_cost = features.outputs[start_index]
        print(f"Output cost from {start_index} to {goal_index}: {output_cost}")
    
    # Print feature arrays info
    print(f"\nFeature arrays shapes:")
    print(f"  cost: {features.cost.shape}")
    print(f"  outputs: {features.outputs.shape}")
    print(f"  neighbor_cost: {features.neighbor_cost.shape}")
    print(f"  ring_cost: {features.ring_cost.shape}")
    print(f"  ring_sonar_count: {features.ring_sonar_count.shape}")
    
    features_dict = features.to_dict()
    print(f"\nFeatures dictionary keys: {list(features_dict.keys())}")
    
    return features, features_dict


def create_raw_map(scale: float = 1.0, width: int = 20, height: int = 20) -> RawMap:
    """
    Create and return a RawMap instance.
    
    Args:
        scale: Scale factor for hexagon size
        width: Width of the hexagonal grid
        height: Height of the hexagonal grid
        
    Returns:
        A RawMap instance
    """
    return RawMap(scale=scale, width=width, height=height)


def create_obstacle_map(
    raw_map: RawMap,
    obstacle_number: int = 10,
    obstacle_indices: list[int] | None = None
) -> ObstacleMap:
    """
    Create and return an ObstacleMap instance.
    
    Args:
        raw_map: The RawMap instance
        obstacle_number: Number of obstacles to place randomly
        obstacle_indices: Specific indices for obstacles (if None, placed randomly)
        
    Returns:
        An ObstacleMap instance
    """
    return ObstacleMap(
        raw_map=raw_map,
        obstacle_number=obstacle_number,
        obstacle_indices=obstacle_indices
    )


if __name__ == "__main__":
    # Create map and obstacle map
    raw_map = create_raw_map(scale=1.0, width=20, height=20)
    obstacle_map = create_obstacle_map(raw_map, obstacle_number=60, obstacle_indices=[43, 44, 24, 183, 184, 203, 204, 205, 224, 225, 244, 245, 262, 263, 264, 265, 285, 286, 287, 267, 132, 133, 152, 190, 191, 211, 212, 213, 355, 376, 113, 112, 94, 92, 91, 63, 64, 19, 18, 17, 16, 39, 38, 37, 59, 231, 232, 233, 234, 251, 252, 253, 271, 290, 291, 292, 309, 310, 329, 351])
    
    # Define sonar positions and goal
    sonar_indices = [235, 236, 238, 259, 314, 349, 373, 391]
    goal_index = 399
    start_index = 0
    
    # Compute and display features
    features, features_dict = demo_features(
        raw_map=raw_map,
        obstacle_map=obstacle_map,
        sonar_indices=sonar_indices,
        goal_index=goal_index,
        start_index=start_index,
        start_as_unknown_number=2,
        single_value=True,
    )

"""
This module defines the RawMap and ObstacleMap classes for representing hexagonal maps and obstacles.
The RawMap class represents a hexagonal map with its geometry, including index, axial, cubic, and Cartesian coordinates, as well as cell polygons and neighbor relationships.
The ObstacleMap class represents a hexagonal map with obstacles, allowing for the placement of obstacles and checking visibility between cells considering the obstacles.
The RawMap class is initialized with a scale, width, and height, and it builds the grid, cell polygons, and neighbor relationships. The ObstacleMap class is initialized with a RawMap, obstacle number, and optional obstacle indices, and it places obstacles and provides a method to check visibility between cells.
"""

import numpy as np
import numpy.typing as npt

from environment.utils import axial_to_cart, neighbors, hex_radius, hex_corners, segment_intersects_polygon

class RawMap:
    """Represents a hexagonal map (geometry only).

    Coordinates are stored as NumPy arrays for efficient numeric ops.
    """

    def __init__(self, scale: float = 1.0, width: int = 20, height: int = 20):
        self.scale = scale  # Default scale for hexagon size
        self.width = width
        self.height = height
        self.total_cells: int = width * height

        self.index_coordinates: npt.NDArray[np.int64] = np.empty(0, dtype=np.int64)
        self.axial_coordinates: npt.NDArray[np.int64] = np.empty((0, 2), dtype=np.int64)
        self.cubic_coordinates: npt.NDArray[np.int64] = np.empty((0, 3), dtype=np.int64)
        self.cartesian_coordinates: npt.NDArray[np.float64] = np.empty((0, 2), dtype=np.float64)
        self.build_grid()

        self.hex_distance: npt.NDArray[np.int64] = np.empty((0, 0), dtype=np.int64)
        self.euclidean_distance: npt.NDArray[np.float64] = np.empty((0, 0), dtype=np.float64)
        self.rings: list[list[npt.NDArray[np.int64]]] = []
        self.build_hex_distance()
        self.build_euclidean_distance()
        self.build_rings()

        self.cell_radius: float = 0.0
        self.cell_polygons: npt.NDArray[np.float64] = np.empty((0, 6, 2), dtype=np.float64)
        self.build_cell_polygons()

        self.neighbor_map: list[list[int]] = []
        self.get_neighbors()

    def build_grid(self) -> None:
        scale = self.scale
        width = self.width
        height = self.height

        r = np.repeat(np.arange(height), width)
        c = np.tile(np.arange(width), height)

        index_coordinates = np.arange(width * height)

        q = c - ((r - (r & 1)) // 2)
        axial_coordinates = np.stack((q, r), axis=1)

        x = q
        z = r
        y = -x - z
        cubic_coordinates = np.stack((x, y, z), axis=1)

        self.index_coordinates = index_coordinates
        self.axial_coordinates = axial_coordinates
        self.cubic_coordinates = cubic_coordinates
        self.cartesian_coordinates = axial_to_cart(axial_coordinates, scale)

    def build_cell_polygons(self) -> None:
        """
        Build the corner coordinates for each hexagonal cell in the grid.
        """
        self.cell_radius = hex_radius(self.cartesian_coordinates)
        self.cell_polygons = np.stack([
            hex_corners(center, self.cell_radius) for center in self.cartesian_coordinates
        ])

    def get_neighbors(self) -> None:
        """
        Get the neighbor adjacency list for the hexagonal grid.
        """
        self.neighbor_map = neighbors(self.cubic_coordinates)

    def build_hex_distance(self) -> None:
        """
        Compute the pairwise cube (hex) distance between every pair of cells.
        """

        diff = (
            self.cubic_coordinates[:, None, :]
            - self.cubic_coordinates[None, :, :]
        )

        self.hex_distance = np.max(np.abs(diff), axis=2).astype(np.int64)

    def build_euclidean_distance(self) -> None:
        """
        Compute pairwise Euclidean distances between all cell centers.
        """

        diff = (
            self.cartesian_coordinates[:, None, :]
            - self.cartesian_coordinates[None, :, :]
        )

        self.euclidean_distance = np.linalg.norm(diff, axis=2)

    def build_rings(self) -> None:
        """
        Precompute the cells belonging to every ring around every cell.
        """

        self.rings = []

        max_radius = self.hex_distance.max()

        for i in range(self.total_cells):

            rings_i = []

            for r in range(max_radius + 1):

                rings_i.append(
                    np.where(self.hex_distance[i] == r)[0]
                )

            self.rings.append(rings_i)

    def get_indices_within_radius(self, index: int, radius: int) -> npt.NDArray[np.int64]:
        """
        Get the indices of cells within a given radius from a specified cell index.

        Parameters:
        - index: The index of the center cell.
        - radius: The radius within which to find neighboring cells.

        Returns:
        - A NumPy array of indices of cells within the specified radius.
        """
        if radius < 0:
            raise ValueError("Radius must be non-negative.")
        if index < 0 or index >= self.total_cells:
            raise ValueError("Index out of bounds.")

        return np.where(self.hex_distance[index] <= radius)[0]

class ObstacleMap:
    """
    Represents a hexagonal map with obstacles.
    """
    def __init__(
            self,
            raw_map: RawMap, obstacle_number: int = 0,
            obstacle_indices: list[int] | None = None
        ):
        self.raw_map = raw_map

        self.obstacle_number = obstacle_number
        self.obstacle_indices = obstacle_indices or []
        self.obstacle_cartesian_coordinates: npt.NDArray[np.float64] = np.empty((0, 2), dtype=np.float64)
        self.place_obstacles()

        self.obstacle_polygons: npt.NDArray[np.float64] = self.raw_map.cell_polygons[self.obstacle_indices]

    def place_obstacles(self) -> None:
        """
        Place obstacles on the hexagonal grid.

        If `obstacle_indices` is provided, use those indices; otherwise, place obstacles randomly
        on the grid, ensuring they do not overlap with existing obstacles and are not placed
        on start or goal position (assumed to be cells 0 and the last cell in the grid).
        """
        if not self.obstacle_indices:
            available_indices = np.setdiff1d(
                self.raw_map.index_coordinates,
                [0, self.raw_map.width * self.raw_map.height - 1]
            )
            self.obstacle_indices = np.random.choice(
                available_indices,
                size=self.obstacle_number,
                replace=False
            ).tolist()
        self.obstacle_cartesian_coordinates = self.raw_map.cartesian_coordinates[self.obstacle_indices]

    def visible(self, source: int, target: int) -> bool:
        """
        Check if the target cell is visible from the source cell, considering obstacles.

        Visibility is determined by checking if the line segment between the centers of the source and target cells
        intersects with any of the obstacle polygons.
        """
        source_center = self.raw_map.cartesian_coordinates[source]
        target_center = self.raw_map.cartesian_coordinates[target]

        for obstacle_polygon in self.obstacle_polygons:
            if segment_intersects_polygon(source_center, target_center, obstacle_polygon):
                return False
        return True

    def compute_visibility_map(self) -> npt.NDArray[np.bool_]:
        """
        Return the precomputed visibility matrix for the obstacle map.
        """
        visibility_map: npt.NDArray[np.bool_] = np.zeros((self.raw_map.total_cells, self.raw_map.total_cells), dtype=bool)
        for source in range(self.raw_map.total_cells):
            # vectorize by calling visible for each target (safe fallback)
            row = np.array([self.visible(int(source), int(target)) for target in range(self.raw_map.total_cells)], dtype=bool)
            visibility_map[source, :] = row
        return visibility_map
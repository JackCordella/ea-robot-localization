from __future__ import annotations

import numpy as np
from typing import List
import math


def create_demo_map(name: str, width: int, height: int) -> np.ndarray:
    grid = np.zeros((height, width), dtype=np.uint8)

    # Outer walls
    grid[0, :] = 1
    grid[-1, :] = 1
    grid[:, 0] = 1
    grid[:, -1] = 1

    if name == "simple_room":
        # Simple room with two asymmetric internal obstacles.
        # Easy map: mostly open, but without a single dominant symmetric wall.
        grid[18:20, 14:36] = 1
        grid[40:42, 46:68] = 1

        # Small vertical feature to break symmetry
        grid[28:40, 58:60] = 1

    elif name == "easy_room":
        # Easy structured room with wide passages.
        # Partial walls are asymmetric and leave clear openings.
        grid[16:18, 15:40] = 1
        grid[36:38, 22:62] = 1
        grid[18:36, 60:62] = 1

        # Extra short obstacle to make the map more distinctive
        grid[36:50, 22:24] = 1

    elif name == "medium_room":
        # Medium-complexity room with several distinguishable areas.
        # Openings are intentionally wider than before.
        grid[12:14, 28:64] = 1
        grid[28:30, 12:24] = 1
        grid[44:46, 36:64] = 1

        grid[14:46, 62:64] = 1
        grid[30:54, 22:24] = 1
        grid[30:40, 62:64] = 0


        # Asymmetric short landmarks
        grid[14:28, 12:14] = 1
        grid[46:60, 62:64] = 1
        grid[46:50, 8:12] = 1

    elif name == "complex_room":
        # Complex room with multiple connected regions and wider doors.
        # The layout avoids very narrow slits and repeated symmetric structures.

        # Horizontal walls
        grid[10:12, 10:36] = 1
        grid[18:20, 38:80] = 1
        grid[36:38, 0:42] = 1
        grid[50:52, 36:80] = 1

        # Vertical walls
        grid[10:28, 36:38] = 1
        grid[28:50, 58:60] = 1
        grid[26:48, 18:20] = 1

        # Extra asymmetric landmarks
        grid[8:12, 56:60] = 1
        grid[44:48, 8:12] = 1

    elif name == "corridor" or name == "corridor_2":
        # Corridor-like map with repeated structure, but not perfectly periodic.
        # Still useful for ambiguity, but less artificially symmetric.

        # Long corridor boundaries
        grid[12:14, 5:75] = 1
        grid[46:48, 5:75] = 1

        # Side alcoves / separators with non-uniform spacing
        upper_xs = [12, 27, 45, 63]
        lower_xs = [18, 36, 55, 69]

        for x in upper_xs:
            grid[14:25, x:x + 2] = 1

        for x in lower_xs:
            grid[35:46, x:x + 2] = 1
 

    else:
        raise ValueError(f"Unknown demo map: {name}")

    return grid


def world_to_grid(x: float, y: float, resolution: float) -> tuple[int, int]:
    col = int(round(x / resolution))
    row = int(round(y / resolution))
    return row, col

def get_world_bounds(occ_grid:np.ndarray, resolution: float) -> List[tuple[float, float], tuple[float, float],tuple[float,float]]:
    height, width = occ_grid.shape
    x_bounds = (0.0, (width - 1) * resolution)
    y_bounds = (0.0, (height - 1) * resolution)
    theta_bound = (-math.pi,math.pi)
    return x_bounds, y_bounds, theta_bound

def is_inside_map(row: int, col: int, occ_grid: np.ndarray) -> bool:
    return 0 <= row < occ_grid.shape[0] and 0 <= col < occ_grid.shape[1]


def is_pose_valid(x: float, y: float, occ_grid: np.ndarray, resolution: float) -> bool:
    row, col = world_to_grid(x, y, resolution)
    if not is_inside_map(row, col, occ_grid):
        return False
    return occ_grid[row, col] == 0

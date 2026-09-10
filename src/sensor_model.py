from __future__ import annotations

import math
import numpy as np

from src.map_utils import world_to_grid, is_inside_map
from src.pose import Pose
from config import SensorConfig


def simulate_scan(
    pose: Pose,
    occ_grid: np.ndarray,
    resolution: float,
    sensor_cfg: SensorConfig,
    rng: np.random.Generator | None = None,
    add_noise: bool = True,
) -> np.ndarray:
    
    angles = np.linspace(
        -math.radians(sensor_cfg.fov_deg) / 2.0,
        math.radians(sensor_cfg.fov_deg) / 2.0,
        sensor_cfg.num_beams,
    )
    ranges = []

    for angle in angles:
        beam_theta = pose.theta + angle
        measured_range = sensor_cfg.max_range_m

        distance = 0.0
        while distance <= sensor_cfg.max_range_m:
            x = pose.x + distance * math.cos(beam_theta)
            y = pose.y + distance * math.sin(beam_theta)
            row, col = world_to_grid(x, y, resolution)
            if not is_inside_map(row, col, occ_grid):
                measured_range = distance
                break
            if occ_grid[row, col] == 1:
                measured_range = distance
                break
            distance += sensor_cfg.step_m

        ranges.append(measured_range)

    scan = np.asarray(ranges, dtype=float)

    if add_noise and sensor_cfg.noise_std_m > 0.0:
        noise = rng.normal(0.0, sensor_cfg.noise_std_m, size=scan.shape)
        scan = np.clip(scan + noise, 0.0, sensor_cfg.max_range_m)
    
    return scan

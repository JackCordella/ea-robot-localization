from __future__ import annotations

import numpy as np

from src.individual import Individual
from src.map_utils import is_pose_valid
from src.perception import (ScanPerception, extract_scan_perception, perception_distance)
from src.pose import Pose
from src.sensor_model import simulate_scan
from config import FitnessBreakdown, SensorConfig




def scan_mse(observed: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean((observed - predicted) ** 2))


def normalized_scan_mse(
    observed: np.ndarray,
    predicted: np.ndarray,
    sensor_max_range: float,
) -> float:
    scale = max(float(sensor_max_range) ** 2, 1e-12)
    return scan_mse(observed, predicted) / scale


def perception_aware_fitness_from_scan(
    pose: Pose,
    observed_scan: np.ndarray,
    observed_perception: ScanPerception,
    occ_grid: np.ndarray,
    resolution: float,
    sensor_cfg: SensorConfig,
    perception_weight: float,
    rng: np.random.Generator | None = None
) -> tuple[float, ScanPerception | None, FitnessBreakdown]:

    if not is_pose_valid(pose.x, pose.y, occ_grid, resolution):
        breakdown = FitnessBreakdown(
            scan_error=float("inf"),
            perception_error=float("inf"),
            fitness=0.0,
        )
        return 0.0, None, breakdown

    predicted_scan = simulate_scan(pose, occ_grid, resolution, sensor_cfg, rng ,add_noise=False)
    predicted_perception = extract_scan_perception(
        predicted_scan,
        sensor_max_range=sensor_cfg.max_range_m,
    )

    scan_error = normalized_scan_mse(
        observed_scan,
        predicted_scan,
        sensor_max_range=sensor_cfg.max_range_m,
    )
    feature_error = perception_distance(
        observed=observed_perception,
        predicted=predicted_perception,
        sensor_max_range=sensor_cfg.max_range_m,
    )

    combined_error = (1-perception_weight)*scan_error + perception_weight * feature_error
    fitness = 1.0 / (1.0 + combined_error)
    breakdown = FitnessBreakdown(
        scan_error=scan_error,
        perception_error=feature_error,
        fitness=fitness,
    )
    return fitness, predicted_perception, breakdown


def evaluate_perception_aware_individual(
    individual: Individual,
    observed_scan: np.ndarray,
    observed_perception: ScanPerception,
    occ_grid: np.ndarray,
    resolution: float,
    sensor_cfg: SensorConfig,
    perception_weight: float,
    rng: np.random.Generator | None = None
) -> FitnessBreakdown:

    fitness, predicted_perception, breakdown = perception_aware_fitness_from_scan(
        pose=individual.pose,
        observed_scan=observed_scan,
        observed_perception=observed_perception,
        occ_grid=occ_grid,
        resolution=resolution,
        sensor_cfg=sensor_cfg,
        perception_weight=perception_weight,
        rng=rng
    )
    individual.fitness = fitness
    individual.perception = predicted_perception
    return breakdown

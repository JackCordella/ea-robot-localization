"""Shared configuration constants and fixed test poses for the experiments."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from config import MapConfig
from src.map_utils import create_demo_map, is_pose_valid
from src.pose import Pose


MAP_NAMES = [
    "simple_room",
    "easy_room",
    "medium_room",
    "complex_room",
    "corridor",
]

METHODS = ["scan_only", "perception_aware"]

# 10 independent stochastic trials per pose/map/configuration.
SEEDS = list(range(10))

DEFAULT_FOV_DEG = 360.0
DEFAULT_NUM_BEAMS = 72
DEFAULT_MAX_RANGE_M = 8.0
DEFAULT_NOISE_STD_M = 0.0
DEFAULT_PERCEPTION_WEIGHT = 0.3

BEAM_VALUES = [18, 36, 72, 108, 144]
BEAM_SENSITIVITY_MAPS = ["simple_room", "complex_room", "corridor"]

# Plotting is intentionally disabled here. Keep plotting utilities in src.visualization
# or in a separate script if needed.
SAVE_RUN_PLOTS = False
SHOW_PROGRESS = True

RESULTS_DIR = Path("results/logs")
PLOTS_DIR = Path("results/plots")

PARAMETER_CONFIGS: list[dict[str, Any]] = [
    {"config_id": "base", "overrides": {}},
    {"config_id": "pop_50", "overrides": {"population_size": 50}},
    {"config_id": "pop_150", "overrides": {"population_size": 150}},
    {"config_id": "gen_80", "overrides": {"generations": 80}},
    {"config_id": "sigma_xy_0_15", "overrides": {"sigma_xy": 0.15}},
    {"config_id": "sigma_xy_0_50", "overrides": {"sigma_xy": 0.50}},
    {"config_id": "tournament_2", "overrides": {"tournament_size": 2}},
    {"config_id": "tournament_5", "overrides": {"tournament_size": 5}},
]

PARAMETER_TUNING_MAPS = ["medium_room", "complex_room", "corridor"]
PARAMETER_TUNING_POSE_IDS = [0, 2, 4]

SYMMETRY_MAPS = ["corridor_2", "complex_room"]
SYMMETRY_POSE_IDS: dict[str, list[int]] = {
    "corridor": [0, 1, 2, 3, 4],
    "complex_room": [0, 1, 2, 3, 4],
}

AMBIGUITY_XY_SAMPLES = 55
AMBIGUITY_THETA_SAMPLES = 24
AMBIGUITY_PERCENTILE = 1.0
AMBIGUITY_MIN_THRESHOLD = 1e-4
AMBIGUITY_MAX_ROWS_PER_POSE = 2500


def get_experiment_poses(map_name: str) -> list[Pose]:
    if map_name == "simple_room":
        return [
            Pose(1.0, 1.0, 0.0),
            Pose(3.0, 2.5, 0.7),
            Pose(4.8, 3.0, 1.5),
            Pose(2.0, 4.5, -1.0),
            Pose(6.5, 4.8, 3.0),
        ]

    if map_name == "easy_room":
        return [
            Pose(1.0, 1.0, 0.2),
            Pose(2.5, 2.2, 1.0),
            Pose(5.0, 3.0, -0.5),
            Pose(6.5, 4.5, 2.2),
            Pose(3.5, 5.5, -2.0),
        ]

    if map_name == "medium_room":
        return [
            Pose(2.0, 2.2, 0.0),
            Pose(1.6, 3.6, 0.8),
            Pose(5.5, 2.0, -0.8),
            Pose(3.4, 5.0, 2.4),
            Pose(7.6, 2.8, -2.6),
        ]

    if map_name == "complex_room":
        return [
            Pose(1.0, 1.6, 0.0),
            Pose(2.8, 3.0, 0.9),
            Pose(4.8, 1.5, -1.2),
            Pose(2.5, 5.0, 2.0),
            Pose(6.8, 3.0, -2.4),
        ]

    if map_name == "corridor":
        return [
            Pose(1.0, 3.0, 0.0),
            Pose(2.4, 3.0, 0.0),
            Pose(3.8, 3.0, 0.0),
            Pose(5.2, 3.0, math.pi),
            Pose(6.6, 3.0, math.pi),
        ]   
    
    if map_name == "corridor_2":
        return [
            Pose(1.7, 1.7, (math.pi)/4),
            Pose(2.6, 4.2, -(math.pi)/4),
            Pose(3.8, 3.0, 0.0),
            Pose(5.2, 4.2, (math.pi)*(3/2)),
            Pose(6.0, 1.8, (math.pi)*(3/4)),
        ]

    raise ValueError(f"Unknown map: {map_name}")


def validate_experiment_poses() -> None:
    map_cfg = MapConfig()

    for map_name in MAP_NAMES:
        occ_grid = create_demo_map(map_name, map_cfg.width, map_cfg.height)
        for pose_id, pose in enumerate(get_experiment_poses(map_name)):
            if not is_pose_valid(pose.x, pose.y, occ_grid, map_cfg.resolution):
                raise ValueError(
                    f"Invalid pose: map={map_name}, pose_id={pose_id}, pose={pose}"
                )

    print("All experiment poses are valid.")

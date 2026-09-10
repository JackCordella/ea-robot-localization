from __future__ import annotations

import math
from typing import Any

from src.metrics import angular_distance, compute_localization_metrics
from src.pose import Pose


def pose_distance_xy(a: Pose, b: Pose) -> float:
    return float(math.hypot(a.x - b.x, a.y - b.y))


def get_top_k_diverse_individuals(
    population,
    k: int = 5,
    min_dist_xy: float = 0.75,
    min_dist_theta: float = 0.50,
):

    sorted_population = sorted(population, key=lambda ind: ind.fitness, reverse=True)
    selected = []

    for candidate in sorted_population:
        is_too_close = False
        for chosen in selected:
            if (
                pose_distance_xy(candidate.pose, chosen.pose) < min_dist_xy
                and angular_distance(candidate.pose.theta, chosen.pose.theta) < min_dist_theta
            ):
                is_too_close = True
                break

        if not is_too_close:
            selected.append(candidate)

        if len(selected) >= k:
            break

    return selected


def top_k_success(top_k, true_pose: Pose, k: int, position_threshold: float = 0.25, theta_threshold: float = 0.25) -> bool:
    for individual in top_k[:k]:
        metrics = compute_localization_metrics(
            estimated_pose=individual.pose,
            true_pose=true_pose,
            position_threshold=position_threshold,
            theta_threshold=theta_threshold,
        )
        if metrics.success:
            return True
    return False

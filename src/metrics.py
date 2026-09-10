import math
from dataclasses import dataclass

from src.pose import Pose
from config import LocalizationMetrics

def angular_distance(a: float, b: float) -> float:
    diff = (a - b + math.pi) % (2.0 * math.pi) - math.pi
    return abs(diff)


def compute_localization_metrics(
    estimated_pose: Pose,
    true_pose: Pose,
    position_threshold: float = 0.25,
    theta_threshold: float = 0.25
) -> LocalizationMetrics:
    
    dx = estimated_pose.x - true_pose.x
    dy = estimated_pose.y - true_pose.y

    position_error = math.sqrt(dx * dx + dy * dy)
    theta_error = angular_distance(estimated_pose.theta, true_pose.theta)

    success = (
        position_error <= position_threshold
        and theta_error <= theta_threshold
    )

    return LocalizationMetrics(
        position_error=position_error,
        theta_error=theta_error,
        success=success,
    )
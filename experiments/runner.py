from __future__ import annotations

import time
from dataclasses import replace
from typing import Any

import numpy as np

from config import EAConfig, MapConfig, SensorConfig
from experiments.configs import (
    DEFAULT_FOV_DEG,
    DEFAULT_MAX_RANGE_M,
    DEFAULT_NOISE_STD_M,
    DEFAULT_NUM_BEAMS,
    DEFAULT_PERCEPTION_WEIGHT,
    METHODS,
    SAVE_RUN_PLOTS,
    SHOW_PROGRESS,
)
from experiments.topk import get_top_k_diverse_individuals, top_k_success
from src.evolutionary_algorithm import EvolutionaryLocalizer
from src.fitness import evaluate_perception_aware_individual
from src.map_utils import create_demo_map, is_pose_valid
from src.metrics import compute_localization_metrics
from src.perception import extract_scan_perception
from src.pose import Pose
from src.sensor_model import simulate_scan


def run_single_localization(
    experiment_name: str,
    map_name: str,
    true_pose: Pose,
    pose_id: int,
    seed: int,
    method: str,
    num_beams: int = DEFAULT_NUM_BEAMS,
    fov_deg: float = DEFAULT_FOV_DEG,
    max_range_m: float = DEFAULT_MAX_RANGE_M,
    noise_std_m: float = DEFAULT_NOISE_STD_M,
    perception_weight: float = DEFAULT_PERCEPTION_WEIGHT,
    ea_cfg_overrides: dict[str, Any] | None = None,
    config_id: str = "default",
    save_plot: bool = SAVE_RUN_PLOTS,
    show_progress: bool = SHOW_PROGRESS,
    compute_top_k: bool = False,
    top_k: int = 5,
    top_k_min_dist_xy: float = 0.75,
    top_k_min_dist_theta: float = 0.50,
) -> dict[str, Any]:
    if method not in set(METHODS):
        raise ValueError(f"Unknown method: {method}")

    rng = np.random.default_rng(seed)
    map_cfg = MapConfig()

    sensor_cfg = SensorConfig(
        num_beams=num_beams,
        fov_deg=fov_deg,
        max_range_m=max_range_m,
        noise_std_m=noise_std_m,
    )

    ea_cfg = EAConfig(seed=seed)
    if ea_cfg_overrides:
        ea_cfg = replace(ea_cfg, **ea_cfg_overrides)
        ea_cfg.seed = seed

    occ_grid = create_demo_map(map_name, map_cfg.width, map_cfg.height)

    if not is_pose_valid(true_pose.x, true_pose.y, occ_grid, map_cfg.resolution):
        raise ValueError(
            f"True pose is not valid: map={map_name}, pose_id={pose_id}, pose={true_pose}"
        )

    observed_scan = simulate_scan(
        pose=true_pose,
        occ_grid=occ_grid,
        resolution=map_cfg.resolution,
        sensor_cfg=sensor_cfg,
        rng=rng,
        add_noise=True,
    )

    observed_perception = extract_scan_perception(
        observed_scan,
        sensor_max_range=sensor_cfg.max_range_m,
    )

    effective_perception_weight = perception_weight if method == "perception_aware" else 0.0

    def evaluator(individual) -> None:
        evaluate_perception_aware_individual(
            individual=individual,
            observed_scan=observed_scan,
            observed_perception=observed_perception,
            occ_grid=occ_grid,
            resolution=map_cfg.resolution,
            sensor_cfg=sensor_cfg,
            perception_weight=effective_perception_weight,
            rng=rng,
        )

    ea = EvolutionaryLocalizer(
        config=ea_cfg,
        occ_grid=occ_grid,
        resolution=map_cfg.resolution,
        rng=rng,
        individual_evaluator=evaluator,
    )

    start_time = time.perf_counter()
    result = ea.run(
        show_progress=show_progress,
        progress_desc=f"{experiment_name} | {map_name} | {method} | p={pose_id} | s={seed}",
    )
    runtime_seconds = time.perf_counter() - start_time

    best = result.best_individual
    estimated_pose = best.pose

    metrics = compute_localization_metrics(estimated_pose=estimated_pose, true_pose=true_pose)

    best_breakdown = evaluate_perception_aware_individual(
        individual=best,
        observed_scan=observed_scan,
        observed_perception=observed_perception,
        occ_grid=occ_grid,
        resolution=map_cfg.resolution,
        sensor_cfg=sensor_cfg,
        perception_weight=effective_perception_weight,
        rng=None,
    )

    scan_error = best_breakdown.scan_error
    feature_error = best_breakdown.perception_error
    combined_error = (1.0 - effective_perception_weight) * scan_error + effective_perception_weight * feature_error
    scan_contribution = (1.0 - effective_perception_weight) * scan_error
    perception_contribution = effective_perception_weight * feature_error


    row: dict[str, Any] = {
        "experiment": experiment_name,
        "config_id": config_id,
        "map_name": map_name,
        "method": method,
        "pose_id": pose_id,
        "seed": seed,
        "true_x": true_pose.x,
        "true_y": true_pose.y,
        "true_theta": true_pose.theta,
        "estimated_x": estimated_pose.x,
        "estimated_y": estimated_pose.y,
        "estimated_theta": estimated_pose.theta,
        "best_fitness": best.fitness,
        "position_error": metrics.position_error,
        "theta_error": metrics.theta_error,
        "success": metrics.success,
        "scan_error": scan_error,
        "feature_error": feature_error,
        "combined_error": combined_error,
        "scan_contribution": scan_contribution,
        "perception_contribution": perception_contribution,
        "runtime_seconds": runtime_seconds,
        "population_size": ea_cfg.population_size,
        "generations": ea_cfg.generations,
        "elite_size": ea_cfg.elite_size,
        "tournament_size": ea_cfg.tournament_size,
        "crossover_rate": ea_cfg.crossover_rate,
        "mutation_rate": ea_cfg.mutation_rate,
        "sigma_xy": ea_cfg.sigma_xy,
        "sigma_theta": ea_cfg.sigma_theta,
        "fov_deg": sensor_cfg.fov_deg,
        "num_beams": sensor_cfg.num_beams,
        "max_range_m": sensor_cfg.max_range_m,
        "noise_std_m": sensor_cfg.noise_std_m,
        "angular_resolution_deg": sensor_cfg.fov_deg / max(sensor_cfg.num_beams - 1, 1),
        "perception_weight": effective_perception_weight,
        "final_best_history": result.best_fitness_history[-1] if result.best_fitness_history else None,
        "final_mean_history": result.mean_fitness_history[-1] if result.mean_fitness_history else None,
    }

    if compute_top_k:
        diverse_top = get_top_k_diverse_individuals(
            result.final_population,
            k=top_k,
            min_dist_xy=top_k_min_dist_xy,
            min_dist_theta=top_k_min_dist_theta,
        )
        row.update(
            {
                "top_k": top_k,
                "top_k_min_dist_xy": top_k_min_dist_xy,
                "top_k_min_dist_theta": top_k_min_dist_theta,
                "num_distinct_modes": len(diverse_top),
                "top1_success": top_k_success(diverse_top, true_pose, k=1),
                "top3_success": top_k_success(diverse_top, true_pose, k=min(3, top_k)),
                "top5_success": top_k_success(diverse_top, true_pose, k=min(5, top_k)),
            }
        )

        for idx in range(top_k):
            if idx < len(diverse_top):
                ind = diverse_top[idx]
                m = compute_localization_metrics(ind.pose, true_pose)
                row[f"top{idx + 1}_x"] = ind.pose.x
                row[f"top{idx + 1}_y"] = ind.pose.y
                row[f"top{idx + 1}_theta"] = ind.pose.theta
                row[f"top{idx + 1}_fitness"] = ind.fitness
                row[f"top{idx + 1}_position_error"] = m.position_error
                row[f"top{idx + 1}_theta_error"] = m.theta_error
            else:
                row[f"top{idx + 1}_x"] = None
                row[f"top{idx + 1}_y"] = None
                row[f"top{idx + 1}_theta"] = None
                row[f"top{idx + 1}_fitness"] = None
                row[f"top{idx + 1}_position_error"] = None
                row[f"top{idx + 1}_theta_error"] = None

    print(
        f"{experiment_name}: map={map_name}, method={method}, pose={pose_id}, seed={seed}, "
        f"success={metrics.success}, pos_err={metrics.position_error:.3f}, "
        f"theta_err={metrics.theta_error:.3f}, fitness={best.fitness:.6f}, "
        f"time={runtime_seconds:.2f}s"
    )

    return row

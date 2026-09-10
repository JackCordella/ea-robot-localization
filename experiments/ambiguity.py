"""Ambiguity and scan-equivalence analysis used by Experiment 5."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from config import MapConfig, SensorConfig
from experiments.configs import (
    AMBIGUITY_MAX_ROWS_PER_POSE,
    AMBIGUITY_MIN_THRESHOLD,
    AMBIGUITY_PERCENTILE,
    AMBIGUITY_THETA_SAMPLES,
    AMBIGUITY_XY_SAMPLES,
    RESULTS_DIR,
)
from experiments.io import append_rows_csv
from src.fitness import normalized_scan_mse
from src.map_utils import create_demo_map, is_pose_valid
from src.metrics import angular_distance, compute_localization_metrics
from src.pose import Pose
from src.sensor_model import simulate_scan
from src.visualization import plot_ambiguity_map, plot_ea_output_on_ambiguity_map


def make_pose_from_row(row: dict[str, Any], prefix: str = "estimated") -> Pose:

    return Pose(
        x=float(row[f"{prefix}_x"]),
        y=float(row[f"{prefix}_y"]),
        theta=float(row[f"{prefix}_theta"]),
    )


def compute_scan_error_for_pose(
    pose: Pose,
    observed_scan: np.ndarray,
    occ_grid: np.ndarray,
    resolution: float,
    sensor_cfg: SensorConfig,
) -> float:

    predicted_scan = simulate_scan(
        pose=pose,
        occ_grid=occ_grid,
        resolution=resolution,
        sensor_cfg=sensor_cfg,
        rng=None,
        add_noise=False,
    )
    return normalized_scan_mse(
        observed=observed_scan,
        predicted=predicted_scan,
        sensor_max_range=sensor_cfg.max_range_m,
    )


def sample_valid_pose_grid(
    occ_grid: np.ndarray,
    resolution: float,
    num_xy_samples: int,
    num_theta_samples: int,
) -> list[Pose]:
 
    height, width = occ_grid.shape
    x_max = (width - 1) * resolution
    y_max = (height - 1) * resolution

    xs = np.linspace(0.0, x_max, num_xy_samples)
    ys = np.linspace(0.0, y_max, num_xy_samples)
    thetas = np.linspace(-math.pi, math.pi, num_theta_samples, endpoint=False)

    poses: list[Pose] = []
    for x in xs:
        for y in ys:
            if not is_pose_valid(float(x), float(y), occ_grid, resolution):
                continue
            for theta in thetas:
                poses.append(Pose(float(x), float(y), float(theta)))

    return poses


def compute_ambiguity_landscape(
    map_name: str,
    true_pose: Pose,
    pose_id: int,
    sensor_cfg: SensorConfig,
    num_xy_samples: int = AMBIGUITY_XY_SAMPLES,
    num_theta_samples: int = AMBIGUITY_THETA_SAMPLES,
    equivalence_percentile: float = AMBIGUITY_PERCENTILE,
    min_threshold: float = AMBIGUITY_MIN_THRESHOLD,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
 
    map_cfg = MapConfig()
    occ_grid = create_demo_map(map_name, map_cfg.width, map_cfg.height)

    observed_scan = simulate_scan(
        pose=true_pose,
        occ_grid=occ_grid,
        resolution=map_cfg.resolution,
        sensor_cfg=sensor_cfg,
        rng=None,
        add_noise=False,
    )

    candidates = sample_valid_pose_grid(
        occ_grid=occ_grid,
        resolution=map_cfg.resolution,
        num_xy_samples=num_xy_samples,
        num_theta_samples=num_theta_samples,
    )

    all_rows: list[dict[str, Any]] = []
    errors: list[float] = []

    for candidate in candidates:
        scan_error = compute_scan_error_for_pose(
            pose=candidate,
            observed_scan=observed_scan,
            occ_grid=occ_grid,
            resolution=map_cfg.resolution,
            sensor_cfg=sensor_cfg,
        )
        errors.append(scan_error)
        all_rows.append(
            {
                "map_name": map_name,
                "pose_id": pose_id,
                "x": candidate.x,
                "y": candidate.y,
                "theta": candidate.theta,
                "scan_error": scan_error,
            }
        )

    if not errors:
        raise RuntimeError(f"No valid sampled poses for map={map_name}, pose_id={pose_id}")

    percentile_threshold = float(np.percentile(errors, equivalence_percentile))
    threshold = max(percentile_threshold, min_threshold)

    equivalent_rows = [row for row in all_rows if float(row["scan_error"]) <= threshold]

    # Keep the equivalent raw CSV compact if the threshold returns many poses.
    equivalent_rows = sorted(equivalent_rows, key=lambda r: float(r["scan_error"]))
    if len(equivalent_rows) > AMBIGUITY_MAX_ROWS_PER_POSE:
        equivalent_rows = equivalent_rows[:AMBIGUITY_MAX_ROWS_PER_POSE]

    metadata = {
        "map_name": map_name,
        "pose_id": pose_id,
        "true_x": true_pose.x,
        "true_y": true_pose.y,
        "true_theta": true_pose.theta,
        "num_candidates": len(all_rows),
        "num_equivalent_candidates": len(equivalent_rows),
        "equivalence_percentile": equivalence_percentile,
        "percentile_threshold": percentile_threshold,
        "equivalence_threshold": threshold,
        "min_scan_error": float(np.min(errors)),
        "median_scan_error": float(np.median(errors)),
        "mean_scan_error": float(np.mean(errors)),
        "max_scan_error": float(np.max(errors)),
        "num_xy_samples": num_xy_samples,
        "num_theta_samples": num_theta_samples,
        "num_beams": sensor_cfg.num_beams,
    }

    # Attach threshold metadata to each equivalent row for easier analysis.
    for row in equivalent_rows:
        row.update(
            {
                "equivalence_threshold": threshold,
                "equivalence_percentile": equivalence_percentile,
                "true_x": true_pose.x,
                "true_y": true_pose.y,
                "true_theta": true_pose.theta,
            }
        )

    # Save a compact heatmap source too: for each sampled x/y keep the best theta.
    heatmap_rows = make_ambiguity_heatmap_rows(all_rows)
    heatmap_path = RESULTS_DIR / "experiment_5_ambiguity_heatmaps_raw.csv"
    append_rows_csv(heatmap_rows, heatmap_path)

    plot_ambiguity_map(
        map_name=map_name,
        true_pose=true_pose,
        pose_id=pose_id,
        heatmap_rows=heatmap_rows,
        equivalent_rows=equivalent_rows,
        threshold=threshold,
    )

    return equivalent_rows, metadata


def make_ambiguity_heatmap_rows(candidate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:

    best_by_xy: dict[tuple[float, float], dict[str, Any]] = {}
    for row in candidate_rows:
        key = (round(float(row["x"]), 6), round(float(row["y"]), 6))
        current = best_by_xy.get(key)
        if current is None or float(row["scan_error"]) < float(current["scan_error"]):
            best_by_xy[key] = row

    out: list[dict[str, Any]] = []
    for row in best_by_xy.values():
        out.append(
            {
                "map_name": row["map_name"],
                "pose_id": row["pose_id"],
                "x": row["x"],
                "y": row["y"],
                "best_theta": row["theta"],
                "min_scan_error": row["scan_error"],
            }
        )
    return out


def nearest_equivalent_pose_distance(
    pose: Pose,
    equivalent_rows: list[dict[str, Any]],
) -> tuple[float | None, float | None, float | None]:

    if not equivalent_rows:
        return None, None, None

    best_row = None
    best_distance = float("inf")

    for row in equivalent_rows:
        d_xy = math.hypot(pose.x - float(row["x"]), pose.y - float(row["y"]))
        d_theta = angular_distance(pose.theta, float(row["theta"]))
        # Combined distance used only to find the nearest sampled representative.
        combined = d_xy + 0.25 * d_theta
        if combined < best_distance:
            best_distance = combined
            best_row = row

    assert best_row is not None
    return (
        float(math.hypot(pose.x - float(best_row["x"]), pose.y - float(best_row["y"]))),
        float(angular_distance(pose.theta, float(best_row["theta"]))),
        float(best_row["scan_error"]),
    )


def classify_pose_by_scan_equivalence(
    pose: Pose,
    true_pose: Pose,
    observed_scan: np.ndarray,
    occ_grid: np.ndarray,
    resolution: float,
    sensor_cfg: SensorConfig,
    equivalence_threshold: float,
    equivalent_rows: list[dict[str, Any]],
) -> dict[str, Any]:

    metrics = compute_localization_metrics(estimated_pose=pose, true_pose=true_pose)
    output_scan_error = compute_scan_error_for_pose(
        pose=pose,
        observed_scan=observed_scan,
        occ_grid=occ_grid,
        resolution=resolution,
        sensor_cfg=sensor_cfg,
    )

    scan_equivalent = bool(output_scan_error <= equivalence_threshold)

    if metrics.success:
        label = "exact_correct"
    elif scan_equivalent:
        label = "ambiguous_correct"
    else:
        label = "true_failure"

    nearest_xy, nearest_theta, nearest_scan_error = nearest_equivalent_pose_distance(
        pose=pose,
        equivalent_rows=equivalent_rows,
    )

    return {
        "exact_success": metrics.success,
        "scan_equivalent_success": scan_equivalent,
        "failure_type": label,
        "output_scan_error": output_scan_error,
        "equivalence_threshold": equivalence_threshold,
        "nearest_equivalent_xy_distance": nearest_xy,
        "nearest_equivalent_theta_distance": nearest_theta,
        "nearest_equivalent_scan_error": nearest_scan_error,
    }


def classify_run_row_against_ambiguity(
    row: dict[str, Any],
    true_pose: Pose,
    equivalent_rows: list[dict[str, Any]],
    equivalence_metadata: dict[str, Any],
    sensor_cfg: SensorConfig,
) -> dict[str, Any]:

    map_cfg = MapConfig()
    occ_grid = create_demo_map(str(row["map_name"]), map_cfg.width, map_cfg.height)
    observed_scan = simulate_scan(
        pose=true_pose,
        occ_grid=occ_grid,
        resolution=map_cfg.resolution,
        sensor_cfg=sensor_cfg,
        rng=None,
        add_noise=False,
    )

    best_pose = make_pose_from_row(row, prefix="estimated")
    best_class = classify_pose_by_scan_equivalence(
        pose=best_pose,
        true_pose=true_pose,
        observed_scan=observed_scan,
        occ_grid=occ_grid,
        resolution=map_cfg.resolution,
        sensor_cfg=sensor_cfg,
        equivalence_threshold=float(equivalence_metadata["equivalence_threshold"]),
        equivalent_rows=equivalent_rows,
    )

    out = dict(row)
    out.update(best_class)
    out.update(
        {
            "ambiguity_num_candidates": equivalence_metadata["num_candidates"],
            "ambiguity_num_equivalent_candidates": equivalence_metadata["num_equivalent_candidates"],
            "ambiguity_percentile": equivalence_metadata["equivalence_percentile"],
            "ambiguity_min_scan_error": equivalence_metadata["min_scan_error"],
            "ambiguity_median_scan_error": equivalence_metadata["median_scan_error"],
        }
    )

    # Classify the top-k diverse hypotheses too, if present in the row.
    top_scan_equivalent_flags: list[bool] = []
    top_exact_flags: list[bool] = []
    top_failure_types: list[str] = []

    for rank in range(1, 6):
        if row.get(f"top{rank}_x") in (None, ""):
            continue
        top_pose = make_pose_from_row(row, prefix=f"top{rank}")
        cls = classify_pose_by_scan_equivalence(
            pose=top_pose,
            true_pose=true_pose,
            observed_scan=observed_scan,
            occ_grid=occ_grid,
            resolution=map_cfg.resolution,
            sensor_cfg=sensor_cfg,
            equivalence_threshold=float(equivalence_metadata["equivalence_threshold"]),
            equivalent_rows=equivalent_rows,
        )
        out[f"top{rank}_scan_equivalent_success"] = cls["scan_equivalent_success"]
        out[f"top{rank}_failure_type"] = cls["failure_type"]
        out[f"top{rank}_output_scan_error"] = cls["output_scan_error"]
        top_scan_equivalent_flags.append(bool(cls["scan_equivalent_success"]))
        top_exact_flags.append(bool(cls["exact_success"]))
        top_failure_types.append(str(cls["failure_type"]))

    out["top5_any_exact_success"] = any(top_exact_flags)
    out["top5_any_scan_equivalent_success"] = any(top_scan_equivalent_flags)
    out["top5_failure_types"] = ";".join(top_failure_types)

    return out

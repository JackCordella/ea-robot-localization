from __future__ import annotations
from typing import Any

import os
import matplotlib.pyplot as plt
import numpy as np
import math

from src.pose import Pose
from src.sensor_model import simulate_scan
from src.map_utils import create_demo_map
from config import MapConfig
from experiments.configs import MAP_NAMES, PLOTS_DIR


def _ensure_parent(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)



def plot_map_only(
    occ_grid: np.ndarray,
    title: str = "Map",
    out_path: str | None = None,
    show: bool = True,
) -> None:

    _ensure_parent(out_path)
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.imshow(
        occ_grid,
        cmap="gray_r",
        origin="lower",
        interpolation="nearest",
    )

    ax.set_title(title)
    ax.set_xlabel("x [grid cells]")
    ax.set_ylabel("y [grid cells]")
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_map_with_pose_and_population(
    occ_grid: np.ndarray,
    resolution: float,
    true_pose: Pose,
    estimated_pose: Pose,
    population: list[Pose],
    out_path: str,
) -> None:
    
    _ensure_parent(out_path)
    plt.figure(figsize=(8, 6))
    plt.imshow(occ_grid, origin="lower", cmap="gray_r")

    if population:
        xs = [p.x / resolution for p in population]
        ys = [p.y / resolution for p in population]
        plt.scatter(xs, ys, s=12, alpha=0.35, label="sampled poses")

    plt.scatter(true_pose.x / resolution, true_pose.y / resolution, s=80, marker="x", label="true pose")
    plt.scatter(estimated_pose.x / resolution, estimated_pose.y / resolution, s=60, marker="o", label="best pose")
    plt.legend(loc="upper right")
    plt.title("Map, true pose, and sampled population")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_scan_comparison(
    observed_scan: np.ndarray,
    candidate_pose: Pose,
    occ_grid: np.ndarray,
    resolution: float,
    sensor_cfg,
    out_path: str,
) -> None:
    
    _ensure_parent(out_path)
    
    predicted = simulate_scan(candidate_pose, occ_grid, resolution, sensor_cfg)
    
    plt.figure(figsize=(8, 4))
    plt.plot(observed_scan, label="observed")
    plt.plot(predicted, label="predicted")
    plt.xlabel("Beam index")
    plt.ylabel("Range [m]")
    plt.title("Observed vs predicted scan")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_fitness_history(
    best_history: list[float],
    mean_history: list[float],
    out_path: str,
) -> None:
    
    _ensure_parent(out_path)

    plt.figure(figsize=(8, 5))
    plt.plot(best_history, label="Best fitness")
    plt.plot(mean_history, label="Mean fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("EA fitness progression")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_ambiguity_map(
    map_name: str,
    true_pose: Pose,
    pose_id: int,
    heatmap_rows: list[dict[str, Any]],
    equivalent_rows: list[dict[str, Any]],
    threshold: float,
) -> None:

    map_cfg = MapConfig()
    occ_grid = create_demo_map(map_name, map_cfg.width, map_cfg.height)

    out_dir = PLOTS_DIR / "experiment_5_symmetry"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(
        occ_grid,
        cmap="gray_r",
        origin="upper",
        extent=[0, occ_grid.shape[1] * map_cfg.resolution, occ_grid.shape[0] * map_cfg.resolution, 0],
    )

    if heatmap_rows:
        xs = np.array([float(r["x"]) for r in heatmap_rows])
        ys = np.array([float(r["y"]) for r in heatmap_rows])
        errors = np.array([float(r["min_scan_error"]) for r in heatmap_rows])
        # Invert errors so brighter/higher values mean more equivalent.
        scores = 1.0 / (1.0 + errors)
        sc = ax.scatter(xs, ys, c=scores, s=14, alpha=0.70)
        fig.colorbar(sc, ax=ax, label="Best scan similarity over theta")

    if equivalent_rows:
        ex = [float(r["x"]) for r in equivalent_rows]
        ey = [float(r["y"]) for r in equivalent_rows]
        ax.scatter(ex, ey, marker="o", s=12, facecolors="none", edgecolors="black", linewidths=0.4, label="scan-equivalent samples")

    ax.scatter([true_pose.x], [true_pose.y], marker="*", s=180, label="true pose")
    ax.arrow(
        true_pose.x,
        true_pose.y,
        0.35 * math.cos(true_pose.theta),
        0.35 * math.sin(true_pose.theta),
        head_width=0.08,
        length_includes_head=True,
    )

    ax.invert_yaxis()

    ax.set_title(f"Scan-equivalent regions | pose {pose_id} | threshold={threshold:.4g}")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_dir / f"ambiguity_map_{map_name}_pose{pose_id}.png", dpi=180)
    plt.close(fig)



def plot_ea_output_on_ambiguity_map(
    row: dict[str, Any],
    equivalent_rows: list[dict[str, Any]],
) -> None:

    import matplotlib.pyplot as plt

    map_cfg = MapConfig()
    map_name = str(row["map_name"])
    pose_id = int(row["pose_id"])
    seed = int(row["seed"])
    occ_grid = create_demo_map(map_name, map_cfg.width, map_cfg.height)

    true_pose = Pose(float(row["true_x"]), float(row["true_y"]), float(row["true_theta"]))
    best_pose = Pose(float(row["estimated_x"]), float(row["estimated_y"]), float(row["estimated_theta"]))

    out_dir = PLOTS_DIR / "experiment_5_symmetry" / "ea_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(
        occ_grid,
        cmap="gray_r",
        origin="upper",
        extent=[0, occ_grid.shape[1] * map_cfg.resolution, occ_grid.shape[0] * map_cfg.resolution, 0],
    )

    if equivalent_rows:
        ex = [float(r["x"]) for r in equivalent_rows]
        ey = [float(r["y"]) for r in equivalent_rows]
        ax.scatter(ex, ey, s=12, alpha=0.35, label="scan-equivalent region")

    ax.scatter([true_pose.x], [true_pose.y], marker="*", s=180, label="true pose")
    ax.scatter([best_pose.x], [best_pose.y], marker="X", s=120, label=f"EA best ({row['failure_type']})")

    for rank in range(1, 6):
        if row.get(f"top{rank}_x") in (None, ""):
            continue
        ax.scatter(
            [float(row[f"top{rank}_x"])],
            [float(row[f"top{rank}_y"])],
            marker="o",
            s=45,
            label=f"top {rank}" if rank == 1 else None,
        )

    ax.invert_yaxis()
    ax.set_title(f"EA output vs ambiguity | pose {pose_id} | try {seed}")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_dir / f"ea_output_{map_name}_pose{pose_id}_seed{seed}.png", dpi=180)
    plt.close(fig)


def visualize_maps(map_names: list[str] = MAP_NAMES) -> None:
    
    map_cfg = MapConfig()
    for name in map_names:
        grid = create_demo_map(name, map_cfg.width, map_cfg.height)
        plot_map_only(
            occ_grid=grid,
            title=name,
            out_path=str(PLOTS_DIR / "maps" / f"{name}.png"),
            show=False,
        )
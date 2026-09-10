from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import math
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from PIL import Image

from config import EAConfig, MapConfig, SensorConfig
from experiments.configs import DEFAULT_PERCEPTION_WEIGHT, get_experiment_poses
from src.evolutionary_algorithm import EvolutionaryLocalizer
from src.fitness import evaluate_perception_aware_individual
from src.individual import Individual
from src.map_utils import create_demo_map
from src.perception import extract_scan_perception
from src.pose import Pose
from src.sensor_model import simulate_scan
from src.visualization import plot_map_with_pose_and_population, plot_scan_comparison, plot_fitness_history
from src.population import get_best_individual, get_population_stats


DEFAULT_OUTPUT_DIR = Path("results/plots/additional")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _beam_angles(sensor_cfg: SensorConfig) -> np.ndarray:
    """Return sensor beam angles relative to the robot heading."""
    return np.linspace(
        -math.radians(sensor_cfg.fov_deg) / 2.0,
        math.radians(sensor_cfg.fov_deg) / 2.0,
        sensor_cfg.num_beams,
    )


def plot_lidar_raycast_example(
    occ_grid: np.ndarray,
    resolution: float,
    pose: Pose,
    sensor_cfg: SensorConfig,
    out_path: Path,
) -> None:
    
    _ensure_dir(out_path.parent)

    scan = simulate_scan(
        pose=pose,
        occ_grid=occ_grid,
        resolution=resolution,
        sensor_cfg=sensor_cfg,
        rng=None,
        add_noise=False,
    )
    angles = _beam_angles(sensor_cfg)

    ray_segments: list[list[tuple[float, float]]] = []
    hit_xs: list[float] = []
    hit_ys: list[float] = []

    for rel_angle, measured_range in zip(angles, scan):
        beam_theta = pose.theta + rel_angle
        end_x = pose.x + measured_range * math.cos(beam_theta)
        end_y = pose.y + measured_range * math.sin(beam_theta)

        ray_segments.append(
            [
                (pose.x / resolution, pose.y / resolution),
                (end_x / resolution, end_y / resolution),
            ]
        )
        hit_xs.append(end_x / resolution)
        hit_ys.append(end_y / resolution)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(occ_grid, origin="lower", cmap="gray_r", interpolation="nearest")

    rays = LineCollection(ray_segments, linewidths=0.7, alpha=0.35)
    ax.add_collection(rays)
    ax.scatter(hit_xs, hit_ys, s=8, alpha=0.55, label="ray endpoints")

    robot_x = pose.x / resolution
    robot_y = pose.y / resolution
    ax.scatter(robot_x, robot_y, marker="*", s=150, label="robot pose")

    heading_len = 0.45 / resolution
    ax.arrow(
        robot_x,
        robot_y,
        heading_len * math.cos(pose.theta),
        heading_len * math.sin(pose.theta),
        head_width=1.8,
        head_length=2.2,
        length_includes_head=True,
    )

    ax.set_title("Simulated 2D LiDAR ray casting")
    ax.set_xlabel("x [grid cells]")
    ax.set_ylabel("y [grid cells]")
    ax.set_aspect("equal")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def run_ea_with_snapshots(
    occ_grid: np.ndarray,
    resolution: float,
    true_pose: Pose,
    sensor_cfg: SensorConfig,
    ea_cfg: EAConfig,
    snapshot_generations: Iterable[int],
) -> list[dict[int, list[Individual]], np.ndarray, list[float], list[float]]:
    
    rng = np.random.default_rng(ea_cfg.seed)

    observed_scan = simulate_scan(
        pose=true_pose,
        occ_grid=occ_grid,
        resolution=resolution,
        sensor_cfg=sensor_cfg,
        rng=rng,
        add_noise=True,
    )
    observed_perception = extract_scan_perception(
        observed_scan,
        sensor_max_range=sensor_cfg.max_range_m,
    )

    
    def evaluator(individual: Individual) -> None:
        evaluate_perception_aware_individual(
            individual=individual,
            observed_scan=observed_scan,
            observed_perception=observed_perception,
            occ_grid=occ_grid,
            resolution=resolution,
            sensor_cfg=sensor_cfg,
            perception_weight=ea_cfg.perception_weight,
            rng=rng,
        )

    localizer = EvolutionaryLocalizer(
        config=ea_cfg,
        occ_grid=occ_grid,
        resolution=resolution,
        rng=rng,
        individual_evaluator=evaluator,
    )

    requested = set(snapshot_generations)
    snapshots: dict[int, list[Individual]] = {}

    best_fitness_history = []
    mean_fitness_history = []

    population = localizer.initialize_population()
    for individual in population:
        evaluator(individual)

    if 0 in requested:
        snapshots[0] = [individual.copy() for individual in population]

    for generation in range(1, ea_cfg.generations + 1):
        
        stats = get_population_stats(population)
        best_fitness_history.append(stats.best_fitness)
        mean_fitness_history.append(stats.mean_fitness)

        population = localizer.make_next_generation(population)
        if generation in requested:
            snapshots[generation] = [individual.copy() for individual in population]

    return snapshots, observed_scan, best_fitness_history, mean_fitness_history


def save_scan_comparisons(
    snapshots: dict[int, list[Individual]],
    observed_scan: np.ndarray,
    occ_grid: np.ndarray,
    resolution: float,
    sensor_cfg: SensorConfig,
    generations: list[int],
    out_dir: Path,
) -> None:
    
    for generation in generations:
        best = get_best_individual(snapshots[generation])
        plot_scan_comparison(
            observed_scan=observed_scan,
            candidate_pose=best.pose,
            occ_grid=occ_grid,
            resolution=resolution,
            sensor_cfg=sensor_cfg,
            out_path=str(out_dir / f"scan_comparison_{generation}.png"),
        )


def save_population_snapshots(
    snapshots: dict[int, list[Individual]],
    occ_grid: np.ndarray,
    resolution: float,
    true_pose: Pose,
    out_dir: Path,
) -> list[Path]:

    frame_paths: list[Path] = []

    for generation in sorted(snapshots):
        population = snapshots[generation]
        best = get_best_individual(population)
        poses = [individual.pose for individual in population]

        if generation == max(snapshots):
            filename = "population_gen_final.png"
        else:
            filename = f"population_gen_{generation:03d}.png"

        out_path = out_dir / filename
        plot_map_with_pose_and_population(
            occ_grid=occ_grid,
            resolution=resolution,
            true_pose=true_pose,
            estimated_pose=best.pose,
            population=poses,
            out_path=str(out_path),
        )
        frame_paths.append(out_path)

    return frame_paths


def make_population_gif(
    frame_paths: list[Path],
    out_path: Path,
    duration_ms: int = 700,
) -> None:

    if not frame_paths:
        return

    frames = [Image.open(path).convert("P") for path in frame_paths]
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )


def generate_case_study_figures(
    map_name: str = "medium_room",
    pose_id: int = 2,
    seed: int = 5,
    out_dir: Path = DEFAULT_OUTPUT_DIR,
) -> None:

    _ensure_dir(out_dir)

    map_cfg = MapConfig(width=80, height=60, resolution=0.1)
    sensor_cfg = SensorConfig(
        num_beams=72,
        fov_deg=360.0,
        max_range_m=8.0,
        step_m=0.05,
        noise_std_m=0.0,
    )
    ea_cfg = replace(
        EAConfig(),
        population_size=100,
        generations=40,
        tournament_size=3,
        crossover_rate=0.9,
        mutation_rate=0.8,
        sigma_xy=0.35,
        sigma_theta=0.25,
        elite_size=3,
        perception_weight=DEFAULT_PERCEPTION_WEIGHT,
        seed=seed,
    )

    occ_grid = create_demo_map(map_name, map_cfg.width, map_cfg.height)
    true_pose = get_experiment_poses(map_name)[pose_id]

    plot_lidar_raycast_example(
        occ_grid=occ_grid,
        resolution=map_cfg.resolution,
        pose=true_pose,
        sensor_cfg=sensor_cfg,
        out_path=out_dir / "lidar_simulation_example.png",
    )

    snapshot_generations = [0, 2, 5, 10, 20, ea_cfg.generations]
    snapshots, observed_scan, best_fitness_history, mean_fitness_history  = run_ea_with_snapshots(
        occ_grid=occ_grid,
        resolution=map_cfg.resolution,
        true_pose=true_pose,
        sensor_cfg=sensor_cfg,
        ea_cfg=ea_cfg,
        snapshot_generations=snapshot_generations,
    )

    plot_fitness_history(
        best_history=best_fitness_history,
        mean_history=mean_fitness_history,
        out_path=out_dir / "ea_fitness_history.png"
    )


    save_scan_comparisons(
        snapshots=snapshots,
        observed_scan=observed_scan,
        occ_grid=occ_grid,
        resolution=map_cfg.resolution,
        sensor_cfg=sensor_cfg,
        generations=(0, 2, 5, 10, 20, ea_cfg.generations),
        out_dir=out_dir,
    )

    frame_paths = save_population_snapshots(
        snapshots=snapshots,
        occ_grid=occ_grid,
        resolution=map_cfg.resolution,
        true_pose=true_pose,
        out_dir=out_dir,
    )
    
    make_population_gif(
        frame_paths=frame_paths,
        out_path=out_dir / "population_convergence.gif",
    )

    print(f"Case-study figures saved in: {out_dir.resolve()}")


if __name__ == "__main__":
    generate_case_study_figures()

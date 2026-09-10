from __future__ import annotations

import numpy as np

from config import MapConfig, SensorConfig, EAConfig, RuntimeConfig
from src.map_utils import create_demo_map
from src.pose import Pose
from src.sensor_model import simulate_scan
from src.fitness import evaluate_perception_aware_individual
from src.perception import extract_scan_perception
from src.metrics import compute_localization_metrics
from src.visualization import (
    plot_map_with_pose_and_population,
    plot_scan_comparison,
    plot_fitness_history,
)
from src.evolutionary_algorithm import EvolutionaryLocalizer
from experiments.suite import run_exp
from case_study import generate_case_study_figures


def main() -> None:
    map_cfg = MapConfig()
    sensor_cfg = SensorConfig()
    ea_cfg = EAConfig()
    run_cfg = RuntimeConfig()

    rng = np.random.default_rng(ea_cfg.seed)

    occ_grid = create_demo_map(run_cfg.map_name, map_cfg.width, map_cfg.height)

    true_pose = Pose(5.2, 1.5, -1.2)
    observed_scan = simulate_scan(true_pose, occ_grid, map_cfg.resolution, sensor_cfg, rng)
    observed_perception = extract_scan_perception(
        observed_scan,
        sensor_max_range=sensor_cfg.max_range_m,
    )


    # Perception-aware evaluator for EA. It updates both fitness and individual.perception.
    def evaluator(individual) -> None:
        evaluate_perception_aware_individual(
            individual=individual,
            observed_scan=observed_scan,
            observed_perception=observed_perception,
            occ_grid=occ_grid,
            resolution=map_cfg.resolution,
            sensor_cfg=sensor_cfg,
            perception_weight=ea_cfg.perception_weight,
            rng=rng
        )

    # EA run
    ea = EvolutionaryLocalizer(
        config=ea_cfg,
        occ_grid=occ_grid,
        resolution=map_cfg.resolution,
        rng=rng,
        individual_evaluator=evaluator,
    )

    print("\n-------Evolutionary Algorithm--------\n")

    result = ea.run()

    best_ea = result.best_individual
    estimated_pose_ea = best_ea.pose
    

    print("\n=== EA Result ===")
    print(f"Best pose: {estimated_pose_ea}")
    print(f"Best fitness: {best_ea.fitness:.6f}")
    if best_ea.perception is not None:
        print(f"Best perception: {best_ea.perception}")


    metrics = compute_localization_metrics(
        estimated_pose=estimated_pose_ea,
        true_pose=true_pose,
    )

    print(f"Position error: {metrics.position_error:.4f} m")
    print(f"Theta error: {metrics.theta_error:.4f} rad")
    print(f"Success: {metrics.success}")

    plot_map_with_pose_and_population(
        occ_grid=occ_grid,
        resolution=map_cfg.resolution,
        true_pose=true_pose,
        estimated_pose=estimated_pose_ea,
        population=[ind.pose for ind in result.final_population],
        out_path="results/plots/ea_population.png",
    )

    plot_scan_comparison(
        observed_scan=observed_scan,
        candidate_pose=estimated_pose_ea,
        occ_grid=occ_grid,
        resolution=map_cfg.resolution,
        sensor_cfg=sensor_cfg,
        out_path="results/plots/ea_scan_comparison.png",
    )

    plot_fitness_history(
        best_history=result.best_fitness_history,
        mean_history=result.mean_fitness_history,
        out_path="results/plots/ea_fitness_history.png",
    )

    print("\nSaved plots:")
    print("- results/plots/ea_population.png")
    print("- results/plots/ea_scan_comparison.png")
    print("- results/plots/ea_fitness_history.png")


if __name__ == "__main__":
    generate_case_study_figures()
    #run_exp()
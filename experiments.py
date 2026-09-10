from __future__ import annotations

from experiments.suite import (
    run_experiment_1_perception_ablation,
    run_experiment_2_all_maps_benchmark,
    run_experiment_3_beam_sensitivity,
    run_experiment_4_parameter_tuning,
    run_experiment_5_symmetry_analysis,
    run_all_experiments,
    run_exp,
)

# Backwards-compatible alias for the old experiment 5 entry point.
run_experiment_5_topk_symmetry = run_experiment_5_symmetry_analysis


if __name__ == "__main__":
    run_experiment_5_symmetry_analysis()

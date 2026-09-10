from __future__ import annotations

import time
from typing import Any

from config import SensorConfig
from experiments.ambiguity import (
    classify_run_row_against_ambiguity,
    compute_ambiguity_landscape,
)
from experiments.configs import (
    BEAM_SENSITIVITY_MAPS,
    BEAM_VALUES,
    DEFAULT_FOV_DEG,
    DEFAULT_MAX_RANGE_M,
    DEFAULT_NOISE_STD_M,
    DEFAULT_NUM_BEAMS,
    MAP_NAMES,
    METHODS,
    PARAMETER_CONFIGS,
    PARAMETER_TUNING_MAPS,
    PARAMETER_TUNING_POSE_IDS,
    PLOTS_DIR,
    RESULTS_DIR,
    SEEDS,
    SYMMETRY_MAPS,
    SYMMETRY_POSE_IDS,
    get_experiment_poses,
    validate_experiment_poses,
)
from experiments.io import append_rows_csv, save_rows_csv
from experiments.runner import run_single_localization
from experiments.summary import write_summary_tables, write_symmetry_summary_tables
from src.visualization import plot_ea_output_on_ambiguity_map


def run_experiment_1_perception_ablation() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for map_name in MAP_NAMES:
        poses = get_experiment_poses(map_name)
        for method in METHODS:
            print(f"\n----- Experiment 1 | map={map_name} | method={method} -----\n")
            for pose_id, true_pose in enumerate(poses):
                for seed in SEEDS:
                    rows.append(
                        run_single_localization(
                            experiment_name="experiment_1_perception_ablation",
                            map_name=map_name,
                            true_pose=true_pose,
                            pose_id=pose_id,
                            seed=seed,
                            method=method,
                        )
                    )

    raw_path = RESULTS_DIR / "experiment_1_perception_ablation_raw.csv"
    save_rows_csv(rows, raw_path)
    write_summary_tables(rows, RESULTS_DIR / "experiment_1")
    return rows


def run_experiment_2_all_maps_benchmark() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for map_name in MAP_NAMES:
        print(f"\n----- Experiment 2 | all-map benchmark | map={map_name} -----\n")
        for pose_id, true_pose in enumerate(get_experiment_poses(map_name)):
            for seed in SEEDS:
                rows.append(
                    run_single_localization(
                        experiment_name="experiment_2_all_maps_benchmark",
                        map_name=map_name,
                        true_pose=true_pose,
                        pose_id=pose_id,
                        seed=seed,
                        method="perception_aware",
                    )
                )

    raw_path = RESULTS_DIR / "experiment_2_all_maps_benchmark_raw.csv"
    save_rows_csv(rows, raw_path)
    write_summary_tables(rows, RESULTS_DIR / "experiment_2")
    return rows


def run_experiment_3_beam_sensitivity() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for map_name in BEAM_SENSITIVITY_MAPS:
        print(f"\n----- Experiment 3 | beam sensitivity | map={map_name} -----\n")
        for pose_id, true_pose in enumerate(get_experiment_poses(map_name)):
            for num_beams in BEAM_VALUES:
                for seed in SEEDS:
                    rows.append(
                        run_single_localization(
                            experiment_name="experiment_3_beam_sensitivity",
                            map_name=map_name,
                            true_pose=true_pose,
                            pose_id=pose_id,
                            seed=seed,
                            method="perception_aware",
                            num_beams=num_beams,
                            config_id=f"beams_{num_beams}",
                        )
                    )

    raw_path = RESULTS_DIR / "experiment_3_beam_sensitivity_raw.csv"
    save_rows_csv(rows, raw_path)
    write_summary_tables(rows, RESULTS_DIR / "experiment_3")
    return rows


def run_experiment_4_parameter_tuning() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for cfg in PARAMETER_CONFIGS:
        config_id = cfg["config_id"]
        overrides = cfg["overrides"]
        print(f"\n----- Experiment 4 | parameter tuning | config={config_id} -----\n")

        for map_name in PARAMETER_TUNING_MAPS:
            poses = get_experiment_poses(map_name)
            for pose_id in PARAMETER_TUNING_POSE_IDS:
                true_pose = poses[pose_id]
                for seed in SEEDS:
                    row = run_single_localization(
                        experiment_name="experiment_4_parameter_tuning",
                        map_name=map_name,
                        true_pose=true_pose,
                        pose_id=pose_id,
                        seed=seed,
                        method="perception_aware",
                        ea_cfg_overrides=overrides,
                        config_id=config_id,
                    )
                    # Store the symbolic override explicitly for easier report tables.
                    row["ea_overrides"] = str(overrides)
                    rows.append(row)

    raw_path = RESULTS_DIR / "experiment_4_parameter_tuning_raw.csv"
    save_rows_csv(rows, raw_path)
    write_summary_tables(rows, RESULTS_DIR / "experiment_4")
    return rows


def run_experiment_5_symmetry_analysis() -> list[dict[str, Any]]:

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Remove append-style files from previous Experiment 5 runs to avoid mixing results.
    for old_file in [
        RESULTS_DIR / "experiment_5_ambiguity_heatmaps_raw.csv",
        RESULTS_DIR / "experiment_5_equivalent_pose_regions_raw.csv",
    ]:
        if old_file.exists():
            old_file.unlink()

    sensor_cfg = SensorConfig(
        num_beams=DEFAULT_NUM_BEAMS,
        fov_deg=DEFAULT_FOV_DEG,
        max_range_m=DEFAULT_MAX_RANGE_M,
        noise_std_m=DEFAULT_NOISE_STD_M,
    )

    all_rows: list[dict[str, Any]] = []
    all_equivalent_rows: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []

    for map_name in SYMMETRY_MAPS:
        poses = get_experiment_poses(map_name)
        selected_pose_ids = SYMMETRY_POSE_IDS.get(map_name, list(range(len(poses))))

        for pose_id in selected_pose_ids:
            true_pose = poses[pose_id]
            print(f"\n----- Experiment 5 | ambiguity map | map={map_name} | pose={pose_id} -----\n")

            equivalent_rows, metadata = compute_ambiguity_landscape(
                map_name=map_name,
                true_pose=true_pose,
                pose_id=pose_id,
                sensor_cfg=sensor_cfg,
            )
            all_equivalent_rows.extend(equivalent_rows)
            metadata_rows.append(metadata)
            append_rows_csv(equivalent_rows, RESULTS_DIR / "experiment_5_equivalent_pose_regions_raw.csv")

            for seed in SEEDS:
                print(f"\n----- Experiment 5 | EA run | map={map_name} | pose={pose_id} | seed={seed} -----\n")
                base_row = run_single_localization(
                    experiment_name="experiment_5_symmetry_analysis",
                    map_name=map_name,
                    true_pose=true_pose,
                    pose_id=pose_id,
                    seed=seed,
                    method="perception_aware",
                    compute_top_k=True,
                    top_k=5,
                    top_k_min_dist_xy=0.75,
                    top_k_min_dist_theta=0.50,
                )

                classified_row = classify_run_row_against_ambiguity(
                    row=base_row,
                    true_pose=true_pose,
                    equivalent_rows=equivalent_rows,
                    equivalence_metadata=metadata,
                    sensor_cfg=sensor_cfg,
                )
                all_rows.append(classified_row)

                # Only plot a few representative failures/ambiguous cases to avoid too many files.
                if seed in (0, 1) or classified_row["failure_type"] != "exact_correct":
                    plot_ea_output_on_ambiguity_map(classified_row, equivalent_rows)

    save_rows_csv(all_rows, RESULTS_DIR / "experiment_5_symmetry_analysis_raw.csv")
    save_rows_csv(metadata_rows, RESULTS_DIR / "experiment_5_ambiguity_metadata.csv")
    save_rows_csv(all_equivalent_rows, RESULTS_DIR / "experiment_5_equivalent_pose_regions_raw.csv")
    write_symmetry_summary_tables(all_rows)

    return all_rows


def run_all_experiments() -> None:


    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    validate_experiment_poses()

    all_rows: list[dict[str, Any]] = []
    all_rows.extend(run_experiment_1_perception_ablation())
    all_rows.extend(run_experiment_2_all_maps_benchmark())
    all_rows.extend(run_experiment_3_beam_sensitivity())
    all_rows.extend(run_experiment_4_parameter_tuning())
    all_rows.extend(run_experiment_5_symmetry_analysis())

    save_rows_csv(all_rows, RESULTS_DIR / "all_experiments_raw.csv")
    write_summary_tables(all_rows, RESULTS_DIR / "all_experiments")

    elapsed = time.perf_counter() - start
    print(f"\nCompleted full experiment suite in {elapsed / 60.0:.2f} minutes.")


def run_exp() -> None:
    run_experiment_5_symmetry_analysis()

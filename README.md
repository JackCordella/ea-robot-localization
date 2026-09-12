# EA Localization Project

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22728714.svg)](https://doi.org/10.5281/zenodo.22728714)

Static global robot localization in a known 2D occupancy map using an Evolutionary Algorithm.

The project addresses a simplified kidnapped-robot/localization setting: the robot is static, the map is already known, and the algorithm receives a single range scan. The goal is to estimate the robot pose

```text
(x, y, theta)
```

by searching the continuous pose space and selecting the pose whose simulated scan is most compatible with the observed scan.

The implementation is intentionally modular. The core localization logic is in `src/`, while the experimental protocol, large-scale runs, CSV generation, summaries, and ambiguity analysis are in `experiments/`.

The full write-up is in
[`report/Report_EA_Robot_Localization_Cordella.pdf`](report/Report_EA_Robot_Localization_Cordella.pdf).

Install with `pip install -r requirements.txt`, then see [section 4](#4-how-to-run-the-project) — **note that `python main.py` does not run `main()`** by default.

---

## 1. Project idea

The localization problem is formulated as an optimization problem.

Given:

- a known 2D occupancy grid map;
- a true robot pose, used only to generate a synthetic observation during experiments;
- a simulated range observation from that true pose;
- a population of candidate poses.

The Evolutionary Algorithm searches for candidate poses that maximize a fitness function. A candidate pose is considered good when the scan simulated from that pose is close to the observed scan.

In other words, the project does not implement a full Bayesian Monte Carlo Localization pipeline. There is no motion model, no temporal tracking, and no particle filtering update over time. Instead, the project studies whether an evolutionary population-based optimizer can recover plausible static poses from a single observation.

This is especially useful for analyzing ambiguous environments, where multiple distinct poses may produce very similar range scans.

---

## 2. Main workflow

The complete pipeline is:

1. Build or load a known occupancy grid map.
2. Select a true pose inside free space.
3. Simulate a range scan from the true pose.
4. Extract optional perception-level scan features.
5. Initialize a population of valid candidate poses.
6. Evaluate each candidate with a scan-based or perception-aware fitness function.
7. Apply tournament selection, crossover, mutation, and elitism.
8. Repeat the evolutionary loop for a fixed number of generations.
9. Return the best pose and the final population.
10. Evaluate the result using position and orientation errors.
11. Save plots, raw CSV results, and summary tables for the report.

---

## 3. Repository structure

```text
ea_localization_project/
|
|-- main.py
|-- case_study.py
|-- config.py
|-- experiments.py
|-- requirements.txt
|-- README.md
|
|-- maps/
|   |-- simple_room.png
|   |-- easy_room.png
|   |-- medium_room.png
|   |-- complex_room.png
|   |-- corridor.png
|
|-- src/
|   |-- __init__.py
|   |-- pose.py
|   |-- individual.py
|   |-- map_utils.py
|   |-- sensor_model.py
|   |-- perception.py
|   |-- fitness.py
|   |-- population.py
|   |-- evolutionary_algorithm.py
|   |-- metrics.py
|   |-- visualization.py
|
|-- experiments/
|   |-- __init__.py
|   |-- configs.py
|   |-- runner.py
|   |-- suite.py
|   |-- summary.py
|   |-- ambiguity.py
|   |-- topk.py
|   |-- io.py
|
|-- results/
|   |-- logs/
|   |-- plots/
```

Some folders such as `__pycache__`, `.vscode`, `venv`, or system files like `.DS_Store` are not conceptually part of the project logic.

---

## 4. How to run the project

> **Read this first.** `main.py` defines `main()`, but its `if __name__ == "__main__"` block
> calls `generate_case_study_figures()` instead. So `python main.py` regenerates the
> case-study figures — it does **not** run the single localization demo. Which entry point
> runs is selected by editing that block, as described below.

**Python 3.10 or newer is required** — the pinned dependencies have no wheels for 3.9,
so on a machine whose default `python3` is older (macOS ships 3.9) the install fails with
`Could not find a version that satisfies the requirement contourpy==1.3.2`. Check with
`python3 --version` first.

Install the dependencies:

```bash
pip install -r requirements.txt
```

Then run:

```bash
python main.py
```

At the bottom of `main.py`, the executable block controls what is currently launched:

```python
if __name__ == "__main__":
    generate_case_study_figures()
    # run_exp()
```

This means that, in the current version, running `python main.py` generates the visual case-study figures. To run the single EA localization demo instead, uncomment `main()` and comment the other calls. To run the experiment suite entry point, uncomment `run_exp()` or call the desired function from `experiments/suite.py`.

The most common entry points are:

| Goal | Entry point |
|---|---|
| Run one simple EA localization example | `main()` in `main.py` |
| Generate report/case-study figures | `generate_case_study_figures()` in `case_study.py` |
| Run the configured experiment entry point | `run_exp()` in `experiments/suite.py` |
| Run all experiments | `run_all_experiments()` in `experiments/suite.py` |

---

## 5. Core configuration: `config.py`

`config.py` contains the main dataclasses used across the project. It centralizes parameters and result containers so that the rest of the code can pass structured objects instead of long argument lists.

### `MapConfig`

Defines the occupancy grid size and metric resolution:

- `width`: number of grid columns;
- `height`: number of grid rows;
- `resolution`: meters per grid cell.

The default map is an `80 x 60` grid with `0.1 m` per cell.

### `SensorConfig`

Defines the simulated range sensor:

- `num_beams`: number of range beams;
- `fov_deg`: field of view in degrees;
- `max_range_m`: maximum measurable range;
- `step_m`: ray-casting step size;
- `noise_std_m`: Gaussian noise standard deviation.

### `EAConfig`

Defines the Evolutionary Algorithm parameters:

- `population_size`: number of individuals;
- `generations`: number of evolutionary iterations;
- `tournament_size`: number of candidates sampled during tournament selection;
- `crossover_rate`: probability of applying crossover;
- `mutation_rate`: probability of mutating a child;
- `sigma_xy`: Gaussian mutation scale for position;
- `sigma_theta`: Gaussian mutation scale for orientation;
- `elite_size`: number of best individuals copied directly into the next generation;
- `perception_weight`: weight of the perception feature error in the combined fitness;
- `seed`: random seed.

### Result dataclasses

`PopulationStats`, `FitnessBreakdown`, `EARunResult`, `RuntimeConfig`, and `LocalizationMetrics` are used to store intermediate statistics, fitness components, final EA output, runtime settings, and localization quality metrics.

---

## 6. Core source code: `src/`

The `src/` folder contains the reusable localization and evolutionary-computing logic. These modules are independent from the experiment protocol and can be reused in different scripts.

---

### 6.1 `src/pose.py`

Defines the basic pose representation.

#### `Pose`

A frozen dataclass representing a robot pose:

```python
Pose(x: float, y: float, theta: float)
```

- `x`: robot x-position in meters;
- `y`: robot y-position in meters;
- `theta`: robot heading in radians.

#### `Pose.wrapped()`

Returns a copy of the pose with `theta` normalized to the interval `[-pi, pi]`. This is important because many operations on angles can produce equivalent but numerically different orientations.

---

### 6.2 `src/individual.py`

Defines the EA individual.

#### `Individual`

An individual is a candidate solution. Its genotype/phenotype is the robot pose itself:

```python
Individual(
    pose: Pose,
    fitness: float,
    perception: ScanPerception | None
)
```

The project uses a real-valued representation: each candidate solution directly stores `(x, y, theta)`.

#### `Individual.copy()`

Creates a safe copy of an individual. This is used during selection, elitism, and population replacement to avoid accidental modifications of parent individuals.

---

### 6.3 `src/map_utils.py`

Contains occupancy-grid utilities.

#### `create_demo_map(name, width, height)`

Creates one of the predefined map layouts used in the experiments. Supported map names include:

- `simple_room`
- `easy_room`
- `medium_room`
- `complex_room`
- `corridor`
- `corridor_2`

The maps are represented as NumPy arrays where free space and obstacles are encoded as grid values. The exact geometry is generated procedurally in code, while the `maps/` folder stores visual map images.

#### `world_to_grid(x, y, resolution)`

Converts metric world coordinates into grid indices. This is necessary because poses and sensor computations are expressed in meters, while occupancy checks are performed on grid cells.

#### `get_world_bounds(occ_grid, resolution)`

Returns the valid continuous bounds for `x`, `y`, and `theta`. These bounds are used when initializing the population and clipping mutated poses.

#### `is_inside_map(row, col, occ_grid)`

Checks whether a grid index is inside the occupancy grid.

#### `is_pose_valid(x, y, occ_grid, resolution)`

Checks whether a continuous pose position is valid. A pose is valid if it lies inside the map and does not fall inside an occupied cell. This function is used in population initialization, crossover validation, mutation validation, and experiment pose validation.

---

### 6.4 `src/sensor_model.py`

Implements the simulated range sensor.

#### `simulate_scan(pose, occ_grid, resolution, sensor_cfg, rng, add_noise)`

Simulates a 2D range scan from a candidate pose using ray casting.

For each beam:

1. The beam angle is computed from the robot heading and the sensor field of view.
2. The ray is advanced step by step through the map.
3. The range stops when the ray hits an obstacle, exits the map, or reaches `max_range_m`.
4. Optional Gaussian noise is added if `add_noise=True`.

This function is central to the project because both the observed scan and all candidate scans are generated through the same sensor model.

---

### 6.5 `src/perception.py`

Extracts compact scan-level descriptors from a raw range scan.

#### `ScanPerception`

A dataclass containing summary features:

- `mean_range`
- `std_range`
- `min_range`
- `max_range`
- `roughness`
- `max_range_ratio`

These features are used to build a perception-aware fitness term. The idea is that two scans can be compared not only beam-by-beam, but also through global structural descriptors.

#### `ScanPerception.as_array()`

Converts the feature object into a NumPy array, making it easier to compute distances.

#### `extract_scan_perception(scan, sensor_max_range)`

Computes the perception features from a scan. In particular, `roughness` summarizes local scan variation, while `max_range_ratio` measures how many beams saturate at maximum range.

#### `perception_distance(observed, predicted, sensor_max_range)`

Computes a normalized distance between the perception features of the observed scan and the predicted scan.

---

### 6.6 `src/fitness.py`

Defines how candidate poses are evaluated.

#### `scan_mse(observed, predicted)`

Computes the mean squared error between two scans.

#### `normalized_scan_mse(observed, predicted, sensor_max_range)`

Normalizes the scan MSE by the square of the sensor maximum range. This makes the error scale more stable across different sensor settings.

#### `perception_aware_fitness_from_scan(...)`

This is the main fitness function.

Given a candidate pose, it:

1. rejects invalid poses with zero fitness;
2. simulates the scan expected from that pose;
3. extracts perception features from the predicted scan;
4. computes the normalized scan error;
5. computes the perception feature error;
6. combines the two errors as:

```text
combined_error = (1 - perception_weight) * scan_error
                 + perception_weight * perception_error
```

7. converts the error into a maximization fitness:

```text
fitness = 1 / (1 + combined_error)
```

A higher fitness means a better match with the observed scan.

#### `evaluate_perception_aware_individual(...)`

Wrapper around the previous function. It evaluates an `Individual`, updates its `fitness`, stores the predicted `perception`, and returns a `FitnessBreakdown` containing the scan error, perception error, and final fitness.

This function is used by the EA through an injected evaluator callback.

---

### 6.7 `src/population.py`

Contains population-level utilities.

#### `initialize_population(...)`

Samples random valid poses inside the map bounds. Each sampled pose must pass `is_pose_valid`. The function returns a list of unevaluated `Individual` objects.

#### `sort_population(population)`

Sorts individuals by decreasing fitness.

#### `get_best_individual(population)`

Returns the individual with maximum fitness.

#### `get_population_stats(population)`

Computes best, mean, and worst fitness values. These statistics are stored across generations to plot convergence curves.

---

### 6.8 `src/evolutionary_algorithm.py`

Implements the core Evolutionary Algorithm.

#### `wrap_angle(theta)`

Normalizes an angle to `[-pi, pi]`.

#### `EvolutionaryLocalizer`

Main EA class. It receives:

- `EAConfig`: algorithm parameters;
- `occ_grid`: the map;
- `resolution`: map resolution;
- `rng`: NumPy random generator;
- `individual_evaluator`: callback used to compute fitness.

The evaluator is injected from outside because the EA should not know the specific observation being used. This keeps the EA generic: it only knows how to evolve individuals, while the caller defines how individuals are evaluated.

#### `initialize_population()`

Delegates to `src.population.initialize_population`, using the map bounds computed from the occupancy grid.

#### `tournament_select(population)`

Implements tournament selection. It randomly samples `tournament_size` individuals and returns a copy of the fittest one.

Larger tournament sizes increase selection pressure. Smaller tournament sizes preserve more diversity.

#### `crossover(parent1, parent2, max_attempts=3)`

Creates a child from two parents.

The position coordinates use a blend-style crossover: the child coordinate is sampled in a range determined by the two parent coordinates. The orientation uses a circular blending strategy based on sine and cosine, avoiding discontinuities around `-pi` and `pi`.

The function tries to generate a valid child pose. If all attempts fail, it falls back to copying the fitter parent.

#### `mutate(individual, max_attempts=3)`

Applies Gaussian mutation:

- `x` and `y` are perturbed with standard deviation `sigma_xy`;
- `theta` is perturbed with standard deviation `sigma_theta`;
- `x` and `y` are clipped to map bounds;
- `theta` is wrapped to `[-pi, pi]`;
- the mutated pose is accepted only if it is valid.

If all mutation attempts produce invalid poses, the original pose is restored.

#### `make_next_generation(population)`

Creates the next generation:

1. Sort the population by fitness.
2. Copy the best `elite_size` individuals directly into the next population.
3. Repeatedly select parents using tournament selection.
4. Apply crossover and mutation.
5. Evaluate each child.
6. Stop when the new population reaches `population_size`.

This is a generational EA with elitism.

#### `run(show_progress=False, progress_desc=None)`

Runs the complete EA:

1. initialize the population;
2. evaluate all individuals;
3. for each generation, store best and mean fitness;
4. create the next generation;
5. return an `EARunResult` containing the best individual, final population, and fitness histories.

The optional progress bar is useful during large experiment batches.

---

### 6.9 `src/metrics.py`

Defines localization evaluation metrics.

#### `angular_distance(a, b)`

Computes the smallest angular distance between two headings.

#### `compute_localization_metrics(estimated_pose, true_pose, position_threshold, theta_threshold)`

Computes:

- Euclidean position error;
- angular error;
- Boolean success flag.

A run is successful if both the position error and the orientation error are below the chosen thresholds.

---

### 6.10 `src/visualization.py`

Contains plotting utilities for maps, populations, scans, convergence curves, and ambiguity analysis.

Important functions:

- `plot_map_only(...)`: saves a map image;
- `plot_map_with_pose_and_population(...)`: shows map, true pose, estimated pose, and final population;
- `plot_scan_comparison(...)`: compares observed scan and scan predicted by a candidate pose;
- `plot_fitness_history(...)`: plots best and mean fitness over generations;
- `plot_ambiguity_map(...)`: visualizes scan-equivalent regions for Experiment 5;
- `plot_ea_output_on_ambiguity_map(...)`: overlays EA output on ambiguity regions;
- `visualize_maps(...)`: utility for inspecting the available maps.

This module is only responsible for visualization. It should not contain experiment logic or EA logic.

---

## 7. Main script: `main.py`

`main.py` is a lightweight executable script for interactive runs.

The `main()` function performs a single localization demo:

1. creates default map, sensor, EA, and runtime configurations;
2. creates the selected map with `create_demo_map`;
3. defines a fixed true pose;
4. simulates the observed scan;
5. extracts perception features from the observed scan;
6. defines an internal `evaluator(individual)` function;
7. creates an `EvolutionaryLocalizer`;
8. runs the EA;
9. computes localization metrics;
10. saves plots into `results/plots/`.

The internal evaluator is important because it closes over the observed scan, observed perception, map, sensor configuration, and perception weight. This allows the EA class to remain independent from the experimental setup.

Generated plots from `main()` include:

```text
results/plots/ea_population.png
results/plots/ea_scan_comparison.png
results/plots/ea_fitness_history.png
```

---

## 8. Visual case study: `case_study.py`

`case_study.py` creates additional figures for the report appendix or visual explanation of the algorithm.

It is not part of the core EA implementation. Its goal is to produce interpretable visual material.

Important functions:

### `plot_lidar_raycast_example(...)`

Shows how simulated LiDAR rays are cast from a pose inside the map. Useful for explaining how the observation is generated.

### `run_ea_with_snapshots(...)`

Runs the EA while saving population snapshots at selected generations. This is useful for visualizing how the population moves from broad exploration to convergence around promising pose regions.

### `save_scan_comparisons(...)`

Saves scan-comparison plots for selected generations.

### `save_population_snapshots(...)`

Saves map plots with the population at selected generations.

### `make_population_gif(...)`

Builds an animated GIF from saved population frames.

### `generate_case_study_figures(...)`

High-level function that generates the full visual case-study output, including ray-casting examples, EA snapshots, scan comparisons, fitness history, and GIFs.

Default output folder:

```text
results/plots/additional/
```

---

## 9. Experiments package: `experiments/`

The `experiments/` package contains the systematic evaluation code. It separates research experiments from the reusable EA implementation.

---

### 9.1 `experiments/configs.py`

Defines shared experiment constants.

Important constants include:

- `MAP_NAMES`: maps used in the main benchmark;
- `METHODS`: `scan_only` and `perception_aware`;
- `SEEDS`: independent stochastic runs;
- `BEAM_VALUES`: number of beams tested in beam-sensitivity experiments;
- `PARAMETER_CONFIGS`: EA parameter variants for tuning analysis;
- `RESULTS_DIR`: CSV output folder;
- `PLOTS_DIR`: plot output folder;
- ambiguity-analysis settings such as `AMBIGUITY_XY_SAMPLES`, `AMBIGUITY_THETA_SAMPLES`, and `AMBIGUITY_PERCENTILE`.

#### `get_experiment_poses(map_name)`

Returns the fixed true poses used for each map. This ensures experiments are reproducible and comparable.

#### `validate_experiment_poses()`

Checks that all configured poses are valid free-space poses in their corresponding maps.

---

### 9.2 `experiments/runner.py`

Contains the function that runs one localization trial.

#### `run_single_localization(...)`

This is the central experiment runner. It executes one full EA run for a given:

- experiment name;
- map;
- true pose;
- pose id;
- random seed;
- method;
- sensor configuration;
- EA parameter override.

The function:

1. builds the map;
2. validates the true pose;
3. simulates the observed scan;
4. extracts observed perception features;
5. selects the effective perception weight;
6. defines the individual evaluator;
7. runs the EA;
8. computes localization metrics;
9. recomputes the final fitness breakdown;
10. stores all relevant values in a dictionary row.

The returned row is designed to be saved directly into CSV files. It includes true pose, estimated pose, errors, success flag, scan/perception errors, runtime, EA parameters, sensor parameters, and optional top-k mode information.

---

### 9.3 `experiments/suite.py`

Defines the high-level experiment suite.

#### `run_experiment_1_perception_ablation()`

Compares:

- `scan_only`
- `perception_aware`

across all maps, all configured poses, and all seeds. This experiment tests whether the perception feature term improves localization robustness.

#### `run_experiment_2_all_maps_benchmark()`

Runs the main benchmark using the perception-aware method over all maps and poses.

#### `run_experiment_3_beam_sensitivity()`

Studies the effect of sensor angular resolution by varying the number of beams. It is useful for understanding whether the algorithm depends strongly on dense scan information.

#### `run_experiment_4_parameter_tuning()`

Tests selected EA parameter configurations, such as population size, number of generations, mutation scale, and tournament size.

This is not a full automatic hyperparameter optimizer. It is a controlled parameter-sensitivity experiment.

#### `run_experiment_5_symmetry_analysis()`

Analyzes ambiguous localization cases. It first computes scan-equivalent pose regions and then compares EA outputs against those regions.

This experiment distinguishes between:

- exact localization success;
- failure caused by convergence to a scan-equivalent pose;
- failure caused by a genuinely wrong pose.

This is important because, in symmetric or repetitive maps, a geometrically different pose can still be observationally plausible.

#### `run_all_experiments()`

Runs all experiments and writes combined raw and summary CSV files.

#### `run_exp()`

Current convenience entry point. In the current code, it calls Experiment 5.

---

### 9.4 `experiments/ambiguity.py`

Implements the ambiguity and scan-equivalence analysis used in Experiment 5.

#### `sample_valid_pose_grid(...)`

Samples a grid of valid poses across the map and across multiple orientations.

#### `compute_scan_error_for_pose(...)`

Computes the normalized scan error between the observed scan and the scan predicted from a candidate pose.

#### `compute_ambiguity_landscape(...)`

Builds an ambiguity landscape for one true pose:

1. simulate the observed scan;
2. sample valid candidate poses;
3. compute scan error for each candidate;
4. define an equivalence threshold using a percentile rule and a minimum threshold;
5. keep candidate poses whose scan error is below the threshold;
6. save heatmap rows;
7. generate an ambiguity plot.

The output is a set of scan-equivalent poses plus metadata describing the ambiguity landscape.

#### `make_ambiguity_heatmap_rows(...)`

Compresses the ambiguity data by keeping, for each sampled `(x, y)`, the best orientation error value. This is used to create readable heatmaps.

#### `nearest_equivalent_pose_distance(...)`

Finds how close a pose is to the nearest scan-equivalent pose region.

#### `classify_pose_by_scan_equivalence(...)`

Classifies an estimated pose according to whether it is exactly correct, scan-equivalent, or incorrect.

#### `classify_run_row_against_ambiguity(...)`

Takes a normal experiment result row and enriches it with ambiguity-aware labels and distances.

---

### 9.5 `experiments/topk.py`

Contains utilities for evaluating multiple candidate modes in the final EA population.

#### `get_top_k_diverse_individuals(...)`

Selects the best individuals while enforcing minimum distance constraints in position and orientation. This avoids returning many nearly identical individuals from the same mode.

#### `top_k_success(...)`

Checks whether the correct pose appears within the top-k diverse candidates.

This is useful in ambiguous maps, where the best single individual may not tell the full story.

---

### 9.6 `experiments/summary.py`

Generates summary CSV tables from raw experiment rows.

Important functions:

- `summarize_rows(...)`: computes grouped statistics;
- `write_summary_tables(...)`: writes standard summary tables;
- `summarize_symmetry_rows(...)`: computes ambiguity-aware summaries;
- `write_symmetry_summary_tables(...)`: writes Experiment 5 summaries.

Typical grouped outputs include summaries by map, method, pose id, number of beams, and configuration id.

---

### 9.7 `experiments/io.py`

Small CSV utility module.

- `save_rows_csv(...)`: writes a list of dictionaries to a CSV file;
- `append_rows_csv(...)`: appends rows and writes the header only when the target file is new.

---

## 10. Maps

The project uses several procedural map layouts:

- `simple_room`: simple environment with low ambiguity;
- `easy_room`: slightly richer room layout;
- `medium_room`: intermediate structure;
- `complex_room`: more obstacles and stronger ambiguity potential;
- `corridor`: repetitive corridor-like structure;
- `corridor_2`: additional corridor variant used for ambiguity/symmetry case studies.

The `maps/` folder stores visual PNG versions of several maps. The actual occupancy grids used by the algorithm are created by `create_demo_map(...)`.

---

## 11. Results and outputs

The project writes outputs mainly into:

```text
results/logs/
results/plots/
```

### `results/logs/`

Contains raw and summarized CSV files, for example:

- raw experiment rows;
- summaries grouped by map;
- summaries grouped by method;
- summaries grouped by configuration;
- ambiguity metadata;
- scan-equivalent pose regions.

Raw CSV files are useful for debugging and further analysis. Summary CSV files are more suitable for tables in the final report.

### `results/plots/`

Contains visual outputs such as:

- population plots;
- scan comparisons;
- fitness-history plots;
- ambiguity heatmaps;
- additional case-study figures;
- GIFs showing population evolution.

---

## 12. Conceptual interpretation of the EA

The EA components are mapped to the localization problem as follows:

| EA concept | Project meaning |
|---|---|
| Individual | Candidate robot pose `(x, y, theta)` |
| Population | Set of possible robot poses |
| Fitness | Compatibility between observed scan and predicted scan |
| Selection | Prefer poses with better scan compatibility |
| Crossover | Combine pose coordinates from two promising candidates |
| Mutation | Locally perturb pose coordinates and orientation |
| Elitism | Preserve the best candidate poses across generations |
| Termination | Stop after a fixed number of generations |

The population-based nature of the EA is important because the localization landscape can be multimodal. In symmetric environments, multiple different poses can produce similar sensor readings. A final population can therefore contain useful information beyond only the single best pose.

---

## 13. Fitness design

The project supports two evaluation modes:

### Scan-only mode

Only the beam-wise normalized scan error is used:

```text
combined_error = scan_error
```

This is the most direct observation model.

### Perception-aware mode

The fitness combines raw scan matching and global scan descriptors:

```text
combined_error = (1 - perception_weight) * scan_error
                 + perception_weight * perception_error
```

This mode is designed to test whether simple global scan features help guide the search, especially when raw scan matching alone is noisy or ambiguous.

In both cases, the final value is converted into a maximization objective:

```text
fitness = 1 / (1 + combined_error)
```

---

## 14. Important implementation choices

### Validity checks

Candidate poses are checked with `is_pose_valid(...)` before being accepted. This prevents individuals from being placed inside walls or outside the map.

### Angle wrapping

Angles are always wrapped to `[-pi, pi]` after mutation or pose normalization. This avoids artificial discontinuities in orientation.

### Circular orientation crossover

The crossover operator blends orientation using sine and cosine rather than directly averaging angles. This avoids errors near the `pi`/`-pi` boundary.

### Elitism

The best individuals are copied into the next generation unchanged. This prevents losing the best discovered solution due to stochastic crossover or mutation.

### Reproducibility

Experiment runs use explicit seeds from `SEEDS`. This makes stochastic results reproducible and allows multiple independent trials.


---

## 15. Current scientific experiments

The project currently supports five main experiment groups.

### Experiment 1: Perception ablation

Compares scan-only fitness against perception-aware fitness.

Purpose: evaluate whether adding scan-level perception descriptors improves performance.

### Experiment 2: All-map benchmark

Runs the main method across all maps.

Purpose: evaluate general performance across environments of different complexity.

### Experiment 3: Beam sensitivity

Varies the number of range beams.

Purpose: evaluate how sensor resolution affects localization accuracy.

### Experiment 4: Parameter tuning

Tests controlled EA parameter variants.

Purpose: understand the impact of population size, generations, mutation scale, and tournament size.

### Experiment 5: Symmetry and ambiguity analysis

Builds scan-equivalent pose regions and compares EA outputs against them.

Purpose: separate algorithmic failure from intrinsic perceptual ambiguity. This is especially important in maps such as corridors or repeated structures.

---

## 16. Recommended navigation for a reader

A good order for reading the code is:

1. `config.py` to understand all parameters and result containers.
2. `src/pose.py` and `src/individual.py` to understand the representation.
3. `src/map_utils.py` to understand maps and valid poses.
4. `src/sensor_model.py` to understand scan simulation.
5. `src/perception.py` and `src/fitness.py` to understand evaluation.
6. `src/population.py` to understand population creation and statistics.
7. `src/evolutionary_algorithm.py` to understand the EA loop.
8. `main.py` to see one complete localization run.
9. `experiments/runner.py` to see how one standardized trial is executed.
10. `experiments/suite.py` to understand the full experimental design.
11. `experiments/ambiguity.py` to understand the symmetry and scan-equivalence analysis.
12. `case_study.py` to understand how additional explanatory figures are generated.


---

## 17. Notes for extending the project

Possible extensions include:

- adding a random-search baseline again as a separate comparison module;
- adding a true particle-filter or MCL baseline;
- introducing a motion model and multiple observations over time;
- adding real map loading from external occupancy-grid files;
- testing additional selection, crossover, and mutation operators;
- using adaptive mutation parameters;
- improving multimodal output analysis by clustering the final population;
- adding noise robustness experiments;
- adding statistical tests over repeated runs.



This README is intended to help readers navigate the codebase and understand how each file contributes to the full localization pipeline.

---

## Citation

Both the code and the report are archived on Zenodo. Each DOI below always resolves to the
latest version. To cite the work itself, prefer the report.

```bibtex
@techreport{cordella_22729482,
  author      = {Cordella, G.},
  title       = {Evolutionary Algorithm for Static Global Robot Localization in 2D Occupancy Maps},
  year        = {2026},
  institution = {Zenodo},
  doi         = {10.5281/zenodo.22729482},
  url         = {https://doi.org/10.5281/zenodo.22729482}
}

@software{cordella_22728714,
  author    = {Cordella, G.},
  title     = {Static Global Robot Localization in 2D Occupancy Maps using an Evolutionary Algorithm},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.22728714},
  url       = {https://doi.org/10.5281/zenodo.22728714}
}
```

## 18. License

Released under the [MIT License](LICENSE).

The maps in `maps/` are synthetic occupancy grids generated for this project. All code in
`src/`, `experiments/`, `main.py`, `case_study.py`, `config.py` and `experiments.py` is my own.

---

## 19. Reproducibility check

Verified on a clean checkout with a fresh virtual environment: `pip install -r requirements.txt`
resolves all pins on Python 3.13, every module in `src/` and `experiments/` imports, and
`python main.py` completes and regenerates all 15 case-study artefacts in
`results/plots/additional/` (figures, scan comparisons and the population-convergence GIF).

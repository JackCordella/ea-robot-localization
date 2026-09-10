from dataclasses import dataclass
from src.individual import Individual
from typing import List



@dataclass
class MapConfig:
    width: int = 80
    height: int = 60
    resolution: float = 0.1  # meters per cell


@dataclass
class SensorConfig:
    num_beams: int = 72
    fov_deg: float = 360.0
    max_range_m: float = 8.0
    step_m: float = 0.05
    noise_std_m: float = 0.0
    

@dataclass
class PopulationStats:
    best_fitness: float
    mean_fitness: float
    worst_fitness: float


@dataclass(frozen=True)
class FitnessBreakdown:
    scan_error: float
    perception_error: float
    fitness: float


@dataclass
class EAConfig:
    population_size: int = 100
    generations: int = 40
    tournament_size: int = 3
    crossover_rate: float = 0.9
    mutation_rate: float = 0.8
    sigma_xy: float = 0.35
    sigma_theta: float = 0.25
    elite_size: int = 3
    perception_weight: float = 0.3
    seed: int = 42


@dataclass
class EARunResult:
    best_individual: Individual
    final_population: List[Individual]
    best_fitness_history: List[float]
    mean_fitness_history: List[float]


@dataclass
class RuntimeConfig:
    map_name: str = "medium_room"
    save_plots: bool = True
    verbose: bool = True


@dataclass(frozen=True)
class LocalizationMetrics:
    position_error: float
    theta_error: float
    success: bool

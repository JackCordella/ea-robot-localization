from __future__ import annotations
from typing import List
import numpy as np

from src.individual import Individual
from src.pose import Pose
from src.map_utils import is_pose_valid
from config import PopulationStats



def initialize_population(
    population_size: int,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    theta_bounds: tuple[float, float],
    occ_grid: np.ndarray,
    resolution: float,
    rng: np.random.Generator,
    max_attempts_per_individual: int = 1000,
) -> List[Individual]:
    
    population: List[Individual] = []

    for _ in range(population_size):
        valid_pose_found = False

        for _ in range(max_attempts_per_individual):
            x = rng.uniform(*x_bounds)
            y = rng.uniform(*y_bounds)
            theta = rng.uniform(*theta_bounds)

            if is_pose_valid(x, y, occ_grid, resolution):
                population.append(Individual(pose=Pose(x=x, y=y, theta=theta)))
                valid_pose_found = True
                break

        if not valid_pose_found:
            raise RuntimeError(
                "Unable to sample a valid initial pose. "
                "Check map occupancy, bounds, and resolution."
            )

    return population


def sort_population(population: List[Individual]) -> List[Individual]:
    return sorted(population, key=lambda ind: ind.fitness, reverse=True)


def get_best_individual(population: List[Individual]) -> Individual:
    return max(population, key=lambda ind: ind.fitness)


def get_population_stats(population: List[Individual]) -> PopulationStats:
    fitness_values = [ind.fitness for ind in population]
    return PopulationStats(
        best_fitness=float(np.max(fitness_values)),
        mean_fitness=float(np.mean(fitness_values)),
        worst_fitness=float(np.min(fitness_values)),
    )


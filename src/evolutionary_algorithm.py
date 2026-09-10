from __future__ import annotations

from typing import Callable, List
import math
import numpy as np
from tqdm import tqdm

from config import EAConfig, EARunResult
from src.pose import Pose
from src.individual import Individual
from src.population import (
    get_best_individual,
    get_population_stats,
    initialize_population,
    sort_population,
)
from src.map_utils import get_world_bounds, is_pose_valid


def wrap_angle(theta: float) -> float:
    return (theta + math.pi) % (2 * math.pi) - math.pi


class EvolutionaryLocalizer:

    def __init__(
        self,
        config: EAConfig,
        occ_grid: np.ndarray,
        resolution: float,
        rng: np.random.Generator,
        individual_evaluator: Callable[[Individual], None] | None = None,
    ) -> None:
        self.config = config
        self.individual_evaluator = individual_evaluator
        if self.individual_evaluator is None:
            raise ValueError("Individual_evaluator must be provided.")

        self.rng = rng
        self.occ_grid = occ_grid
        self.resolution = resolution
        self.x_bound, self.y_bound, self.theta_bound = get_world_bounds(
            self.occ_grid, self.resolution
        )

    
    def initialize_population(self) -> List[Individual]:
        return initialize_population(
            population_size=self.config.population_size,
            x_bounds=self.x_bound,
            y_bounds=self.y_bound,
            theta_bounds=self.theta_bound,
            occ_grid=self.occ_grid,
            resolution=self.resolution,
            rng=self.rng,
        )
    
    
    def tournament_select(self, population: List[Individual]) -> Individual:
        competitors = self.rng.choice(
            population,
            size=self.config.tournament_size,
            replace=False,
        )
        winner = max(competitors, key=lambda ind: ind.fitness)
        return winner.copy()
    


    def crossover(self, parent1: Individual, parent2: Individual, max_attempts: int = 3) -> Individual:
        if self.rng.random() > self.config.crossover_rate:
            return parent1.copy()

        for _ in range(max_attempts):
            alpha = self.rng.uniform(0.0, 1.0)

            def blend_coord(a: float, b: float) -> float:
                low = min(a, b)
                high = max(a, b)
                d = high - low
                if d == 0.0:
                    return a
                return float(self.rng.uniform(low - alpha * d, high + alpha * d))

            def circular_blend_angle(theta1, theta2, alpha):
                x = alpha * np.cos(theta1) + (1.0 - alpha) * np.cos(theta2)
                y = alpha * np.sin(theta1) + (1.0 - alpha) * np.sin(theta2)
                return np.arctan2(y, x)
            
            child_x = blend_coord(parent1.pose.x, parent2.pose.x)
            child_y = blend_coord(parent1.pose.y, parent2.pose.y)
            child_theta = circular_blend_angle(parent1.pose.theta, parent2.pose.theta, alpha)

            if is_pose_valid(child_x, child_y, self.occ_grid, self.resolution):
                return Individual(
                    pose=Pose(x=child_x, y=child_y, theta=child_theta),
                    fitness=float("-inf"),
                )

        fallback_parent = parent1 if parent1.fitness >= parent2.fitness else parent2
        return fallback_parent.copy()

    
    
    def mutate(self, individual: Individual, max_attempts: int = 3) -> None:
        if self.rng.random() > self.config.mutation_rate:
            return

        original_pose = individual.pose

        for _ in range(max_attempts):
            new_x = original_pose.x + self.rng.normal(0.0, self.config.sigma_xy)
            new_y = original_pose.y + self.rng.normal(0.0, self.config.sigma_xy)
            new_theta = original_pose.theta + self.rng.normal(0.0, self.config.sigma_theta)

            new_x = float(np.clip(new_x, *self.x_bound))
            new_y = float(np.clip(new_y, *self.y_bound))
            new_theta = wrap_angle(new_theta)

            if is_pose_valid(new_x, new_y, self.occ_grid, self.resolution):
                individual.pose = Pose(x=new_x, y=new_y, theta=new_theta)
                return

        individual.pose = original_pose


    
    def make_next_generation(self, population: List[Individual]) -> List[Individual]:
        sorted_pop = sort_population(population)

        next_population: List[Individual] = [
            ind.copy() for ind in sorted_pop[: self.config.elite_size]
        ]

        while len(next_population) < self.config.population_size:
            parent1 = self.tournament_select(population)
            parent2 = self.tournament_select(population)

            child = self.crossover(parent1, parent2)
            self.mutate(child)
            self.individual_evaluator(child)

            next_population.append(child)

        return next_population

    
    
    def run(
            self,
            show_progress: bool = False,
            progress_desc: str | None = None
        ) -> EARunResult:
        
        population = self.initialize_population()
        for individual in population:
            self.individual_evaluator(individual)
        
        best_fitness_history: List[float] = []
        mean_fitness_history: List[float] = []

        progress_bar = None
        generation_iter = range(self.config.generations)

        if show_progress:
            progress_bar = tqdm(
                generation_iter,
                total=self.config.generations,
                desc=progress_desc or "EA generations",
                leave=False,
                ncols=100,
            )
            generation_iter = progress_bar

        for generation in generation_iter:
            stats = get_population_stats(population)
            best_fitness_history.append(stats.best_fitness)
            mean_fitness_history.append(stats.mean_fitness)

            best = get_best_individual(population)
            # print(
            #     f"Generation {generation:03d} | "
            #     f"Best: {best.fitness:.6f} | "
            #     f"Mean: {stats.mean_fitness:.6f}"
            # )

            population = self.make_next_generation(population)
            

        final_best = get_best_individual(population)

        return EARunResult(
            best_individual=final_best,
            final_population=population,
            best_fitness_history=best_fitness_history,
            mean_fitness_history=mean_fitness_history,
        )
from __future__ import annotations

from dataclasses import dataclass

from src.pose import Pose
from src.perception import ScanPerception

@dataclass
class Individual:
    pose: Pose
    fitness: float = float("-inf")
    perception: ScanPerception | None = None

    def copy(self) -> "Individual":
        return Individual(
            pose=Pose(self.pose.x, self.pose.y, self.pose.theta),
            fitness=self.fitness,
            perception=self.perception,
        )

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    theta: float

    def wrapped(self) -> "Pose":
        theta = ((self.theta + math.pi) % (2.0 * math.pi)) - math.pi
        return Pose(self.x, self.y, theta)

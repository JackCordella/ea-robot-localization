from __future__ import annotations
from dataclasses import dataclass

import numpy as np

@dataclass(frozen=True)
class ScanPerception:

    mean_range: float
    std_range: float
    min_range: float
    max_range: float
    roughness: float
    max_range_ratio: float

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [
                self.mean_range,
                self.std_range,
                self.min_range,
                self.max_range,
                self.roughness,
                self.max_range_ratio,
            ],
            dtype=float,
        )

def extract_scan_perception(scan: np.ndarray, sensor_max_range: float) -> ScanPerception:

    if scan.size == 0:
        raise ValueError("Cannot extract perception from an empty scan.")

    scan = np.asarray(scan, dtype=float)
    roughness = 0.0 if scan.size < 2 else float(np.mean(np.abs(np.diff(scan))))
    max_hits = np.isclose(scan, sensor_max_range, rtol=0.0, atol=1e-9)

    return ScanPerception(
        mean_range=float(np.mean(scan)),
        std_range=float(np.std(scan)),
        min_range=float(np.min(scan)),
        max_range=float(np.max(scan)),
        roughness=roughness,
        max_range_ratio=float(np.mean(max_hits)),
    )


def perception_distance(
    observed: ScanPerception,
    predicted: ScanPerception,
    sensor_max_range: float,
) -> float:

    scale = max(float(sensor_max_range), 1e-12)
    obs = observed.as_array().copy()
    pred = predicted.as_array().copy()

    # Normalize all metric features. max_range_ratio is already dimensionless.
    obs[:5] /= scale
    pred[:5] /= scale

    return float(np.mean(np.abs(obs - pred)))

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np

from experiments.configs import RESULTS_DIR
from experiments.io import save_rows_csv



def _safe_mean(values: Iterable[float]) -> float | None:
    clean = [float(v) for v in values if v is not None]
    return float(np.mean(clean)) if clean else None


def _safe_std(values: Iterable[float]) -> float | None:
    clean = [float(v) for v in values if v is not None]
    return float(np.std(clean, ddof=0)) if clean else None


def _safe_median(values: Iterable[float]) -> float | None:
    clean = [float(v) for v in values if v is not None]
    return float(np.median(clean)) if clean else None


def summarize_rows(rows: list[dict[str, Any]], group_keys: list[str]) -> list[dict[str, Any]]:
    if not rows:
        return []

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(k) for k in group_keys)
        groups.setdefault(key, []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for key, group in groups.items():
        out = {k: v for k, v in zip(group_keys, key)}

        def values(name: str) -> list[float]:
            return [float(r[name]) for r in group if r.get(name) not in (None, "")]

        out.update(
            {
                "n_runs": len(group),
                "success_rate": float(np.mean([bool(r["success"]) for r in group])),
                "mean_position_error": _safe_mean(values("position_error")),
                "std_position_error": _safe_std(values("position_error")),
                "median_position_error": _safe_median(values("position_error")),
                "mean_theta_error": _safe_mean(values("theta_error")),
                "std_theta_error": _safe_std(values("theta_error")),
                "median_theta_error": _safe_median(values("theta_error")),
                "mean_best_fitness": _safe_mean(values("best_fitness")),
                "std_best_fitness": _safe_std(values("best_fitness")),
                "mean_runtime_seconds": _safe_mean(values("runtime_seconds")),
            }
        )

        # Top-k columns exist only in experiment 5.
        for k_name in ["top1_success", "top3_success", "top5_success"]:
            if k_name in group[0]:
                out[f"{k_name}_rate"] = float(np.mean([bool(r[k_name]) for r in group]))

        if "num_distinct_modes" in group[0]:
            out["mean_num_distinct_modes"] = _safe_mean(values("num_distinct_modes"))

        summary_rows.append(out)

    return summary_rows


def write_summary_tables(rows: list[dict[str, Any]], prefix: str | Path) -> None:
    if not rows:
        return

    prefix = Path(prefix)

    candidate_groupings = [
        ["method"],
        ["map_name"],
        ["map_name", "method"],
        ["map_name", "pose_id", "method"],
        ["map_name", "num_beams", "method"],
        ["config_id"],
        ["config_id", "map_name"],
    ]

    for group_keys in candidate_groupings:
        if all(k in rows[0] for k in group_keys):
            suffix = "_summary_by_" + "_".join(group_keys) + ".csv"
            save_rows_csv(summarize_rows(rows, group_keys), f"{prefix}{suffix}")


def summarize_symmetry_rows(rows: list[dict[str, Any]], group_keys: list[str]) -> list[dict[str, Any]]:
    """Summarize the new ambiguity-aware Experiment 5 rows."""

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(k) for k in group_keys)
        groups.setdefault(key, []).append(row)

    summary: list[dict[str, Any]] = []
    for key, group in groups.items():
        out = {k: v for k, v in zip(group_keys, key)}
        n = len(group)
        exact_rate = float(np.mean([bool(r["exact_success"]) for r in group]))
        equivalent_rate = float(np.mean([bool(r["scan_equivalent_success"]) for r in group]))
        top5_equivalent_rate = float(np.mean([bool(r.get("top5_any_scan_equivalent_success")) for r in group]))
        true_failure_rate = float(np.mean([r["failure_type"] == "true_failure" for r in group]))
        ambiguous_correct_rate = float(np.mean([r["failure_type"] == "ambiguous_correct" for r in group]))

        out.update(
            {
                "n_runs": n,
                "exact_success_rate": exact_rate,
                "scan_equivalent_success_rate": equivalent_rate,
                "ambiguity_gain": equivalent_rate - exact_rate,
                "ambiguous_correct_rate": ambiguous_correct_rate,
                "true_failure_rate": true_failure_rate,
                "top5_scan_equivalent_success_rate": top5_equivalent_rate,
                "top5_ambiguity_gain": top5_equivalent_rate - exact_rate,
                "mean_output_scan_error": _safe_mean([float(r["output_scan_error"]) for r in group]),
                "median_output_scan_error": _safe_median([float(r["output_scan_error"]) for r in group]),
                "mean_position_error": _safe_mean([float(r["position_error"]) for r in group]),
                "median_position_error": _safe_median([float(r["position_error"]) for r in group]),
            }
        )
        summary.append(out)

    return summary


def write_symmetry_summary_tables(rows: list[dict[str, Any]]) -> None:


    for group_keys, name in [
        (["map_name"], "experiment_5_symmetry_summary_by_map_name.csv"),
        (["map_name", "pose_id"], "experiment_5_symmetry_summary_by_map_name_pose_id.csv"),
        (["failure_type"], "experiment_5_symmetry_summary_by_failure_type.csv"),
    ]:
        save_rows_csv(
            summarize_symmetry_rows(rows, group_keys),
            RESULTS_DIR / name,
        )

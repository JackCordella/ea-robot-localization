"""CSV input/output helpers for experiment runs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def save_rows_csv(rows: list[dict[str, Any]], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        print(f"No rows to save for {out_path}")
        return

    # Use the union of all columns because the combined nightly CSV contains
    # experiment-specific fields, for example top-k columns only in experiment 5.
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", restval="")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows -> {out_path}")


def append_rows_csv(rows: list[dict[str, Any]], out_path: str | Path) -> None:
    """Append rows to a CSV, writing the header only if the file is new."""

    if not rows:
        return

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = out_path.exists() and out_path.stat().st_size > 0

    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with out_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", restval="")
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)



from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def build_windows(
    trajectories: list[dict],
    input_sequence_length: int = 30,
    prediction_horizon: int = 10,
    target_scenarios: int | None = None,
    random_seed: int | None = None,
) -> list[dict]:
    eligible = []
    for trajectory in trajectories:
        points = trajectory.get("points", [])
        point_count = len(points) if points else int(trajectory.get("point_count", 0))
        if point_count < input_sequence_length + prediction_horizon:
            continue
        eligible.append((trajectory, point_count))

    if target_scenarios is not None and len(eligible) > target_scenarios:
        rng = np.random.default_rng(random_seed)
        selected = rng.choice(len(eligible), size=target_scenarios, replace=False)
        eligible = [eligible[index] for index in sorted(selected)]

    windows: list[dict] = []
    rng = np.random.default_rng(random_seed)
    for trajectory, point_count in eligible:
        maximum_start = point_count - input_sequence_length - prediction_horizon
        starts = (
            [int(rng.integers(0, maximum_start + 1))]
            if target_scenarios is not None
            else range(maximum_start + 1)
        )
        for idx in starts:
            windows.append(
                {
                    "scenario_id": f"{trajectory.get('trajectory_id')}_window_{idx}",
                    "trajectory_id": trajectory.get("trajectory_id"),
                    "mmsi": trajectory.get("mmsi"),
                    "window_start_index": idx,
                    "window_end_index": idx + input_sequence_length - 1,
                    "prediction_start_index": idx + input_sequence_length,
                    "prediction_end_index": idx
                    + input_sequence_length
                    + prediction_horizon
                    - 1,
                    "input_sequence_length": input_sequence_length,
                    "prediction_horizon": prediction_horizon,
                }
            )
    return windows


def split_train_validation_test(
    trajectories: list[dict],
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
    random_seed: int | None = None,
) -> Dict[str, List[dict]]:
    if not trajectories:
        return {"train": [], "validation": [], "test": []}
    rng = (
        pd.Series(range(len(trajectories)))
        if random_seed is None
        else pd.Series(range(len(trajectories))).sample(
            frac=1.0, random_state=random_seed
        )
    )
    order = list(rng.index)
    items = [trajectories[i] for i in order]
    n = len(items)
    n_validation = max(1, int(round(n * val_fraction))) if n > 1 else 0
    n_test = max(1, int(round(n * test_fraction))) if n > 1 else 0
    if n_validation + n_test >= n:
        n_validation = max(1, n // 3)
        n_test = max(1, n - n_validation - 1)
    if n_validation + n_test >= n:
        n_validation = max(0, n // 2)
        n_test = n - n_validation

    test_start = n - n_test
    validation_start = n - n_test - n_validation

    train = items[:validation_start]
    validation = items[validation_start:test_start]
    test = items[test_start:]
    return {"train": train, "validation": validation, "test": test}


def split_scenarios_by_vessel(
    scenarios: pd.DataFrame,
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
    random_seed: int | None = None,
) -> Dict[str, set]:
    """Split scenarios by MMSI so one vessel cannot leak across partitions."""
    if scenarios.empty:
        return {"train": set(), "validation": set(), "test": set()}
    if scenarios["mmsi"].isna().any():
        raise ValueError(
            "Scenarios contain missing MMSI values; cannot split by vessel."
        )
    vessels = scenarios[["mmsi"]].drop_duplicates().to_dict(orient="records")
    vessel_splits = split_train_validation_test(
        vessels,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        random_seed=random_seed,
    )
    return {
        name: {record["mmsi"] for record in records}
        for name, records in vessel_splits.items()
    }

#!/usr/bin/env python3
from __future__ import annotations

import argparse

import pandas as pd

from src.config.settings import get_config
from src.scenarios.windows import build_windows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build scenario windows from trajectory metadata."
    )
    parser.add_argument(
        "--input", required=True, help="Trajectory metadata parquet path"
    )
    parser.add_argument(
        "--points",
        default="data/trajectories/trajectory_points.parquet",
        help="Accepted for compatibility; point counts are read from metadata",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = get_config()
    metadata = pd.read_parquet(args.input)
    trajectories = metadata.to_dict(orient="records")
    windows = build_windows(
        trajectories,
        input_sequence_length=config.input_sequence_length,
        prediction_horizon=config.prediction_horizon,
        target_scenarios=config.target_scenarios,
        random_seed=config.random_seed,
    )
    scenario_df = pd.DataFrame(windows)
    out_path = config.scenarios_dir / "scenarios.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scenario_df.to_parquet(out_path, index=False)
    print(f"Wrote {len(scenario_df)} scenarios to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pandas as pd

from src.config.settings import get_config
from src.trajectories.segmenter import build_trajectory_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild AIS trajectories from validated data."
    )
    parser.add_argument("--input", required=True, help="Valid AIS parquet file")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = get_config()
    df = pd.read_parquet(args.input)
    out, points_df = build_trajectory_outputs(
        df,
        max_gap_minutes=config.max_time_gap_minutes,
        max_distance_gap_km=config.max_distance_gap_km,
        min_points=config.min_points,
        min_duration_minutes=config.min_duration_minutes,
    )
    out_path = config.trajectories_dir / "trajectory_metadata.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    points_out = config.trajectories_dir / "trajectory_points.parquet"
    points_df.to_parquet(points_out, index=False)
    print(
        f"Wrote {len(out)} trajectories and {len(points_df)} points to "
        f"{out_path} and {points_out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

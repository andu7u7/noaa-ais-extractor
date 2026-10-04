#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.preprocessing.pipeline import profile_dataframe, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile an AIS CSV file and save summary reports."
    )
    parser.add_argument("--input", required=True, help="Input CSV file")
    parser.add_argument(
        "--json-output",
        default="reports/data_profile.json",
        help="Output JSON path for the profile summary",
    )
    parser.add_argument(
        "--md-output",
        default="reports/data_profile.md",
        help="Output Markdown path for a readable profile summary",
    )
    return parser.parse_args()


def _build_markdown(profile: dict) -> str:
    rows = profile.get("rows", 0)
    cols = profile.get("columns", [])
    dtypes = profile.get("dtypes", {})
    nulls = profile.get("nulls", {})
    return (
        "\n".join(
            [
                "# AIS Data Profile",
                "",
                f"- Rows: {rows}",
                f"- Columns: {len(cols)}",
                f"- Unique MMSIs: {profile.get('mmsi_unique', 0)}",
                f"- Time min: {profile.get('time_min', 'N/A')}",
                f"- Time max: {profile.get('time_max', 'N/A')}",
                "",
                "## Columns",
                *[f"- {name}: {dtypes.get(name, 'unknown')}" for name in cols],
                "",
                "## Missing values",
                *[f"- {name}: {nulls.get(name, 0)}" for name in cols],
            ]
        )
        + "\n"
    )


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    json_output = Path(args.json_output)
    md_output = Path(args.md_output)

    df = pd.read_csv(input_path, compression="infer", low_memory=False)
    profile = profile_dataframe(df)

    write_json(json_output, profile)
    md_output.parent.mkdir(parents=True, exist_ok=True)
    md_output.write_text(_build_markdown(profile), encoding="utf-8")

    print(f"Profiled {len(df)} rows; wrote {json_output} and {md_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config.settings import get_config
from src.preprocessing.pipeline import preprocess_raw_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean and validate NOAA AIS raw data."
    )
    parser.add_argument("--input", required=True, help="Input CSV file")
    parser.add_argument("--output", required=True, help="Output parquet path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stats = preprocess_raw_csv(args.input, args.output)
    config = get_config()
    config.reports_dir.mkdir(parents=True, exist_ok=True)
    validation_report = {
        "source_file": str(Path(args.input)),
        "cleaned_file": str(Path(args.output)),
        **stats,
        "retention_fraction": (
            stats["records"] / stats["stats"]["raw_records"]
            if stats["stats"]["raw_records"]
            else 0.0
        ),
        "validation_fields": ["mmsi", "timestamp", "LAT", "LON", "SOG", "COG"],
        "deduplication_key": ["mmsi", "timestamp_utc"],
        "notes": [
            "Rows are processed in bounded-memory chunks; disk-backed deduplication is global and does not require chronological input.",
            "Records with missing/invalid MMSI, timestamp, coordinates, SOG, or COG are excluded and counted.",
            "Only point/navigation columns required downstream are retained in the cleaned Parquet.",
        ],
    }
    report_path = config.reports_dir / "preprocessing_report.json"
    temporary_path = report_path.with_name(f"{report_path.name}.tmp")
    temporary_path.write_text(
        json.dumps(validation_report, indent=2, default=str), encoding="utf-8"
    )
    temporary_path.replace(report_path)
    print(stats)
    print(f"Wrote validation report to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from src.download.noaa_downloader import NOAADownloader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download NOAA AIS daily archives.")
    parser.add_argument("--date", help="Single date in YYYY-MM-DD format")
    parser.add_argument("--start", help="Start date in YYYY-MM-DD format")
    parser.add_argument("--end", help="End date in YYYY-MM-DD format")
    parser.add_argument(
        "--dates", help="Comma-separated dates, e.g. 2025-05-01,2025-05-15"
    )
    parser.add_argument(
        "--recommended",
        action="store_true",
        help="Download a small sample for initial profiling",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    downloader = NOAADownloader()
    if args.recommended:
        dates = ["2025-05-01", "2025-05-02", "2025-05-03"]
        results = downloader.download_dates(dates)
        print(json.dumps(results, indent=2, default=str))
        return (
            0
            if any(
                result["status"] in {"downloaded", "already_downloaded"}
                for result in results
            )
            else 1
        )
    if args.date:
        results = downloader.download_dates([args.date])
        print(json.dumps(results, indent=2, default=str))
        return 0 if results[0]["status"] in {"downloaded", "already_downloaded"} else 1
    if args.start and args.end:
        results = downloader.download_date_range(args.start, args.end)
        print(json.dumps(results, indent=2, default=str))
        return (
            0
            if any(
                result["status"] in {"downloaded", "already_downloaded"}
                for result in results
            )
            else 1
        )
    if args.dates:
        dates = [d.strip() for d in args.dates.split(",") if d.strip()]
        results = downloader.download_dates(dates)
        print(json.dumps(results, indent=2, default=str))
        return (
            0
            if any(
                result["status"] in {"downloaded", "already_downloaded"}
                for result in results
            )
            else 1
        )
    print(
        "No download target supplied. Use --date, --start/--end, --dates, or --recommended"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

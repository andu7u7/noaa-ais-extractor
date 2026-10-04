from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.config.settings import get_config
from src.utils.distance import haversine_km
from src.validation.ais_rules import validate_raw_dataframe


CONFIG = get_config()


def read_noaa_csv_zst(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, compression="infer", sep=",", low_memory=False)
    return df


def profile_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {"rows": 0}
    timestamp_col = next(
        (
            name
            for name in ("timestamp", "BaseDateTime", "base_date_time")
            if name in df.columns
        ),
        None,
    )
    profile = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "dtypes": {k: str(v) for k, v in df.dtypes.items()},
        "nulls": {k: int(v) for k, v in df.isnull().sum().items()},
        "mmsi_unique": (
            int(df.get("MMSI", df.get("mmsi")).nunique())
            if "MMSI" in df.columns or "mmsi" in df.columns
            else 0
        ),
        "time_min": (
            str(pd.to_datetime(df[timestamp_col], errors="coerce", utc=True).min())
            if timestamp_col
            else None
        ),
        "time_max": (
            str(pd.to_datetime(df[timestamp_col], errors="coerce", utc=True).max())
            if timestamp_col
            else None
        ),
    }
    return profile


def create_intermediate_parquet(df: pd.DataFrame, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, path)


def preprocess_raw_csv(
    path: str | Path, output_path: str | Path, chunksize: int = 250_000
) -> dict[str, Any]:
    input_path = Path(path)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = out_path.with_name(f"{out_path.name}.tmp")
    dedupe_path = out_path.with_name(f"{out_path.name}.dedupe.sqlite")
    temp_path.unlink(missing_ok=True)
    dedupe_path.unlink(missing_ok=True)
    total_stats = {
        "raw_records": 0,
        "valid_records": 0,
        "invalid_mmsi": 0,
        "invalid_timestamp": 0,
        "invalid_lat_lon": 0,
        "invalid_sog": 0,
        "invalid_cog": 0,
        "duplicates": 0,
    }
    writer = None
    dedupe_db = sqlite3.connect(dedupe_path)
    dedupe_db.execute("PRAGMA journal_mode=OFF")
    dedupe_db.execute("PRAGMA synchronous=OFF")
    dedupe_db.execute("PRAGMA cache_size=-64000")
    dedupe_db.execute(
        "CREATE TABLE seen_keys (mmsi TEXT NOT NULL, timestamp TEXT NOT NULL, "
        "PRIMARY KEY (mmsi, timestamp)) WITHOUT ROWID"
    )

    def process_chunk(chunk: pd.DataFrame) -> None:
        nonlocal writer
        total_stats["raw_records"] += len(chunk)
        mmsi_column = next(
            (name for name in ("mmsi", "MMSI") if name in chunk.columns), None
        )
        time_column = next(
            (
                name
                for name in ("timestamp", "BaseDateTime", "base_date_time")
                if name in chunk.columns
            ),
            None,
        )
        if mmsi_column is None or time_column is None:
            raise ValueError("Input CSV requires MMSI and timestamp fields.")
        timestamps = pd.to_datetime(chunk[time_column], errors="coerce", utc=True)
        mmsi_keys = chunk[mmsi_column].astype("string").fillna("<missing-mmsi>")
        timestamp_keys = timestamps.astype("string").fillna("<invalid-time>")
        key_pairs = list(zip(mmsi_keys.tolist(), timestamp_keys.tolist(), strict=True))
        keep_rows = []
        batch_size = 400
        for offset in range(0, len(key_pairs), batch_size):
            batch = key_pairs[offset : offset + batch_size]
            values_sql = ",".join("(?, ?)" for _ in batch)
            parameters = [part for pair in batch for part in pair]
            inserted = dedupe_db.execute(
                "INSERT OR IGNORE INTO seen_keys (mmsi, timestamp) VALUES "
                f"{values_sql} RETURNING mmsi, timestamp",
                parameters,
            ).fetchall()
            newly_seen = set(inserted)
            batch_seen = set()
            for index, pair in enumerate(batch):
                if pair in newly_seen and pair not in batch_seen:
                    keep_rows.append(offset + index)
                    batch_seen.add(pair)
        dedupe_db.commit()
        total_stats["duplicates"] += len(chunk) - len(keep_rows)
        unique_chunk = chunk.iloc[keep_rows]
        cleaned, stats = validate_raw_dataframe(unique_chunk)
        for name in total_stats:
            if name not in {"raw_records", "duplicates"}:
                total_stats[name] += stats[name]
        if cleaned.empty:
            return
        # Keep the stable point schema needed downstream, not heterogeneous
        # vessel metadata whose inferred types can vary between chunks.
        cleaned = cleaned[["mmsi", "timestamp", "LAT", "LON", "SOG", "COG"]]
        cleaned = cleaned.sort_values(["mmsi", "timestamp"])
        table = pa.Table.from_pandas(cleaned, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(temp_path, table.schema)
        writer.write_table(table)

    try:
        chunks = pd.read_csv(
            input_path,
            compression="infer",
            sep=",",
            low_memory=False,
            chunksize=chunksize,
        )
        for chunk in chunks:
            process_chunk(chunk)
    except Exception:
        if writer is not None:
            writer.close()
        dedupe_db.close()
        temp_path.unlink(missing_ok=True)
        dedupe_path.unlink(missing_ok=True)
        raise
    finally:
        if writer is not None and writer.is_open:
            writer.close()
        if dedupe_db:
            dedupe_db.close()
        dedupe_path.unlink(missing_ok=True)

    if total_stats["valid_records"] == 0:
        temp_path.unlink(missing_ok=True)
        dedupe_path.unlink(missing_ok=True)
        raise ValueError(
            "No valid AIS rows were found. Check the source schema and validation rules."
        )
    temp_path.replace(out_path)
    return {"records": total_stats["valid_records"], "stats": total_stats}


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)


def build_dataset_summary(
    raw_counts: dict, final_counts: dict, files: list[str], date_list: list[str]
) -> dict:
    return {
        "source": "NOAA Office for Coastal Management",
        "dataset": "Nationwide Automatic Identification System 2025",
        "year": 2025,
        "downloaded_dates": date_list,
        "files": files,
        "raw_records": raw_counts.get("raw_records", 0),
        "valid_records": raw_counts.get("valid_records", 0),
        "trajctories": final_counts.get("trajectories", 0),
        "scenarios": final_counts.get("scenarios", 0),
        "parameters": {
            "MIN_POINTS": CONFIG.min_points,
            "MIN_DURATION_MINUTES": CONFIG.min_duration_minutes,
            "MAX_TIME_GAP_MINUTES": CONFIG.max_time_gap_minutes,
            "MAX_DISTANCE_GAP_KM": CONFIG.max_distance_gap_km,
            "INPUT_SEQUENCE_LENGTH": CONFIG.input_sequence_length,
            "PREDICTION_HORIZON": CONFIG.prediction_horizon,
            "TARGET_SCENARIOS": CONFIG.target_scenarios,
        },
    }

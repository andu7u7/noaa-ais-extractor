from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Iterable, Optional

import pandas as pd


def validate_mmsi(value) -> Optional[int]:
    if value is None or pd.isna(value):
        return None
    try:
        text = str(value).strip()
    except Exception:
        return None
    if not text or not text.isdigit():
        return None
    value_int = int(text)
    if len(text) != 9 or value_int == 0:
        return None
    return value_int


def validate_timestamp(value) -> Optional[pd.Timestamp]:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        ts = value
    else:
        ts = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(ts):
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def validate_lat_lon(lat, lon) -> Optional[tuple[float, float]]:
    if lat is None or lon is None or pd.isna(lat) or pd.isna(lon):
        return None
    try:
        lat_val = float(lat)
        lon_val = float(lon)
    except (TypeError, ValueError):
        return None
    if not (-90.0 <= lat_val <= 90.0 and -180.0 <= lon_val <= 180.0):
        return None
    return lat_val, lon_val


def validate_sog(value) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(val):
        return None
    sentinel_values = {102.3, 102.4, 102.5, 255.0, 511.0}
    if val in sentinel_values:
        return None
    if -0.1 <= val <= 102.2:
        return val
    return None


def validate_cog(value) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(val):
        return None
    if val in {360.0, 511.0, 360, 511}:
        return None
    if 0.0 <= val <= 360.0:
        return val
    return None


def validate_heading(value) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(val):
        return None
    sentinel_values = {511.0, 360.0}
    if val in sentinel_values:
        return None
    if 0.0 <= val <= 359.0:
        return val
    return None


def deduplicate_records(
    df: pd.DataFrame, key_cols: Iterable[str] = ("mmsi", "BaseDateTime")
) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    cols = list(key_cols)
    out = df.copy()
    out = out.drop_duplicates(subset=cols, keep="first").copy()
    return out


def validate_raw_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    stats = {
        "raw_records": int(len(df)),
        "valid_records": 0,
        "invalid_mmsi": 0,
        "invalid_timestamp": 0,
        "invalid_lat_lon": 0,
        "invalid_sog": 0,
        "invalid_cog": 0,
        "duplicates": 0,
    }
    if df.empty:
        return df.copy(), stats

    df = df.copy()
    aliases = {
        "mmsi": ("mmsi", "MMSI"),
        "timestamp": ("timestamp", "BaseDateTime", "base_date_time"),
        "LAT": ("LAT", "lat", "latitude"),
        "LON": ("LON", "lon", "longitude"),
        "SOG": ("SOG", "sog"),
        "COG": ("COG", "cog"),
    }
    for canonical, candidates in aliases.items():
        source = next((name for name in candidates if name in df.columns), None)
        if source is None:
            raise ValueError(
                f"Required AIS field '{canonical}' not found. "
                f"Accepted names: {', '.join(candidates)}. "
                f"Available columns: {', '.join(map(str, df.columns))}"
            )
        if source != canonical:
            df[canonical] = df[source]

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    duplicates_before = len(df)
    df = deduplicate_records(df, key_cols=("mmsi", "timestamp"))
    stats["duplicates"] = duplicates_before - len(df)

    mmsi_text = df["mmsi"].astype("string").str.strip()
    mmsi_valid = mmsi_text.str.fullmatch(r"\d{9}", na=False)
    mmsi_numeric = pd.to_numeric(mmsi_text, errors="coerce")
    mmsi_valid &= mmsi_numeric.gt(0)

    timestamps = df["timestamp"]
    timestamp_valid = timestamps.notna()

    latitude = pd.to_numeric(df["LAT"], errors="coerce")
    longitude = pd.to_numeric(df["LON"], errors="coerce")
    coordinates_valid = latitude.between(-90.0, 90.0) & longitude.between(-180.0, 180.0)

    sog = pd.to_numeric(df["SOG"], errors="coerce")
    sog_valid = sog.between(-0.1, 102.2) & ~sog.isin(
        [102.3, 102.4, 102.5, 255.0, 511.0]
    )
    cog = pd.to_numeric(df["COG"], errors="coerce")
    cog_valid = cog.between(0.0, 360.0) & ~cog.isin([360.0, 511.0])

    stats["invalid_mmsi"] = int((~mmsi_valid).sum())
    remaining = mmsi_valid
    stats["invalid_timestamp"] = int((remaining & ~timestamp_valid).sum())
    remaining &= timestamp_valid
    stats["invalid_lat_lon"] = int((remaining & ~coordinates_valid).sum())
    remaining &= coordinates_valid
    stats["invalid_sog"] = int((remaining & ~sog_valid).sum())
    remaining &= sog_valid
    stats["invalid_cog"] = int((remaining & ~cog_valid).sum())
    valid = remaining & cog_valid

    filtered = df.loc[valid].copy()
    filtered["mmsi"] = mmsi_numeric.loc[valid].astype("int64")
    filtered["timestamp"] = timestamps.loc[valid]
    filtered["LAT"] = latitude.loc[valid].astype("float64")
    filtered["LON"] = longitude.loc[valid].astype("float64")
    filtered["SOG"] = sog.loc[valid].astype("float64")
    filtered["COG"] = cog.loc[valid].astype("float64")
    stats["valid_records"] = int(len(filtered))
    return filtered, stats

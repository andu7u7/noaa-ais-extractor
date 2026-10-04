from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from src.utils.distance import haversine_km


def build_trajectory_outputs(
    df: pd.DataFrame,
    max_gap_minutes: int = 30,
    max_distance_gap_km: float = 15.0,
    min_points: int = 20,
    min_duration_minutes: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Vectorized trajectory segmentation for large AIS tables."""
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    points = df.sort_values(["mmsi", "timestamp"], kind="mergesort").reset_index(
        drop=True
    )
    by_vessel = points.groupby("mmsi", sort=False)
    gaps = by_vessel["timestamp"].diff().dt.total_seconds().div(60.0)
    lat1 = points["LAT"].shift().to_numpy(dtype="float64")
    lon1 = points["LON"].shift().to_numpy(dtype="float64")
    lat2 = points["LAT"].to_numpy(dtype="float64")
    lon2 = points["LON"].to_numpy(dtype="float64")
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    haversine_a = (
        np.sin(delta_phi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    )
    haversine_a = np.clip(haversine_a, 0.0, 1.0)
    step_distance = (
        2.0 * 6371.0 * np.arctan2(np.sqrt(haversine_a), np.sqrt(1.0 - haversine_a))
    )
    same_vessel = points["mmsi"].eq(points["mmsi"].shift()).to_numpy()
    step_distance = np.where(same_vessel, step_distance, 0.0)
    starts = (
        by_vessel.cumcount().eq(0)
        | gaps.gt(max_gap_minutes)
        | pd.Series(same_vessel & (step_distance > max_distance_gap_km))
    )
    points["_segment_id"] = starts.astype("int64").cumsum()
    same_segment = ~starts.to_numpy()
    points["_step_distance_km"] = np.where(same_segment, step_distance, 0.0)

    grouped = points.groupby("_segment_id", sort=False)
    metadata = grouped.agg(
        mmsi=("mmsi", "first"),
        start_time=("timestamp", "first"),
        end_time=("timestamp", "last"),
        point_count=("timestamp", "size"),
        start_lat=("LAT", "first"),
        start_lon=("LON", "first"),
        end_lat=("LAT", "last"),
        end_lon=("LON", "last"),
        distance_km=("_step_distance_km", "sum"),
        mean_sog=("SOG", "mean"),
        max_sog=("SOG", "max"),
        mean_cog=("COG", "mean"),
    )
    metadata["duration_minutes"] = (
        metadata["end_time"] - metadata["start_time"]
    ).dt.total_seconds() / 60.0
    eligible = metadata.loc[
        metadata["point_count"].ge(min_points)
        & metadata["duration_minutes"].ge(min_duration_minutes)
    ].copy()
    eligible.index.name = "_segment_id"
    eligible = eligible.reset_index()
    eligible["trajectory_id"] = "trajectory_" + eligible["_segment_id"].astype(str)
    eligible_ids = eligible["_segment_id"]
    points = points.loc[points["_segment_id"].isin(eligible_ids)].copy()
    points["trajectory_id"] = "trajectory_" + points["_segment_id"].astype(str)
    points = points.drop(columns=["_segment_id", "_step_distance_km"])
    eligible = eligible.drop(columns=["_segment_id"])
    return eligible.reset_index(drop=True), points.reset_index(drop=True)


def detect_gaps(
    df: pd.DataFrame, max_gap_minutes: int = 30, time_col: str = "timestamp"
) -> list[dict]:
    if df.empty:
        return []
    if len(df) == 1 and time_col in df.columns:
        value = df.iloc[0][time_col]
        if pd.api.types.is_list_like(value) and not isinstance(value, pd.Timestamp):
            ordered = pd.to_datetime(list(value))
        else:
            ordered = (
                pd.to_datetime(df[time_col].tolist())
                if time_col in df.columns
                else pd.to_datetime(df.iloc[:, 0].tolist())
            )
    else:
        ordered = (
            pd.to_datetime(df[time_col].tolist())
            if time_col in df.columns
            else pd.to_datetime(df.iloc[:, 0].tolist())
        )
    ordered = pd.Series(ordered).sort_values().reset_index(drop=True)
    gaps = []
    for prev, curr in zip(ordered.iloc[:-1], ordered.iloc[1:]):
        diff = (curr - prev).total_seconds() / 60.0
        if diff > max_gap_minutes:
            gaps.append(
                {
                    "start": prev,
                    "end": curr,
                    "gap_minutes": float(diff),
                }
            )
    return gaps


def segment_trajectories(
    df: pd.DataFrame, max_gap_minutes: int = 30, time_col: str = "timestamp"
) -> list[pd.DataFrame]:
    if df.empty:
        return []
    if len(df) == 1 and time_col in df.columns:
        value = df.iloc[0][time_col]
        if pd.api.types.is_list_like(value) and not isinstance(value, pd.Timestamp):
            rows = [
                {"mmsi": df.iloc[0].get("mmsi", 0), "timestamp": ts}
                for ts in pd.to_datetime(list(value))
            ]
            df = pd.DataFrame(rows)
    segments: list[pd.DataFrame] = []
    for mmsi, group in df.sort_values(time_col).groupby("mmsi", sort=True):
        current = []
        for _, row in group.sort_values(time_col).iterrows():
            if not current:
                current = [row.to_dict()]
                continue
            prev = pd.Timestamp(current[-1]["timestamp"])
            curr = pd.Timestamp(row["timestamp"])
            gap_minutes = (curr - prev).total_seconds() / 60.0
            if gap_minutes > max_gap_minutes:
                segments.append(pd.DataFrame(current))
                current = [row.to_dict()]
            else:
                current.append(row.to_dict())
        if current:
            segments.append(pd.DataFrame(current))
    return segments


def build_trajectory_metadata(segments: list[pd.DataFrame]) -> list[dict]:
    metadata = []
    for idx, segment in enumerate(segments):
        if segment.empty:
            continue
        segment = segment.sort_values("timestamp").copy()
        first = segment.iloc[0]
        last = segment.iloc[-1]
        distance_km = sum(
            haversine_km(
                float(prev.LAT),
                float(prev.LON),
                float(curr.LAT),
                float(curr.LON),
            )
            for prev, curr in zip(
                segment.iloc[:-1].itertuples(index=False),
                segment.iloc[1:].itertuples(index=False),
            )
        )
        metadata.append(
            {
                "trajectory_id": f"trajectory_{idx + 1}",
                "mmsi": int(first["mmsi"]),
                "start_time": first["timestamp"],
                "end_time": last["timestamp"],
                "duration_minutes": float(
                    (last["timestamp"] - first["timestamp"]).total_seconds() / 60.0
                ),
                "point_count": int(len(segment)),
                "start_lat": float(first["LAT"]),
                "start_lon": float(first["LON"]),
                "end_lat": float(last["LAT"]),
                "end_lon": float(last["LON"]),
                "distance_km": float(distance_km),
                "mean_sog": (
                    float(segment["SOG"].mean()) if "SOG" in segment.columns else 0.0
                ),
                "max_sog": (
                    float(segment["SOG"].max()) if "SOG" in segment.columns else 0.0
                ),
                "mean_cog": (
                    float(segment["COG"].mean()) if "COG" in segment.columns else 0.0
                ),
            }
        )
    return metadata

#!/usr/bin/env python3
"""Build, train and evaluate GRU/LSTM against AIS data; export one SPSS CSV.

No synthetic AIS values or forced-positive improvements are used. A negative
Pre-minus-Post difference means the model did worse than persistence.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config.settings import get_config
from src.preprocessing.pipeline import preprocess_raw_csv
from src.scenarios.windows import split_train_validation_test
from src.trajectories.segmenter import build_trajectory_outputs


INPUT_STEPS = 30
TARGET_STEPS = 10
EARTH_RADIUS_KM = 6371.0
OUTPUT_COLUMNS = [
    "ID_Trayectoria",
    "MMSI",
    "Pre_ADE",
    "Post_GRU_ADE",
    "Post_LSTM_ADE",
    "Pre_FDE",
    "Post_GRU_FDE",
    "Post_LSTM_FDE",
    "Dif_ADE_GRU",
    "Dif_ADE_LSTM",
    "Dif_FDE_GRU",
    "Dif_FDE_LSTM",
]


class SeqRegressor(nn.Module):
    def __init__(self, cell: str, input_size: int = 6, hidden_size: int = 16):
        super().__init__()
        recurrent = nn.LSTM if cell == "LSTM" else nn.GRU
        self.encoder = recurrent(input_size, hidden_size, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, TARGET_STEPS * 2),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(values)
        hidden = (
            encoded[1][0][-1] if isinstance(self.encoder, nn.LSTM) else encoded[1][-1]
        )
        return self.head(hidden).reshape(-1, TARGET_STEPS, 2)


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1 = np.radians(np.asarray(lat1, dtype=np.float64))
    lon1 = np.radians(np.asarray(lon1, dtype=np.float64))
    lat2 = np.radians(np.asarray(lat2, dtype=np.float64))
    lon2 = np.radians(np.asarray(lon2, dtype=np.float64))
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1
    a = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2) ** 2
    )
    a = np.clip(a, 0, 1)
    return 2 * EARTH_RADIUS_KM * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def local_offsets(
    lat: np.ndarray, lon: np.ndarray, origin_lat: float, origin_lon: float
) -> np.ndarray:
    north = np.radians(lat - origin_lat) * EARTH_RADIUS_KM
    east = (
        np.radians(lon - origin_lon) * EARTH_RADIUS_KM * np.cos(np.radians(origin_lat))
    )
    return np.column_stack([east, north])


def offsets_to_coordinates(
    offsets: np.ndarray, origin_lat: float, origin_lon: float
) -> tuple[np.ndarray, np.ndarray]:
    lat = origin_lat + np.degrees(offsets[:, 1] / EARTH_RADIUS_KM)
    cos_lat = max(abs(np.cos(np.radians(origin_lat))), 1e-8)
    lon = origin_lon + np.degrees(offsets[:, 0] / (EARTH_RADIUS_KM * cos_lat))
    return lat, lon


def scenario_features(
    points: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    points = points.sort_values("timestamp").reset_index(drop=True)
    if len(points) != INPUT_STEPS + TARGET_STEPS:
        raise ValueError(f"Expected 40 AIS points per scenario; got {len(points)}")
    lat = points["LAT"].to_numpy(dtype=np.float64)
    lon = points["LON"].to_numpy(dtype=np.float64)
    sog = points["SOG"].to_numpy(dtype=np.float64)
    cog = np.radians(points["COG"].to_numpy(dtype=np.float64))
    seconds = (
        pd.to_datetime(points["timestamp"], utc=True).astype("int64").to_numpy() / 1e9
    )
    intervals = np.diff(seconds[:INPUT_STEPS], prepend=seconds[0])
    input_offsets = local_offsets(
        lat[:INPUT_STEPS], lon[:INPUT_STEPS], lat[INPUT_STEPS - 1], lon[INPUT_STEPS - 1]
    )
    x = np.column_stack(
        [
            input_offsets[:, 0],
            input_offsets[:, 1],
            sog[:INPUT_STEPS],
            np.sin(cog[:INPUT_STEPS]),
            np.cos(cog[:INPUT_STEPS]),
            np.log1p(intervals),
        ]
    )
    y = local_offsets(
        lat[INPUT_STEPS:], lon[INPUT_STEPS:], lat[INPUT_STEPS - 1], lon[INPUT_STEPS - 1]
    )
    return x, y, lat, lon


def make_scenarios(
    metadata: pd.DataFrame, points: pd.DataFrame, config, seed: int
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    candidates = metadata.loc[
        metadata["point_count"].ge(INPUT_STEPS + TARGET_STEPS)
    ].copy()
    if candidates.empty:
        raise ValueError(
            "No eligible trajectories have enough AIS points for a 30+10 window."
        )
    # Select exactly one actual trajectory per MMSI to keep the independent unit unambiguous.
    candidates = candidates.sample(frac=1, random_state=seed).drop_duplicates("mmsi")
    if len(candidates) > config.target_scenarios:
        candidates = candidates.sample(n=config.target_scenarios, random_state=seed)
    candidates = candidates.sort_values("trajectory_id").reset_index(drop=True)
    if len(candidates) < 253:
        raise ValueError(
            f"Only {len(candidates)} eligible vessels. At least 253 unique vessels are "
            "needed for >50 in a 20% test split. No synthetic rows will be created."
        )

    points_by_trajectory = {
        trajectory_id: group.sort_values("timestamp").reset_index(drop=True)
        for trajectory_id, group in points.groupby("trajectory_id", sort=False)
    }
    records = []
    features, targets = [], []
    for _, trajectory in candidates.iterrows():
        trajectory_id = trajectory["trajectory_id"]
        vessel_points = points_by_trajectory.get(trajectory_id)
        if vessel_points is None:
            continue
        count = len(vessel_points)
        start_max = count - INPUT_STEPS - TARGET_STEPS
        start = int(rng.integers(start_max + 1))
        selected = vessel_points.iloc[start : start + INPUT_STEPS + TARGET_STEPS]
        x, y, lat, lon = scenario_features(selected)
        features.append(x)
        targets.append(y)
        records.append(
            {
                "ID_Trayectoria": str(trajectory_id),
                "MMSI": int(trajectory["mmsi"]),
                "scenario_id": f"{trajectory_id}_spss_{start}",
                "input_end_lat": float(lat[INPUT_STEPS - 1]),
                "input_end_lon": float(lon[INPUT_STEPS - 1]),
                "target_lat": lat[INPUT_STEPS:].copy(),
                "target_lon": lon[INPUT_STEPS:].copy(),
            }
        )
    if len(records) < 253:
        raise ValueError(
            f"Only {len(records)} usable vessels after joining trajectory points; refusing synthetic expansion."
        )
    scenario_table = pd.DataFrame(records)
    split = split_train_validation_test(
        scenario_table[["MMSI"]].to_dict(orient="records"),
        val_fraction=0.2,
        test_fraction=0.2,
        random_seed=seed,
    )
    split_by_mmsi = {
        name: {int(row["MMSI"]) for row in rows} for name, rows in split.items()
    }
    scenario_table["Particion"] = scenario_table["MMSI"].map(
        {mmsi: name for name, vessels in split_by_mmsi.items() for mmsi in vessels}
    )
    if scenario_table.loc[scenario_table.Particion.eq("test"), "MMSI"].nunique() <= 50:
        raise ValueError("Generated test split does not exceed 50 independent MMSI.")
    return (
        np.stack(features).astype("float32"),
        np.stack(targets).astype("float32"),
        scenario_table,
        np.array([*split_by_mmsi["train"]]),
        np.array([*split_by_mmsi["validation"]]),
    )


def train_one_model(
    cell: str,
    x: np.ndarray,
    y: np.ndarray,
    frame: pd.DataFrame,
    config,
    args,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    train_mask = frame.Particion.eq("train").to_numpy()
    validation_mask = frame.Particion.eq("validation").to_numpy()
    test_mask = frame.Particion.eq("test").to_numpy()
    from sklearn.preprocessing import StandardScaler

    input_scaler = StandardScaler().fit(x[train_mask].reshape(-1, x.shape[-1]))
    x_scaled = (
        input_scaler.transform(x.reshape(-1, x.shape[-1]))
        .reshape(x.shape)
        .astype("float32")
    )
    target_mean = y[train_mask].mean(axis=(0, 1), keepdims=True)
    target_scale = y[train_mask].std(axis=(0, 1), keepdims=True)
    target_scale = np.where(target_scale < 1e-6, 1.0, target_scale)
    y_scaled = ((y - target_mean) / target_scale).astype("float32")

    dataset = TensorDataset(
        torch.from_numpy(x_scaled[train_mask]), torch.from_numpy(y_scaled[train_mask])
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, generator=generator
    )
    model = SeqRegressor(cell, hidden_size=args.hidden_size).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    loss_fn = nn.SmoothL1Loss()
    best_state = None
    best_validation = float("inf")
    best_epoch = 0
    patience_count = 0
    val_x = torch.from_numpy(x_scaled[validation_mask]).to(device)
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_prediction = model(val_x).cpu().numpy() * target_scale + target_mean
        val_errors = []
        val_frame = frame.loc[validation_mask].reset_index(drop=True)
        for prediction, (_, scenario) in zip(
            val_prediction, val_frame.iterrows(), strict=True
        ):
            pred_lat, pred_lon = offsets_to_coordinates(
                prediction, scenario.input_end_lat, scenario.input_end_lon
            )
            errors = haversine_km(
                pred_lat, pred_lon, scenario.target_lat, scenario.target_lon
            )
            val_errors.append(errors.mean())
        validation_ade = float(np.mean(val_errors))
        if validation_ade < best_validation - 1e-6:
            best_validation, best_epoch = validation_ade, epoch
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= args.patience:
                break
    if best_state is None:
        raise RuntimeError(f"{cell}: training produced no selected checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    if args.validation_only:
        return np.empty((0, TARGET_STEPS, 2), dtype=np.float32), best_validation
    test_x = torch.from_numpy(x_scaled[test_mask]).to(device)
    with torch.no_grad():
        prediction = model(test_x).cpu().numpy() * target_scale + target_mean
    return prediction, best_validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train LSTM and GRU on real AIS and export one SPSS CSV."
    )
    parser.add_argument(
        "--raw",
        default="data/raw/2025-05-01.csv.zst",
        help="Original NOAA .csv/.csv.zst",
    )
    parser.add_argument(
        "--reuse-derived",
        action="store_true",
        help="Reuse existing cleaned/trajectory Parquet files instead of rebuilding them",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/final/ais_procesado_spss.csv")
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="Tune epochs/hyperparameters using train+validation only; do not score or write test results",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1:
        raise ValueError("epochs, patience, and batch size must be positive.")
    config = get_config()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cleaned_path = config.intermediate_dir / "cleaned.parquet"
    metadata_path = config.trajectories_dir / "trajectory_metadata.parquet"
    points_path = config.trajectories_dir / "trajectory_points.parquet"
    if args.reuse_derived:
        missing = [
            str(path)
            for path in (cleaned_path, metadata_path, points_path)
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(f"Cannot reuse missing derived files: {missing}")
    else:
        preprocess_raw_csv(args.raw, cleaned_path)
        cleaned = pd.read_parquet(cleaned_path)
        metadata, points = build_trajectory_outputs(
            cleaned,
            max_gap_minutes=config.max_time_gap_minutes,
            max_distance_gap_km=config.max_distance_gap_km,
            min_points=config.min_points,
            min_duration_minutes=config.min_duration_minutes,
        )
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata.to_parquet(metadata_path, index=False)
        points.to_parquet(points_path, index=False)
    metadata = pd.read_parquet(metadata_path)
    points = pd.read_parquet(points_path)
    x, y, frame, _, _ = make_scenarios(metadata, points, config, args.seed)
    predictions: dict[str, np.ndarray] = {}
    best_validation: dict[str, float] = {}
    for cell in ("GRU", "LSTM"):
        predictions[cell], best_validation[cell] = train_one_model(
            cell, x, y, frame, config, args, device
        )

    if args.validation_only:
        print("Validation-only tuning run; test partition was not scored or exported.")
        print("Best validation ADE (km):", json.dumps(best_validation, sort_keys=True))
        return 0

    test_mask = frame.Particion.eq("test").to_numpy()
    test_frame = frame.loc[test_mask].reset_index(drop=True)
    rows = []
    for index, (_, scenario) in enumerate(test_frame.iterrows()):
        row = {
            "ID_Trayectoria": scenario.ID_Trayectoria,
            "MMSI": scenario.MMSI,
        }
        actual_lat = scenario.target_lat
        actual_lon = scenario.target_lon
        origin_lat, origin_lon = scenario.input_end_lat, scenario.input_end_lon
        baseline_lat = np.full(TARGET_STEPS, origin_lat)
        baseline_lon = np.full(TARGET_STEPS, origin_lon)
        pre_errors = haversine_km(baseline_lat, baseline_lon, actual_lat, actual_lon)
        row["Pre_ADE"] = float(pre_errors.mean() * 1000.0)
        row["Pre_FDE"] = float(pre_errors[-1] * 1000.0)
        for cell in ("GRU", "LSTM"):
            pred_lat, pred_lon = offsets_to_coordinates(
                predictions[cell][index], origin_lat, origin_lon
            )
            post_errors = haversine_km(pred_lat, pred_lon, actual_lat, actual_lon)
            row[f"Post_{cell}_ADE"] = float(post_errors.mean() * 1000.0)
            row[f"Post_{cell}_FDE"] = float(post_errors[-1] * 1000.0)
        row["Dif_ADE_GRU"] = row["Pre_ADE"] - row["Post_GRU_ADE"]
        row["Dif_ADE_LSTM"] = row["Pre_ADE"] - row["Post_LSTM_ADE"]
        row["Dif_FDE_GRU"] = row["Pre_FDE"] - row["Post_GRU_FDE"]
        row["Dif_FDE_LSTM"] = row["Pre_FDE"] - row["Post_LSTM_FDE"]
        rows.append(row)

    output = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    if output["MMSI"].nunique() <= 50:
        raise ValueError(
            "Evaluation requires more than 50 real independent test MMSIs; no synthetic records were added."
        )
    numeric = output.select_dtypes(include=["number"])
    if (
        output.isna().any().any()
        or not np.isfinite(numeric.to_numpy(dtype=np.float64)).all()
    ):
        raise ValueError(
            "Output contains missing/non-finite data; refusing to export a misleading CSV."
        )
    if not output["MMSI"].is_unique:
        raise ValueError("Expected one aggregated test trajectory per MMSI.")
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.tmp")
    output.to_csv(temporary, index=False, sep=",", decimal=".", float_format="%.6f")
    temporary.replace(target)
    print(f"Exported {len(output)} real test vessels to {target}")
    print(f"Pre-ADE mean: {output.Pre_ADE.mean():.2f} m")
    print(f"GRU Post-ADE mean: {output.Post_GRU_ADE.mean():.2f} m")
    print(f"LSTM Post-ADE mean: {output.Post_LSTM_ADE.mean():.2f} m")
    print(
        "Positive Dif values indicate error reduction; negative values are retained honestly."
    )
    print("Best validation ADE (km):", json.dumps(best_validation, sort_keys=True))
    for model in ("GRU", "LSTM"):
        for metric in ("ADE", "FDE"):
            difference = output[f"Dif_{metric}_{model}"].to_numpy(dtype=np.float64)
            shapiro = stats.shapiro(difference) if len(difference) >= 3 else None
            wilcoxon = (
                stats.wilcoxon(difference, alternative="two-sided", method="auto")
                if np.any(difference != 0)
                else None
            )
            print(
                f"{model} {metric}: improved={(difference > 0).sum()}, "
                f"worsened={(difference < 0).sum()}, median Dif={np.median(difference):.2f} m, "
                f"Shapiro p={shapiro.pvalue:.3g}, "
                f"Wilcoxon p={wilcoxon.pvalue:.3g}"
                if wilcoxon is not None and shapiro is not None
                else f"{model} {metric}: insufficient paired observations for inference"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

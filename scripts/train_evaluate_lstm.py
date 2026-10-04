#!/usr/bin/env python3
"""Train a compact LSTM and evaluate it once on the held-out AIS test split."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from scripts.build_dataset import _haversine_distances_km
from src.config.settings import get_config


INPUT_STEPS = 30
TARGET_STEPS = 10
EARTH_RADIUS_KM = 6371.0


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary_path = path.with_name(f"{path.name}.tmp")
    frame.to_csv(temporary_path, index=False)
    temporary_path.replace(path)


class AISLSTM(nn.Module):
    def __init__(self, input_size: int = 6, hidden_size: int = 16) -> None:
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, batch_first=True
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, TARGET_STEPS * 2),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.encoder(values)
        return self.head(hidden[-1]).reshape(-1, TARGET_STEPS, 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a small LSTM on train/validation and report held-out test accuracy."
    )
    parser.add_argument(
        "--data",
        default="data/final/spss_geolocation_study.csv",
        help="Consolidated scenario CSV",
    )
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    return parser.parse_args()


def _sequence_matrix(row: pd.Series) -> np.ndarray:
    lat = np.array([row[f"input_{step:02d}_lat"] for step in range(1, INPUT_STEPS + 1)])
    lon = np.array([row[f"input_{step:02d}_lon"] for step in range(1, INPUT_STEPS + 1)])
    sog = np.array([row[f"input_{step:02d}_sog"] for step in range(1, INPUT_STEPS + 1)])
    cog = np.radians(
        np.array([row[f"input_{step:02d}_cog"] for step in range(1, INPUT_STEPS + 1)])
    )
    elapsed = np.array(
        [row[f"input_{step:02d}_elapsed_sec"] for step in range(1, INPUT_STEPS + 1)]
    )
    latitude_origin = lat[-1]
    north_km = np.radians(lat - latitude_origin) * EARTH_RADIUS_KM
    east_km = (
        np.radians(lon - lon[-1])
        * EARTH_RADIUS_KM
        * np.cos(np.radians(latitude_origin))
    )
    intervals = np.diff(elapsed, prepend=elapsed[0])
    return np.column_stack(
        [east_km, north_km, sog, np.sin(cog), np.cos(cog), np.log1p(intervals)]
    )


def _target_offsets(row: pd.Series) -> np.ndarray:
    origin_lat = float(row["input_30_lat"])
    origin_lon = float(row["input_30_lon"])
    target_lat = np.array(
        [row[f"target_{step:02d}_lat"] for step in range(1, TARGET_STEPS + 1)]
    )
    target_lon = np.array(
        [row[f"target_{step:02d}_lon"] for step in range(1, TARGET_STEPS + 1)]
    )
    north_km = np.radians(target_lat - origin_lat) * EARTH_RADIUS_KM
    east_km = (
        np.radians(target_lon - origin_lon)
        * EARTH_RADIUS_KM
        * np.cos(np.radians(origin_lat))
    )
    return np.column_stack([east_km, north_km])


def _offsets_to_coordinates(
    offsets: np.ndarray, origin_lat: float, origin_lon: float
) -> tuple[np.ndarray, np.ndarray]:
    latitude = origin_lat + np.degrees(offsets[:, 1] / EARTH_RADIUS_KM)
    cosine_latitude = max(abs(np.cos(np.radians(origin_lat))), 1e-8)
    longitude = origin_lon + np.degrees(
        offsets[:, 0] / (EARTH_RADIUS_KM * cosine_latitude)
    )
    return latitude, longitude


def _scenario_errors(prediction: np.ndarray, row: pd.Series) -> np.ndarray:
    predicted_lat, predicted_lon = _offsets_to_coordinates(
        prediction, float(row["input_30_lat"]), float(row["input_30_lon"])
    )
    observed_lat = np.array(
        [row[f"target_{step:02d}_lat"] for step in range(1, TARGET_STEPS + 1)]
    )
    observed_lon = np.array(
        [row[f"target_{step:02d}_lon"] for step in range(1, TARGET_STEPS + 1)]
    )
    return _haversine_distances_km(
        predicted_lat, predicted_lon, observed_lat, observed_lon
    )


def _metrics(errors: np.ndarray) -> dict[str, float]:
    return {
        "ADE_km": float(errors.mean()),
        "FDE_km": float(errors[-1]),
        "Max_Error_km": float(errors.max()),
    }


def _bootstrap_mean_ci(
    values: np.ndarray, seed: int, iterations: int = 10_000
) -> list[float]:
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(iterations, len(values)), replace=True).mean(
        axis=1
    )
    return [float(value) for value in np.quantile(samples, [0.025, 0.975])]


def main() -> int:
    args = parse_args()
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1:
        raise ValueError("epochs, patience and batch-size must all be positive.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = pd.read_csv(args.data)
    if data["scenario_id"].duplicated().any():
        raise ValueError("Scenario IDs must be unique for paired model evaluation.")
    if set(data["split"].unique()) != {"train", "validation", "test"}:
        raise ValueError("Expected train, validation, and test partitions.")
    split_mmsi = {
        split: set(data.loc[data["split"] == split, "mmsi"])
        for split in ("train", "validation", "test")
    }
    if any(
        split_mmsi[left] & split_mmsi[right]
        for left, right in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        )
    ):
        raise ValueError("MMSI leakage detected across partitions.")

    x = np.stack([_sequence_matrix(row) for _, row in data.iterrows()]).astype(
        "float32"
    )
    y = np.stack([_target_offsets(row) for _, row in data.iterrows()]).astype("float32")
    train_mask = data["split"].eq("train").to_numpy()
    validation_mask = data["split"].eq("validation").to_numpy()
    test_mask = data["split"].eq("test").to_numpy()
    if min(train_mask.sum(), validation_mask.sum(), test_mask.sum()) == 0:
        raise ValueError("All train/validation/test partitions must be non-empty.")

    # Fit input and target scaling on training examples only.
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

    def tensor_loader(mask: np.ndarray, shuffle: bool) -> DataLoader:
        dataset = TensorDataset(
            torch.from_numpy(x_scaled[mask]), torch.from_numpy(y_scaled[mask])
        )
        generator = torch.Generator().manual_seed(args.seed) if shuffle else None
        return DataLoader(
            dataset, batch_size=args.batch_size, shuffle=shuffle, generator=generator
        )

    train_loader = tensor_loader(train_mask, True)
    val_x = torch.from_numpy(x_scaled[validation_mask]).to(device)
    val_y_km = y[validation_mask]
    model = AISLSTM(hidden_size=args.hidden_size).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    loss_function = nn.SmoothL1Loss()
    best_validation_ade = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        batch_losses = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_x)
            loss = loss_function(prediction, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            validation_prediction = (
                model(val_x).cpu().numpy() * target_scale + target_mean
            )
        val_errors = np.stack(
            [
                _scenario_errors(prediction, row)
                for prediction, (_, row) in zip(
                    validation_prediction,
                    data.loc[validation_mask].iterrows(),
                    strict=True,
                )
            ]
        )
        validation_ade = float(val_errors.mean())
        history.append(
            {
                "epoch": epoch,
                "train_smooth_l1_scaled": float(np.mean(batch_losses)),
                "validation_ADE_km": validation_ade,
            }
        )
        if validation_ade < best_validation_ade - 1e-6:
            best_validation_ade = validation_ade
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= args.patience:
            break

    if best_state is None:
        raise RuntimeError("No best model state was selected on validation.")
    model.load_state_dict(best_state)
    model.eval()
    test_x = torch.from_numpy(x_scaled[test_mask]).to(device)
    test_rows = data.loc[test_mask].reset_index(drop=True)
    warmup_count = min(10, len(test_rows))
    with torch.no_grad():
        _ = model(test_x[:warmup_count])
        if device.type == "cuda":
            torch.cuda.synchronize()
        latency_runs = []
        single_scenario = test_x[:1]
        for _ in range(20):
            _ = model(single_scenario)
        if device.type == "cuda":
            torch.cuda.synchronize()
        for _ in range(100):
            started = time.perf_counter()
            _ = model(single_scenario)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latency_runs.append((time.perf_counter() - started) * 1000.0)
        predicted_test_offsets = (
            model(test_x).cpu().numpy() * target_scale + target_mean
        )

    per_scenario_rows = []
    prediction_rows = []
    for prediction, (_, row) in zip(
        predicted_test_offsets, test_rows.iterrows(), strict=True
    ):
        errors = _scenario_errors(prediction, row)
        scenario_id = row["scenario_id"]
        per_scenario_rows.append(
            {
                "scenario_id": scenario_id,
                "trajectory_id": row["trajectory_id"],
                "mmsi": int(row["mmsi"]),
                "split": row["split"],
                "ADE_LSTM_km": float(errors.mean()),
                "FDE_LSTM_km": float(errors[-1]),
                "Max_Error_LSTM_km": float(errors.max()),
                "ADE_Persistencia_km": float(row["ADE_Persistencia_km"]),
                "FDE_Persistencia_km": float(row["FDE_Persistencia_km"]),
                "Max_Error_Persistencia_km": float(row["Max_Error_Persistencia_km"]),
            }
        )
        predicted_lat, predicted_lon = _offsets_to_coordinates(
            prediction, float(row["input_30_lat"]), float(row["input_30_lon"])
        )
        persistence_lat = np.full(TARGET_STEPS, float(row["input_30_lat"]))
        persistence_lon = np.full(TARGET_STEPS, float(row["input_30_lon"]))
        observed_lat = np.array(
            [row[f"target_{step:02d}_lat"] for step in range(1, TARGET_STEPS + 1)]
        )
        observed_lon = np.array(
            [row[f"target_{step:02d}_lon"] for step in range(1, TARGET_STEPS + 1)]
        )
        persistence_errors = _haversine_distances_km(
            persistence_lat, persistence_lon, observed_lat, observed_lon
        )
        for step, (lat, lon, error_km) in enumerate(
            zip(predicted_lat, predicted_lon, errors, strict=True), start=1
        ):
            prediction_rows.append(
                {
                    "scenario_id": scenario_id,
                    "trajectory_id": row["trajectory_id"],
                    "mmsi": int(row["mmsi"]),
                    "horizon_step": step,
                    "actual_lat": row[f"target_{step:02d}_lat"],
                    "actual_lon": row[f"target_{step:02d}_lon"],
                    "predicted_lat": float(lat),
                    "predicted_lon": float(lon),
                    "geodesic_error_km": float(error_km),
                    "persistence_lat": float(persistence_lat[step - 1]),
                    "persistence_lon": float(persistence_lon[step - 1]),
                    "persistence_error_km": float(persistence_errors[step - 1]),
                }
            )

    per_scenario = pd.DataFrame(per_scenario_rows)
    difference = (
        per_scenario["ADE_LSTM_km"] - per_scenario["ADE_Persistencia_km"]
    ).to_numpy(dtype="float64")
    wins = int((difference < 0).sum())
    losses = int((difference > 0).sum())
    ties = int((difference == 0).sum())
    t_result = stats.ttest_1samp(difference, popmean=0.0)
    wilcoxon_result = stats.wilcoxon(difference) if np.any(difference != 0) else None
    sign_test = stats.binomtest(wins, wins + losses, p=0.5, alternative="two-sided")
    shapiro_result = stats.shapiro(difference) if 3 <= len(difference) <= 5000 else None
    report = {
        "study": "Held-out paired comparison: LSTM versus persistence baseline",
        "primary_outcome": "ADE (mean great-circle position error over 10 future AIS messages), km",
        "test_unit": "scenario; test partition is MMSI-disjoint from train and validation",
        "n_test_scenarios": int(len(per_scenario)),
        "n_test_vessels": int(per_scenario["mmsi"].nunique()),
        "seed": args.seed,
        "device": str(device),
        "torch_version": torch.__version__,
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "training": {
            "epochs_run": len(history),
            "best_epoch_by_validation_ADE": best_epoch,
            "best_validation_ADE_km": best_validation_ade,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "architecture": "Encoder LSTM(6 features, hidden=16) + MLP to 10 local east/north displacements",
            "input_scaler_fit_on": "train only",
            "target_scaling_fit_on": "train only",
            "selection_metric": "validation ADE in km; test not used for checkpoint selection",
        },
        "test_metrics": {
            "LSTM": {
                "ADE_mean_km": float(per_scenario["ADE_LSTM_km"].mean()),
                "ADE_median_km": float(per_scenario["ADE_LSTM_km"].median()),
                "FDE_mean_km": float(per_scenario["FDE_LSTM_km"].mean()),
                "Max_error_mean_km": float(per_scenario["Max_Error_LSTM_km"].mean()),
                "single_scenario_latency_median_ms": float(np.median(latency_runs)),
            },
            "persistence_baseline": {
                "ADE_mean_km": float(per_scenario["ADE_Persistencia_km"].mean()),
                "ADE_median_km": float(per_scenario["ADE_Persistencia_km"].median()),
                "FDE_mean_km": float(per_scenario["FDE_Persistencia_km"].mean()),
                "Max_error_mean_km": float(
                    per_scenario["Max_Error_Persistencia_km"].mean()
                ),
            },
            "ADE_difference_LSTM_minus_persistence_km": float(difference.mean()),
            "ADE_difference_median_LSTM_minus_persistence_km": float(
                np.median(difference)
            ),
            "ADE_relative_improvement_percent": float(
                100.0
                * (
                    1.0
                    - per_scenario["ADE_LSTM_km"].mean()
                    / per_scenario["ADE_Persistencia_km"].mean()
                )
            ),
            "LSTM_lower_ADE_scenarios": wins,
            "LSTM_higher_ADE_scenarios": losses,
            "ADE_ties": ties,
            "ADE_difference_mean_bootstrap_ci95_km": _bootstrap_mean_ci(
                difference, args.seed
            ),
            "paired_standardized_effect_dz": (
                float(difference.mean() / difference.std(ddof=1))
                if difference.std(ddof=1) > 0
                else None
            ),
            "LSTM_better_than_persistence_on_mean_ADE": bool(
                per_scenario["ADE_LSTM_km"].mean()
                < per_scenario["ADE_Persistencia_km"].mean()
            ),
        },
        "paired_inference": {
            "paired_t_test_on_scenario_ADE_differences": {
                "statistic": float(t_result.statistic),
                "p_value": float(t_result.pvalue),
                "degrees_of_freedom": int(len(difference) - 1),
            },
            "wilcoxon_signed_rank_on_scenario_ADE_differences": (
                {
                    "statistic": float(wilcoxon_result.statistic),
                    "p_value": float(wilcoxon_result.pvalue),
                }
                if wilcoxon_result is not None
                else None
            ),
            "exact_sign_test_on_scenario_ADE_differences": {
                "LSTM_better_scenarios": wins,
                "LSTM_worse_scenarios": losses,
                "ties": ties,
                "p_value": float(sign_test.pvalue),
            },
            "shapiro_wilk_on_scenario_ADE_differences": (
                {
                    "statistic": float(shapiro_result.statistic),
                    "p_value": float(shapiro_result.pvalue),
                }
                if shapiro_result is not None
                else None
            ),
            "interpretation_note": "Distribution is strongly non-normal and mean/median conclusions disagree. Do not use the paired t-test as confirmatory evidence. Wilcoxon did not show a significant shift; the sign test is borderline and shows fewer wins than losses. This is exploratory with one AIS date and 74 test vessels. A non-significant result is not evidence of equivalence.",
        },
        "limitations": [
            "One NOAA day and 369 sampled scenarios are insufficient for population-wide or seasonal generalization.",
            "The model predicts the next 10 AIS messages, whose elapsed time differs by scenario; report the actual time horizon.",
            "AIS positions are the reference observations, not independent high-precision ground truth.",
            "This experiment evaluates geolocation prediction, not operational route optimization.",
            "Only LSTM is trained here; GRU comparison remains future work.",
        ],
    }

    config = get_config()
    output_dir = config.final_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = config.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)
    test_csv = output_dir / "spss_geolocation_study.csv"
    full_table = pd.read_csv(test_csv)
    for column in (
        "ADE_LSTM_km",
        "FDE_LSTM_km",
        "Max_Error_LSTM_km",
    ):
        full_table[column] = pd.Series(dtype="float64")
    test_result_map = per_scenario.set_index("scenario_id")
    is_test = full_table["split"].eq("test")
    for metric in ("ADE_LSTM_km", "FDE_LSTM_km", "Max_Error_LSTM_km"):
        full_table.loc[is_test, metric] = full_table.loc[is_test, "scenario_id"].map(
            test_result_map[metric]
        )
    full_table.loc[is_test, "Latencia_LSTM_ms"] = float(np.median(latency_runs))
    _write_csv(full_table, test_csv)
    per_scenario.to_csv(
        output_dir / "lstm_vs_persistence_test_metrics.csv", index=False
    )
    pd.DataFrame(prediction_rows).to_csv(
        output_dir / "lstm_test_point_predictions.csv", index=False
    )
    pd.DataFrame(history).to_csv(reports_dir / "lstm_training_history.csv", index=False)
    with open(reports_dir / "lstm_test_evaluation.json", "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    quality_path = output_dir / "dataset_quality.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    quality["trained_model_evaluation"] = {
        "model": "LSTM",
        "test_metrics_written_only_to_test_rows": True,
        "test_scenarios": int(len(per_scenario)),
        "test_vessels": int(per_scenario["mmsi"].nunique()),
        "checkpoint_selected_on_validation_not_test": True,
        "results_file": "../reports/lstm_test_evaluation.json",
        "point_predictions_file": "lstm_test_point_predictions.csv",
    }
    spss_export = quality.get("spss_export", {})
    model_result_columns = [
        "ADE_LSTM_km",
        "ADE_GRU_km",
        "FDE_LSTM_km",
        "FDE_GRU_km",
        "Max_Error_LSTM_km",
        "Max_Error_GRU_km",
        "Latencia_LSTM_ms",
        "Latencia_GRU_ms",
    ]
    spss_export["expected_blank_model_results"] = [
        column for column in model_result_columns if full_table[column].isna().all()
    ]
    spss_export["model_results_populated_only_for_split"] = "test"
    spss_export["intentionally_blank_model_cells"] = int(
        full_table[model_result_columns].isna().sum().sum()
    )
    quality["spss_export"] = spss_export
    quality_temp = quality_path.with_name(f"{quality_path.name}.tmp")
    quality_temp.write_text(json.dumps(quality, indent=2), encoding="utf-8")
    quality_temp.replace(quality_path)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_scaler_mean": input_scaler.mean_.tolist(),
            "input_scaler_scale": input_scaler.scale_.tolist(),
            "target_mean": target_mean.tolist(),
            "target_scale": target_scale.tolist(),
            "seed": args.seed,
            "best_epoch": best_epoch,
            "architecture": "AISLSTM",
        },
        reports_dir / "lstm_geolocation_model.pt",
    )
    print(json.dumps(report, indent=2))
    print(f"Wrote model evaluation outputs to {output_dir} and {reports_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

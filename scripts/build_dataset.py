#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config.settings import get_config
from src.scenarios.windows import split_scenarios_by_vessel


FEATURES = ["LAT", "LON", "SOG", "COG"]


def _wide_csv_row(record: dict, split_name: str) -> dict:
    row = {
        "split": split_name,
        "scenario_id": record["scenario_id"],
        "trajectory_id": record["trajectory_id"],
        "mmsi": record["mmsi"],
    }
    for role in ("input", "target"):
        for step, (timestamp, values) in enumerate(
            zip(record[f"{role}_timestamps"], record[role], strict=True), start=1
        ):
            prefix = f"{role}_{step:02d}"
            row[f"{prefix}_timestamp_utc"] = timestamp
            row.update(
                {
                    f"{prefix}_{feature.lower()}": float(value)
                    for feature, value in zip(FEATURES, values, strict=True)
                }
            )
    return row


def _write_csv(df: pd.DataFrame, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.tmp")
    df.to_csv(temporary_path, index=False)
    temporary_path.replace(path)


def _haversine_distances_km(
    latitude_1, longitude_1, latitude_2, longitude_2
) -> np.ndarray:
    latitude_1 = np.radians(np.asarray(latitude_1, dtype="float64"))
    longitude_1 = np.radians(np.asarray(longitude_1, dtype="float64"))
    latitude_2 = np.radians(np.asarray(latitude_2, dtype="float64"))
    longitude_2 = np.radians(np.asarray(longitude_2, dtype="float64"))
    delta_latitude = latitude_2 - latitude_1
    delta_longitude = longitude_2 - longitude_1
    haversine_a = (
        np.sin(delta_latitude / 2.0) ** 2
        + np.cos(latitude_1) * np.cos(latitude_2) * np.sin(delta_longitude / 2.0) ** 2
    )
    haversine_a = np.clip(haversine_a, 0.0, 1.0)
    return 2.0 * 6371.0 * np.arctan2(np.sqrt(haversine_a), np.sqrt(1.0 - haversine_a))


def _make_spss_ready_table(
    wide_rows: list[dict], metadata_rows: list[dict]
) -> pd.DataFrame:
    """One numeric, flat row per scenario with actual targets and baseline outcomes."""
    wide = pd.DataFrame(wide_rows).reset_index(drop=True)
    metadata = pd.DataFrame(metadata_rows).reset_index(drop=True)
    if len(wide) != len(metadata):
        raise ValueError("Scenario feature and metadata row counts do not match.")
    if wide["scenario_id"].tolist() != metadata["scenario_id"].tolist():
        raise ValueError("Scenario feature and metadata order does not match.")

    timestamp_columns = [column for column in wide if column.endswith("_timestamp_utc")]
    first_timestamp = pd.to_datetime(wide["input_01_timestamp_utc"], utc=True)
    wide["fecha_utc"] = first_timestamp.dt.strftime("%Y-%m-%d")
    for column in timestamp_columns:
        role_step = column.removesuffix("_timestamp_utc")
        timestamps = pd.to_datetime(wide[column], utc=True)
        wide[f"{role_step}_elapsed_sec"] = (
            timestamps - first_timestamp
        ).dt.total_seconds()
    wide = wide.drop(columns=timestamp_columns)

    prediction_latitude = wide["input_30_lat"].to_numpy()
    prediction_longitude = wide["input_30_lon"].to_numpy()
    baseline_errors = np.column_stack(
        [
            _haversine_distances_km(
                prediction_latitude,
                prediction_longitude,
                wide[f"target_{step:02d}_lat"].to_numpy(),
                wide[f"target_{step:02d}_lon"].to_numpy(),
            )
            for step in range(1, 11)
        ]
    )
    metrics = _build_model_evaluation_template(metadata.to_dict(orient="records"))
    metrics["ADE_Persistencia_km"] = baseline_errors.mean(axis=1)
    metrics["FDE_Persistencia_km"] = baseline_errors[:, -1]
    metrics["Max_Error_Persistencia_km"] = baseline_errors.max(axis=1)
    extra_metrics = metrics.drop(
        columns=["ID_Escenario_Ruta", "ID_Trayectoria", "MMSI", "Particion"]
    )
    result = pd.concat([wide, extra_metrics], axis=1)
    populated_columns = [
        column
        for column in result.columns
        if not any(model in column for model in ("LSTM", "GRU"))
    ]
    if result[populated_columns].isna().any().any():
        raise ValueError("SPSS analysis table contains missing source/baseline values.")
    if not np.isfinite(
        result.select_dtypes(include=["number"]).to_numpy(dtype="float64")
    ).all():
        raise ValueError("SPSS analysis table contains non-finite numeric values.")
    return result


def _build_model_evaluation_template(metadata_rows: list[dict]) -> pd.DataFrame:
    """Blank, paired-per-scenario evaluation sheet for model predictions."""
    columns = [
        "ID_Escenario_Ruta",
        "ID_Trayectoria",
        "MMSI",
        "Particion",
        "Horizonte_final_min",
        "Span_objetivo_min",
        "ADE_LSTM_km",
        "ADE_GRU_km",
        "ADE_Persistencia_km",
        "FDE_LSTM_km",
        "FDE_GRU_km",
        "FDE_Persistencia_km",
        "Max_Error_LSTM_km",
        "Max_Error_GRU_km",
        "Max_Error_Persistencia_km",
        "Latencia_LSTM_ms",
        "Latencia_GRU_ms",
    ]
    rows = []
    for metadata in metadata_rows:
        input_end = pd.Timestamp(metadata["input_end_utc"])
        target_start = pd.Timestamp(metadata["target_start_utc"])
        target_end = pd.Timestamp(metadata["target_end_utc"])
        rows.append(
            {
                "ID_Escenario_Ruta": metadata["scenario_id"],
                "ID_Trayectoria": metadata["trajectory_id"],
                "MMSI": metadata["mmsi"],
                "Particion": metadata["split"],
                "Horizonte_final_min": (target_end - input_end).total_seconds() / 60.0,
                "Span_objetivo_min": (target_end - target_start).total_seconds() / 60.0,
                "ADE_LSTM_km": pd.NA,
                "ADE_GRU_km": pd.NA,
                "ADE_Persistencia_km": metadata.get("ADE_Persistencia_km", pd.NA),
                "FDE_LSTM_km": pd.NA,
                "FDE_GRU_km": pd.NA,
                "FDE_Persistencia_km": metadata.get("FDE_Persistencia_km", pd.NA),
                "Max_Error_LSTM_km": pd.NA,
                "Max_Error_GRU_km": pd.NA,
                "Max_Error_Persistencia_km": metadata.get(
                    "Max_Error_Persistencia_km", pd.NA
                ),
                "Latencia_LSTM_ms": pd.NA,
                "Latencia_GRU_ms": pd.NA,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _derive_motion_features(points: pd.DataFrame) -> pd.DataFrame:
    points = points.sort_values("timestamp").reset_index(drop=True).copy()
    points["delta_time_seconds"] = (
        points["timestamp"].diff().dt.total_seconds().fillna(0.0)
    )
    latitude = points["LAT"].to_numpy(dtype="float64")
    longitude = points["LON"].to_numpy(dtype="float64")
    previous_latitude = np.roll(latitude, 1)
    previous_longitude = np.roll(longitude, 1)
    previous_latitude[0] = latitude[0]
    previous_longitude[0] = longitude[0]
    phi1 = np.radians(previous_latitude)
    phi2 = np.radians(latitude)
    delta_phi = np.radians(latitude - previous_latitude)
    delta_lambda = np.radians(longitude - previous_longitude)
    haversine_a = (
        np.sin(delta_phi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    )
    haversine_a = np.clip(haversine_a, 0.0, 1.0)
    points["distance_from_previous_km"] = (
        2.0 * 6371.0 * np.arctan2(np.sqrt(haversine_a), np.sqrt(1.0 - haversine_a))
    )
    course_radians = np.radians(points["COG"].to_numpy(dtype="float64"))
    points["cog_sin"] = np.sin(course_radians)
    points["cog_cos"] = np.cos(course_radians)
    points["sog_kmh"] = points["SOG"] * 1.852
    return points


def _write_analysis_metadata(
    output_dir: Path,
    long_rows: list[dict],
    metadata_rows: list[dict],
    config,
) -> None:
    long_df = pd.DataFrame(long_rows)
    metadata_df = pd.DataFrame(metadata_rows)
    dictionary = [
        (
            "Observations",
            "split",
            "Partición del dataset",
            "Nominal",
            "train/validation/test",
        ),
        (
            "Observations",
            "scenario_id",
            "Identificador del escenario; solo clave",
            "Nominal",
            "—",
        ),
        (
            "Observations",
            "trajectory_id",
            "Identificador de trayectoria; solo clave",
            "Nominal",
            "—",
        ),
        (
            "Observations",
            "mmsi",
            "Identificador del buque; tratar como nominal, no cuantitativo",
            "Nominal",
            "—",
        ),
        (
            "Observations",
            "sequence_role",
            "Tramo observado o predicción objetivo",
            "Nominal",
            "input/target",
        ),
        (
            "Observations",
            "step",
            "Orden de la observación dentro del tramo",
            "Ordinal",
            "paso 1..30 o 1..10",
        ),
        (
            "Observations",
            "timestamp_utc",
            "Fecha/hora AIS en UTC",
            "Fecha/tiempo",
            "ISO-8601 UTC",
        ),
        ("Observations", "LAT", "Latitud geográfica", "Escala", "grados; -90..90"),
        ("Observations", "LON", "Longitud geográfica", "Escala", "grados; -180..180"),
        ("Observations", "SOG", "Velocidad sobre el fondo", "Escala", "nudos"),
        (
            "Observations",
            "COG",
            "Rumbo sobre el fondo; variable circular",
            "Circular",
            "grados; interpretar con cuidado en pruebas lineales",
        ),
        ("Observations", "sog_kmh", "Velocidad convertida desde SOG", "Escala", "km/h"),
        (
            "Observations",
            "cog_sin",
            "Componente seno del rumbo",
            "Escala",
            "adimensional; usar junto con cog_cos",
        ),
        (
            "Observations",
            "cog_cos",
            "Componente coseno del rumbo",
            "Escala",
            "adimensional; usar junto con cog_sin",
        ),
        (
            "Observations",
            "delta_time_seconds",
            "Tiempo desde el punto AIS previo",
            "Escala",
            "segundos; 0 para el primer punto de la trayectoria",
        ),
        (
            "Observations",
            "distance_from_previous_km",
            "Distancia Haversine desde el punto AIS previo",
            "Escala",
            "km; 0 para el primer punto de la trayectoria",
        ),
        (
            "Vessel summary",
            "n_scenarios",
            "Escenarios incluidos para el buque",
            "Escala",
            "conteo",
        ),
        (
            "Vessel summary",
            "mean_sog_knots",
            "Velocidad media por buque",
            "Escala",
            "nudos",
        ),
        (
            "Vessel summary",
            "sd_sog_knots",
            "Desviación estándar de velocidad por buque",
            "Escala",
            "nudos",
        ),
        (
            "Vessel summary",
            "mean_course_deg",
            "Rumbo medio circular por buque",
            "Circular",
            "grados",
        ),
        (
            "Vessel summary",
            "total_distance_km",
            "Distancia en las ventanas seleccionadas",
            "Escala",
            "km; no es una ruta completa",
        ),
        (
            "Vessel summary",
            "input_mean_sog_knots",
            "SOG media del tramo de entrada por buque",
            "Escala",
            "nudos",
        ),
        (
            "Vessel summary",
            "target_mean_sog_knots",
            "SOG media del tramo objetivo por buque",
            "Escala",
            "nudos",
        ),
        (
            "Vessel summary",
            "target_minus_input_mean_sog_knots",
            "Cambio descriptivo de SOG entre tramos emparejados",
            "Escala",
            "nudos",
        ),
        (
            "Scenario metadata",
            "input_start_utc/input_end_utc",
            "Inicio/fin del tramo de entrada",
            "Fecha/tiempo",
            "ISO-8601 UTC",
        ),
        (
            "Scenario metadata",
            "target_start_utc/target_end_utc",
            "Inicio/fin del tramo objetivo",
            "Fecha/tiempo",
            "ISO-8601 UTC",
        ),
        (
            "spss_geolocation_study.csv",
            "input_NN_* / target_NN_*",
            "Posición, SOG y COG por paso AIS; input son predictores y target es la posición observada de referencia",
            "Escala/circular según variable",
            "LAT/LON en grados, SOG en nudos, COG circular en grados",
        ),
        (
            "spss_geolocation_study.csv",
            "input_NN_elapsed_sec / target_NN_elapsed_sec",
            "Tiempo transcurrido desde el primer punto de entrada, para tratar el muestreo AIS irregular",
            "Escala",
            "segundos",
        ),
        (
            "spss_geolocation_study.csv",
            "ADE_Persistencia_km / FDE_Persistencia_km / Max_Error_Persistencia_km",
            "Errores geodésicos reales de un baseline que repite la última posición de entrada",
            "Escala",
            "km; baseline calculado contra los puntos target observados",
        ),
        (
            "spss_geolocation_study.csv",
            "ADE_LSTM_km / ADE_GRU_km / FDE_* / Max_Error_* / Latencia_*",
            "Resultados del LSTM para test; GRU queda pendiente de implementación",
            "Escala",
            "km para errores; ms para latencia; train/validation quedan sin métricas de inferencia",
        ),
    ]
    role_counts = (
        long_df.groupby(["scenario_id", "sequence_role"]).size().unstack(fill_value=0)
    )
    sequence_lengths_valid = bool(
        (
            role_counts.get("input", pd.Series(dtype="int64"))
            == config.input_sequence_length
        ).all()
        and (
            role_counts.get("target", pd.Series(dtype="int64"))
            == config.prediction_horizon
        ).all()
    )
    timestamp_order_valid = bool(
        long_df.groupby("scenario_id", sort=False)["timestamp_utc"]
        .apply(lambda values: pd.to_datetime(values, utc=True).is_monotonic_increasing)
        .all()
    )
    targets_follow_inputs = bool(
        (
            pd.to_datetime(metadata_df["target_start_utc"], utc=True)
            > pd.to_datetime(metadata_df["input_end_utc"], utc=True)
        ).all()
    )
    vessels = long_df.groupby("mmsi", as_index=False).agg(
        split=("split", "first"),
        n_scenarios=("scenario_id", "nunique"),
        n_observations=("scenario_id", "size"),
        mean_lat=("LAT", "mean"),
        mean_lon=("LON", "mean"),
        mean_sog_knots=("SOG", "mean"),
        sd_sog_knots=("SOG", "std"),
        mean_sog_kmh=("sog_kmh", "mean"),
        mean_delta_time_seconds=("delta_time_seconds", "mean"),
        total_distance_km=("distance_from_previous_km", "sum"),
        mean_cog_sin=("cog_sin", "mean"),
        mean_cog_cos=("cog_cos", "mean"),
    )
    vessels["mean_course_deg"] = (
        np.degrees(np.arctan2(vessels["mean_cog_sin"], vessels["mean_cog_cos"])) % 360.0
    )
    vessels["course_concentration"] = np.hypot(
        vessels["mean_cog_sin"], vessels["mean_cog_cos"]
    )
    vessels = vessels.drop(columns=["mean_cog_sin", "mean_cog_cos"])
    scenario_speed = (
        long_df.groupby(["mmsi", "scenario_id", "sequence_role"])["SOG"]
        .mean()
        .groupby(["mmsi", "sequence_role"])
        .mean()
        .unstack("sequence_role")
    )
    scenario_speed = scenario_speed.rename(
        columns={
            "input": "input_mean_sog_knots",
            "target": "target_mean_sog_knots",
        }
    )
    vessels = vessels.merge(scenario_speed, on="mmsi", how="left")
    vessels["target_minus_input_mean_sog_knots"] = (
        vessels["target_mean_sog_knots"] - vessels["input_mean_sog_knots"]
    )
    _write_csv(vessels, output_dir / "vessel_summary.csv")
    dictionary_df = pd.DataFrame(
        dictionary,
        columns=[
            "file",
            "variable",
            "description_es",
            "measurement_level_spss",
            "unit_or_values",
        ],
    )
    dictionary_df["file"] = dictionary_df["file"].replace(
        {
            "Observations": "observations_long.csv",
            "Vessel summary": "vessel_summary.csv",
            "Scenario metadata": "scenario_metadata.csv",
        }
    )
    _write_csv(dictionary_df, output_dir / "data_dictionary.csv")

    splits = {
        name: set(metadata_df.loc[metadata_df["split"] == name, "mmsi"])
        for name in ("train", "validation", "test")
    }
    overlaps = {
        f"{left}_vs_{right}": len(splits[left] & splits[right])
        for left, right in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        )
    }
    ranges = {
        feature: {
            "min": float(long_df[feature].min()),
            "max": float(long_df[feature].max()),
        }
        for feature in (
            "LAT",
            "LON",
            "SOG",
            "COG",
            "sog_kmh",
            "distance_from_previous_km",
        )
    }
    quality = {
        "scenario_count": int(metadata_df["scenario_id"].nunique()),
        "observation_rows": int(len(long_df)),
        "rows_per_split": {
            name: int((metadata_df["split"] == name).sum())
            for name in ("train", "validation", "test")
        },
        "independent_vessels_per_split": {
            name: int(metadata_df.loc[metadata_df["split"] == name, "mmsi"].nunique())
            for name in ("train", "validation", "test")
        },
        "unique_vessels": int(metadata_df["mmsi"].nunique()),
        "mmsi_overlap_between_splits": overlaps,
        "missing_cells_in_observations": int(long_df.isna().sum().sum()),
        "feature_ranges": ranges,
        "sequence_lengths_valid": sequence_lengths_valid,
        "timestamps_increasing_within_scenario": timestamp_order_valid,
        "targets_follow_inputs": targets_follow_inputs,
        "maximum_delta_time_seconds": float(long_df["delta_time_seconds"].max()),
        "configured_max_time_gap_seconds": float(config.max_time_gap_minutes * 60),
        "configured_max_distance_gap_km": float(config.max_distance_gap_km),
        "all_values_within_basic_bounds": bool(
            long_df["LAT"].between(-90, 90).all()
            and long_df["LON"].between(-180, 180).all()
            and long_df["SOG"].between(-0.1, 102.2).all()
            and long_df["COG"].between(0, 360, inclusive="left").all()
            and long_df["distance_from_previous_km"].ge(0).all()
            and long_df["distance_from_previous_km"]
            .le(config.max_distance_gap_km + 1e-9)
            .all()
            and long_df["delta_time_seconds"].le(config.max_time_gap_minutes * 60).all()
        ),
        "interpretation": (
            "These are consecutive AIS observations, not equally spaced samples. "
            "Use vessel/scenario as the independent unit; rows within a trajectory "
            "are serially dependent. COG is circular."
        ),
    }
    quality["passed"] = bool(
        quality["missing_cells_in_observations"] == 0
        and all(value == 0 for value in overlaps.values())
        and quality["all_values_within_basic_bounds"]
        and sequence_lengths_valid
        and timestamp_order_valid
        and targets_follow_inputs
    )
    temporary_path = output_dir / "dataset_quality.json.tmp"
    temporary_path.write_text(json.dumps(quality, indent=2), encoding="utf-8")
    temporary_path.replace(output_dir / "dataset_quality.json")
    guidance = f"""# Guía de análisis estadístico

## Unidad de análisis y tamaño efectivo

El conjunto contiene {quality['scenario_count']} escenarios y {quality['unique_vessels']} MMSI únicos. Las {quality['observation_rows']} filas de `observations_long.csv` son mediciones repetidas dentro de escenarios y buques. Para comparar modelos use las diferencias emparejadas de ADE/FDE por escenario test; el test actual contiene {quality['independent_vessels_per_split']['test']} MMSI.

## Normalidad y pruebas t

- No pruebe normalidad usando las 14,760 observaciones como si fueran independientes. Además, LAT/LON mezclan regiones geográficas y COG es circular; el rumbo crudo no debe tratarse como variable lineal normal.
- SOG puede ser asimétrica y contener muchos ceros. Revise histogramas/Q-Q plots y resúmenes por buque. La normalidad relevante para una t pareada es la de las diferencias por buque, no la de cada observación AIS.
- Para comparar dos modelos sobre los mismos buques de prueba, calcule una métrica de error por MMSI (por ejemplo, error geodésico medio por buque) y use t pareada solo si las diferencias son razonablemente simétricas/compatibles con normalidad. Si no, considere Wilcoxon pareada o bootstrap por buque.
- Para grupos independientes, compare una observación-resumen por buque; Welch es preferible a asumir varianzas iguales. Defina primero la pregunta y los grupos: train/validation/test son particiones técnicas, no tratamientos experimentales.
- Con {quality['independent_vessels_per_split']['test']} buques de prueba, la potencia depende del tamaño del efecto y variabilidad; informe intervalos de confianza y efecto, no solo p-valores. Un Shapiro-Wilk aislado no demuestra normalidad.

Defina ADE como resultado primario antes del análisis. Examine la distribución de la diferencia LSTM-baseline mediante Q-Q/Shapiro-Wilk; si no es aproximadamente normal, priorice Wilcoxon o bootstrap por MMSI. Una diferencia no significativa no demuestra equivalencia.

## Alcance para optimización de rutas

Los datos describen trayectorias AIS observadas y sus secuencias de entrada/objetivo; no contienen clima, corrientes, batimetría, zonas restringidas, puertos, combustible ni costos. Son apropiados para estudiar predicción de movimiento y estadísticas descriptivas, pero una optimización operacional requiere incorporar esas capas y definir una función de costo/restricciones. Los timestamps son irregulares; use `delta_time_seconds` o resample con una metodología explícita antes de modelos que exijan intervalos uniformes.
"""
    (output_dir / "statistical_guidance.md").write_text(guidance, encoding="utf-8")
    if not quality["passed"]:
        raise ValueError(f"Dataset quality checks failed: {quality}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build point-backed train, validation, and test windows."
    )
    parser.add_argument("--scenarios", required=True, help="Scenarios parquet path")
    parser.add_argument(
        "--points",
        default="data/trajectories/trajectory_points.parquet",
        help="Trajectory points parquet path",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = get_config()
    scenarios = pd.read_parquet(args.scenarios)
    if scenarios.empty:
        raise ValueError("No scenarios available; final dataset was not generated.")
    features = FEATURES
    split_mmsi = split_scenarios_by_vessel(scenarios, random_seed=config.random_seed)
    split_trajectory_ids = {
        name: set(scenarios.loc[scenarios["mmsi"].isin(mmsi_values), "trajectory_id"])
        for name, mmsi_values in split_mmsi.items()
    }
    output_dir = config.final_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    wide_csv_rows = {name: [] for name in split_trajectory_ids}
    long_csv_rows = []
    metadata_csv_rows = []
    for split_name, trajectory_ids in split_trajectory_ids.items():
        split_scenarios = scenarios[scenarios["trajectory_id"].isin(trajectory_ids)]
        if split_scenarios.empty:
            continue
        points = pd.read_parquet(
            args.points,
            filters=[("trajectory_id", "in", list(trajectory_ids))],
        )
        missing = set(features + ["trajectory_id", "timestamp"]) - set(points.columns)
        if missing:
            raise ValueError(
                f"Trajectory points file is missing columns: {sorted(missing)}"
            )
        points_by_trajectory = {
            trajectory_id: _derive_motion_features(group)
            for trajectory_id, group in points.groupby("trajectory_id", sort=False)
        }
        for scenario in split_scenarios.to_dict(orient="records"):
            trajectory_id = scenario["trajectory_id"]
            trajectory_points = points_by_trajectory.get(trajectory_id)
            if trajectory_points is None:
                continue
            input_start = int(scenario["window_start_index"])
            input_end = int(scenario["window_end_index"]) + 1
            target_start = int(scenario["prediction_start_index"])
            target_end = int(scenario["prediction_end_index"]) + 1
            input_values = trajectory_points.iloc[input_start:input_end][features]
            target_values = trajectory_points.iloc[target_start:target_end][features]
            input_points = trajectory_points.iloc[input_start:input_end]
            target_points = trajectory_points.iloc[target_start:target_end]
            if (
                len(input_values) != config.input_sequence_length
                or len(target_values) != config.prediction_horizon
            ):
                continue
            input_timestamps = [
                pd.Timestamp(value).isoformat() for value in input_points["timestamp"]
            ]
            target_timestamps = [
                pd.Timestamp(value).isoformat() for value in target_points["timestamp"]
            ]
            persistence_errors = _haversine_distances_km(
                np.full(len(target_points), float(input_values.iloc[-1]["LAT"])),
                np.full(len(target_points), float(input_values.iloc[-1]["LON"])),
                target_values["LAT"].to_numpy(),
                target_values["LON"].to_numpy(),
            )
            record = {
                "scenario_id": scenario.get("scenario_id"),
                "trajectory_id": trajectory_id,
                "mmsi": scenario.get("mmsi"),
                "input": input_values.to_numpy().tolist(),
                "target": target_values.to_numpy().tolist(),
                "input_timestamps": input_timestamps,
                "target_timestamps": target_timestamps,
            }
            wide_csv_rows[split_name].append(_wide_csv_row(record, split_name))
            metadata_csv_rows.append(
                {
                    "split": split_name,
                    "scenario_id": record["scenario_id"],
                    "trajectory_id": trajectory_id,
                    "mmsi": record["mmsi"],
                    "input_start_utc": input_timestamps[0],
                    "input_end_utc": input_timestamps[-1],
                    "target_start_utc": target_timestamps[0],
                    "target_end_utc": target_timestamps[-1],
                    "input_sequence_length": len(input_timestamps),
                    "prediction_horizon": len(target_timestamps),
                    "ADE_Persistencia_km": float(persistence_errors.mean()),
                    "FDE_Persistencia_km": float(persistence_errors[-1]),
                    "Max_Error_Persistencia_km": float(persistence_errors.max()),
                }
            )
            for role, selected_points in (
                ("input", input_points),
                ("target", target_points),
            ):
                for step, point in enumerate(
                    selected_points.itertuples(index=False), start=1
                ):
                    values = point._asdict()
                    long_csv_rows.append(
                        {
                            "split": split_name,
                            "scenario_id": record["scenario_id"],
                            "trajectory_id": trajectory_id,
                            "mmsi": record["mmsi"],
                            "sequence_role": role,
                            "step": step,
                            "timestamp_utc": pd.Timestamp(
                                values["timestamp"]
                            ).isoformat(),
                            **{feature: float(values[feature]) for feature in features},
                            "sog_kmh": float(values["sog_kmh"]),
                            "cog_sin": float(values["cog_sin"]),
                            "cog_cos": float(values["cog_cos"]),
                            "delta_time_seconds": float(values["delta_time_seconds"]),
                            "distance_from_previous_km": float(
                                values["distance_from_previous_km"]
                            ),
                        }
                    )
        if split_name == "train" and not wide_csv_rows[split_name]:
            raise ValueError(
                "Training split has no valid windows. Check trajectory count and "
                "sequence/horizon settings."
            )

    _write_csv(pd.DataFrame(long_csv_rows), output_dir / "observations_long.csv")
    _write_csv(pd.DataFrame(metadata_csv_rows), output_dir / "scenario_metadata.csv")
    spss_table = _make_spss_ready_table(
        [row for split_rows in wide_csv_rows.values() for row in split_rows],
        metadata_csv_rows,
    )
    _write_csv(spss_table, output_dir / "spss_geolocation_study.csv")
    _write_analysis_metadata(output_dir, long_csv_rows, metadata_csv_rows, config)
    quality_path = output_dir / "dataset_quality.json"
    quality_report = json.loads(quality_path.read_text(encoding="utf-8"))
    expected_pending = [
        "ADE_LSTM_km",
        "ADE_GRU_km",
        "FDE_LSTM_km",
        "FDE_GRU_km",
        "Max_Error_LSTM_km",
        "Max_Error_GRU_km",
        "Latencia_LSTM_ms",
        "Latencia_GRU_ms",
    ]
    actual_pending = sorted(
        column for column in spss_table if spss_table[column].isna().all()
    )
    expected_pending_sorted = sorted(expected_pending)
    spss_passed = (
        len(spss_table) == len(metadata_csv_rows)
        and spss_table["scenario_id"].is_unique
        and actual_pending == expected_pending_sorted
    )
    quality_report["spss_export"] = {
        "file": "spss_geolocation_study.csv",
        "rows": int(len(spss_table)),
        "columns": int(len(spss_table.columns)),
        "one_row_per_scenario": bool(spss_table["scenario_id"].is_unique),
        "baseline_metrics_complete": bool(
            spss_table[
                [
                    "ADE_Persistencia_km",
                    "FDE_Persistencia_km",
                    "Max_Error_Persistencia_km",
                ]
            ]
            .notna()
            .all()
            .all()
        ),
        "expected_blank_model_results": actual_pending,
        "passed": bool(spss_passed),
    }
    quality_report["passed"] = bool(quality_report["passed"] and spss_passed)
    quality_temp = quality_path.with_name(f"{quality_path.name}.tmp")
    quality_temp.write_text(json.dumps(quality_report, indent=2), encoding="utf-8")
    quality_temp.replace(quality_path)
    if not spss_passed:
        raise ValueError("The consolidated SPSS CSV failed its structural validation.")
    print(f"Wrote long observations and scenario metadata to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

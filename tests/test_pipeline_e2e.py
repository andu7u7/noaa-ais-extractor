import sys

import pandas as pd

from src.config.settings import ProjectConfig
from src.preprocessing.pipeline import preprocess_raw_csv
from scripts import build_dataset, build_scenarios, build_trajectories


def test_pipeline_creates_point_backed_final_dataset(tmp_path, monkeypatch):
    raw_path = tmp_path / "raw.csv"
    records = []
    for vessel in range(3):
        for minute in range(50):
            records.append(
                {
                    "MMSI": 123456780 + vessel,
                    "BaseDateTime": f"2025-05-0{vessel + 1}T00:{minute:02d}:00Z",
                    "LAT": 27.0 + vessel + minute * 0.001,
                    "LON": -80.0 + minute * 0.001,
                    "SOG": 8.0 + vessel,
                    "COG": 90.0,
                }
            )
    pd.DataFrame(records).to_csv(raw_path, index=False)

    config = ProjectConfig(
        intermediate_dir=tmp_path / "intermediate",
        trajectories_dir=tmp_path / "trajectories",
        scenarios_dir=tmp_path / "scenarios",
        final_dir=tmp_path / "final",
        min_points=20,
        min_duration_minutes=30,
        max_time_gap_minutes=30,
        input_sequence_length=30,
        prediction_horizon=10,
        random_seed=42,
    )
    monkeypatch.setattr(build_trajectories, "get_config", lambda: config)
    monkeypatch.setattr(build_scenarios, "get_config", lambda: config)
    monkeypatch.setattr(build_dataset, "get_config", lambda: config)

    cleaned_path = config.intermediate_dir / "cleaned.parquet"
    preprocess_stats = preprocess_raw_csv(raw_path, cleaned_path)
    assert preprocess_stats["records"] == 150

    monkeypatch.setattr(
        sys,
        "argv",
        ["build-trajectories", "--input", str(cleaned_path)],
    )
    assert build_trajectories.main() == 0

    metadata_path = config.trajectories_dir / "trajectory_metadata.parquet"
    points_path = config.trajectories_dir / "trajectory_points.parquet"
    assert pd.read_parquet(metadata_path)["trajectory_id"].nunique() == 3
    assert len(pd.read_parquet(points_path)) == 150

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build-scenarios",
            "--input",
            str(metadata_path),
            "--points",
            str(points_path),
        ],
    )
    assert build_scenarios.main() == 0

    scenarios_path = config.scenarios_dir / "scenarios.parquet"
    assert len(pd.read_parquet(scenarios_path)) > 0

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build-dataset",
            "--scenarios",
            str(scenarios_path),
            "--points",
            str(points_path),
        ],
    )
    assert build_dataset.main() == 0

    long_table = pd.read_csv(config.final_dir / "observations_long.csv")
    scenario_table = pd.read_csv(config.final_dir / "scenario_metadata.csv")
    dictionary = pd.read_csv(config.final_dir / "data_dictionary.csv")
    quality = pd.read_json(config.final_dir / "dataset_quality.json", typ="series")
    spss_ready = pd.read_csv(config.final_dir / "spss_geolocation_study.csv")

    assert len(spss_ready) == len(scenario_table)
    assert spss_ready["scenario_id"].is_unique
    assert spss_ready["ADE_Persistencia_km"].notna().all()
    assert spss_ready["FDE_Persistencia_km"].notna().all()
    assert spss_ready["Max_Error_Persistencia_km"].notna().all()
    assert spss_ready["ADE_LSTM_km"].isna().all()
    assert "input_30_elapsed_sec" in spss_ready.columns
    assert "target_10_elapsed_sec" in spss_ready.columns
    assert not any(column.endswith("_timestamp_utc") for column in spss_ready.columns)
    assert set(spss_ready.columns[spss_ready.isna().any()]) == {
        "ADE_LSTM_km",
        "ADE_GRU_km",
        "FDE_LSTM_km",
        "FDE_GRU_km",
        "Max_Error_LSTM_km",
        "Max_Error_GRU_km",
        "Latencia_LSTM_ms",
        "Latencia_GRU_ms",
    }
    assert (
        spss_ready.filter(regex=r"^(input|target)_\d+_elapsed_sec$").notna().all().all()
    )
    assert (
        spss_ready[
            ["ADE_Persistencia_km", "FDE_Persistencia_km", "Max_Error_Persistencia_km"]
        ]
        .ge(0)
        .all()
        .all()
    )
    assert (
        spss_ready["ADE_Persistencia_km"]
        <= spss_ready["Max_Error_Persistencia_km"] + 1e-12
    ).all()
    assert len(long_table) == 120
    assert set(long_table["sequence_role"]) == {"input", "target"}
    assert {
        "sog_kmh",
        "cog_sin",
        "cog_cos",
        "delta_time_seconds",
        "distance_from_previous_km",
    }.issubset(long_table.columns)
    assert len(scenario_table) == 3
    assert "COG" in set(dictionary["variable"])
    assert quality["passed"]
    assert quality["spss_export"]["passed"]
    assert quality["spss_export"]["rows"] == len(scenario_table)
    obsolete_exports = (
        "exploratory_sog_tests.json",
        "spss_model_evaluation_template_all.csv",
        "spss_model_evaluation_template_test.csv",
        "spss_comparison_template_369.csv",
        "spss_test_comparison_template.csv",
        "train.csv",
        "validation.csv",
        "test.csv",
        "train.parquet",
        "validation.parquet",
        "test.parquet",
    )
    assert all(not (config.final_dir / name).exists() for name in obsolete_exports)


def test_preprocessing_deduplicates_across_chunk_boundary(tmp_path):
    raw_path = tmp_path / "boundary.csv"
    pd.DataFrame(
        [
            {
                "MMSI": 123456789,
                "BaseDateTime": "2025-05-01 00:00:00",
                "LAT": 1,
                "LON": 2,
                "SOG": 1,
                "COG": 10,
            },
            {
                "MMSI": 123456789,
                "BaseDateTime": "2025-05-01 00:01:00",
                "LAT": 1,
                "LON": 2,
                "SOG": 1,
                "COG": 10,
            },
            {
                "MMSI": 123456789,
                "BaseDateTime": "2025-05-01 00:01:00",
                "LAT": 1,
                "LON": 2,
                "SOG": 1,
                "COG": 10,
            },
            {
                "MMSI": 123456789,
                "BaseDateTime": "2025-05-01 00:02:00",
                "LAT": 1,
                "LON": 2,
                "SOG": 1,
                "COG": 10,
            },
        ]
    ).to_csv(raw_path, index=False)

    output_path = tmp_path / "deduplicated.parquet"
    result = preprocess_raw_csv(raw_path, output_path, chunksize=2)
    assert result["records"] == 3
    assert result["stats"]["duplicates"] == 1
    assert len(pd.read_parquet(output_path)) == 3

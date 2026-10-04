import pandas as pd
import numpy as np
import torch

from src.validation.ais_rules import (
    validate_mmsi,
    validate_timestamp,
    validate_lat_lon,
    validate_sog,
    validate_cog,
    deduplicate_records,
    validate_raw_dataframe,
)
from src.utils.distance import haversine_km
from src.trajectories.segmenter import (
    build_trajectory_outputs,
    detect_gaps,
    segment_trajectories,
)
from src.scenarios.windows import (
    build_windows,
    split_scenarios_by_vessel,
    split_train_validation_test,
)
from src.download.noaa_downloader import NOAADownloader
from scripts.train_evaluate_lstm import AISLSTM, _haversine_distances_km
from scripts.run_spss_dl import OUTPUT_COLUMNS, SeqRegressor


def test_validate_mmsi():
    assert validate_mmsi("123456789") == 123456789
    assert validate_mmsi("0") is None
    assert validate_mmsi("9999999999") is None


def test_noaa_blob_url_pattern(tmp_path):
    downloader = NOAADownloader(base_dir=tmp_path)
    assert downloader.candidate_urls("2025-05-01") == [
        "https://noaaocm.blob.core.windows.net/ais/csv2/csv2025/"
        "ais-2025-05-01.csv.zst"
    ]


def test_validate_timestamp():
    ts = validate_timestamp("2025-05-01T00:00:00Z")
    assert ts is not None
    assert str(ts.tzinfo) == "UTC"


def test_validate_lat_lon():
    assert validate_lat_lon(10.0, 20.0) == (10.0, 20.0)
    assert validate_lat_lon(91.0, 20.0) is None
    assert validate_lat_lon(10.0, 181.0) is None


def test_validate_sog_and_cog():
    assert validate_sog(12.5) == 12.5
    assert validate_sog(-1.0) is None
    assert validate_cog(359.0) == 359.0
    assert validate_cog(360.5) is None


def test_deduplicate_records():
    df = pd.DataFrame(
        [
            {
                "mmsi": 123456789,
                "BaseDateTime": "2025-05-01T00:00:00Z",
                "LAT": 1.0,
                "LON": 2.0,
            },
            {
                "mmsi": 123456789,
                "BaseDateTime": "2025-05-01T00:00:00Z",
                "LAT": 1.0,
                "LON": 2.0,
            },
            {
                "mmsi": 123456789,
                "BaseDateTime": "2025-05-01T00:01:00Z",
                "LAT": 1.1,
                "LON": 2.1,
            },
        ]
    )
    out = deduplicate_records(df)
    assert len(out) == 2


def test_validate_raw_noaa_column_names():
    raw = pd.DataFrame(
        [
            {
                "MMSI": 123456789,
                "BaseDateTime": "2025-05-01T00:00:00Z",
                "LAT": 10.0,
                "LON": 20.0,
                "SOG": 5.0,
                "COG": 180.0,
            },
            {
                "MMSI": 123456789,
                "BaseDateTime": "2025-05-01T00:01:00Z",
                "LAT": 10.1,
                "LON": 20.1,
                "SOG": 5.5,
                "COG": 181.0,
            },
        ]
    )
    cleaned, stats = validate_raw_dataframe(raw)
    assert len(cleaned) == 2
    assert cleaned["mmsi"].tolist() == [123456789, 123456789]
    assert stats["valid_records"] == 2


def test_haversine_distance():
    d = haversine_km(0.0, 0.0, 0.0, 1.0)
    assert d > 0
    assert np.isfinite(d)


def test_lstm_output_and_geodesic_metric_units():
    model = AISLSTM(hidden_size=8)
    prediction = model(torch.zeros((2, 30, 6)))
    assert tuple(prediction.shape) == (2, 10, 2)
    distance = _haversine_distances_km([0.0], [0.0], [0.0], [1.0])
    assert 111.0 < distance[0] < 112.0


def test_gru_and_lstm_spss_regressors_have_expected_shapes():
    sample = torch.zeros((3, 30, 6))
    for cell in ("GRU", "LSTM"):
        model = SeqRegressor(cell=cell, hidden_size=8)
        assert tuple(model(sample).shape) == (3, 10, 2)
    assert OUTPUT_COLUMNS == [
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


def test_detect_gaps_and_segment():
    df = pd.DataFrame(
        [
            {
                "mmsi": 1,
                "timestamp": pd.to_datetime(
                    [
                        "2025-05-01 00:00:00Z",
                        "2025-05-01 00:01:00Z",
                        "2025-05-01 00:30:00Z",
                        "2025-05-01 00:31:00Z",
                    ]
                ),
            },
        ]
    )
    gaps = detect_gaps(df, max_gap_minutes=10)
    assert len(gaps) >= 1
    segments = segment_trajectories(df, max_gap_minutes=10)
    assert isinstance(segments, list)


def test_vectorized_trajectory_builder_splits_spatial_jumps():
    points = pd.DataFrame(
        {
            "mmsi": [123456789, 123456789, 123456789],
            "timestamp": pd.to_datetime(
                [
                    "2025-05-01T00:00:00Z",
                    "2025-05-01T00:01:00Z",
                    "2025-05-01T00:02:00Z",
                ]
            ),
            "LAT": [0.0, 0.0, 0.0],
            "LON": [0.0, 0.18, 0.19],
            "SOG": [10.0, 10.0, 10.0],
            "COG": [90.0, 90.0, 90.0],
        }
    )
    metadata, segmented_points = build_trajectory_outputs(
        points,
        max_gap_minutes=30,
        max_distance_gap_km=15.0,
        min_points=1,
        min_duration_minutes=0,
    )
    assert metadata["trajectory_id"].nunique() == 2
    assert segmented_points["trajectory_id"].nunique() == 2
    assert metadata["distance_km"].max() < 15.0


def test_build_windows_and_split():
    trajectories = [
        {
            "trajectory_id": "t1",
            "mmsi": 1,
            "point_count": 15,
            "start_time": pd.Timestamp("2025-05-01T00:00:00Z"),
            "end_time": pd.Timestamp("2025-05-01T00:50:00Z"),
        },
        {
            "trajectory_id": "t2",
            "mmsi": 2,
            "point_count": 15,
            "start_time": pd.Timestamp("2025-05-02T00:00:00Z"),
            "end_time": pd.Timestamp("2025-05-02T00:50:00Z"),
        },
        {
            "trajectory_id": "t3",
            "mmsi": 3,
            "point_count": 15,
            "start_time": pd.Timestamp("2025-05-03T00:00:00Z"),
            "end_time": pd.Timestamp("2025-05-03T00:50:00Z"),
        },
    ]
    windows = build_windows(
        trajectories, input_sequence_length=10, prediction_horizon=5
    )
    assert len(windows) >= 1
    split = split_train_validation_test(
        trajectories, val_fraction=0.2, test_fraction=0.2
    )
    assert set(split.keys()) == {"train", "validation", "test"}


def test_reproducible_split():
    trajectories = [
        {
            "trajectory_id": f"t{i}",
            "mmsi": i,
            "start_time": pd.Timestamp("2025-05-01T00:00:00Z"),
        }
        for i in range(10)
    ]
    a = split_train_validation_test(trajectories, random_seed=123)
    b = split_train_validation_test(trajectories, random_seed=123)
    assert a == b


def test_vessel_split_prevents_mmsi_leakage():
    scenarios = pd.DataFrame(
        [{"trajectory_id": f"t{i}", "mmsi": 100000000 + i} for i in range(10)]
        + [{"trajectory_id": "another_segment_same_vessel", "mmsi": 100000000}]
    )
    split = split_scenarios_by_vessel(scenarios, random_seed=19)
    assert split["train"].isdisjoint(split["validation"])
    assert split["train"].isdisjoint(split["test"])
    assert split["validation"].isdisjoint(split["test"])
    assert sum(100000000 in vessels for vessels in split.values()) == 1

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass
class ProjectConfig:
    data_year: int = int(os.getenv("DATA_YEAR", "2025"))
    raw_dir: Path = Path(os.getenv("RAW_DIR", "data/raw"))
    extracted_dir: Path = Path(os.getenv("EXTRACTED_DIR", "data/extracted"))
    intermediate_dir: Path = Path(os.getenv("INTERMEDIATE_DIR", "data/intermediate"))
    trajectories_dir: Path = Path(os.getenv("TRAJECTORIES_DIR", "data/trajectories"))
    scenarios_dir: Path = Path(os.getenv("SCENARIOS_DIR", "data/scenarios"))
    final_dir: Path = Path(os.getenv("FINAL_DIR", "data/final"))
    logs_dir: Path = Path(os.getenv("LOGS_DIR", "logs"))
    reports_dir: Path = Path(os.getenv("REPORTS_DIR", "reports"))
    test_dir: Path = Path(os.getenv("TEST_DIR", "tests"))

    min_points: int = int(os.getenv("MIN_POINTS", "20"))
    min_duration_minutes: int = int(os.getenv("MIN_DURATION_MINUTES", "30"))
    max_time_gap_minutes: int = int(os.getenv("MAX_TIME_GAP_MINUTES", "30"))
    max_distance_gap_km: float = float(os.getenv("MAX_DISTANCE_GAP_KM", "15.0"))
    input_sequence_length: int = int(os.getenv("INPUT_SEQUENCE_LENGTH", "30"))
    prediction_horizon: int = int(os.getenv("PREDICTION_HORIZON", "10"))
    target_scenarios: int = int(os.getenv("TARGET_SCENARIOS", "369"))
    random_seed: int = int(os.getenv("RANDOM_SEED", "42"))

    noaa_base_url: str = os.getenv(
        "NOAA_BASE_URL",
        "https://noaaocm.blob.core.windows.net/ais",
    )
    source_name: str = os.getenv("SOURCE_NAME", "NOAA Office for Coastal Management")

    def ensure_directories(self) -> None:
        for directory in [
            self.raw_dir,
            self.extracted_dir,
            self.intermediate_dir,
            self.trajectories_dir,
            self.scenarios_dir,
            self.final_dir,
            self.logs_dir,
            self.reports_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)


CONFIG = ProjectConfig()


def get_config() -> ProjectConfig:
    return CONFIG

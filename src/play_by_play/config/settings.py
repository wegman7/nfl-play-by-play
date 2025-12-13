"""
Configuration settings for NFL play-by-play win probability prediction project.

All configuration is centralized in dataclasses for type safety and easy access.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List

# Project root directory (3 levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class MLflowConfig:
    """MLflow tracking and experiment configuration."""

    tracking_uri: str = field(default_factory=lambda: f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")
    artifact_location: str = field(
        default_factory=lambda: (PROJECT_ROOT / "mlruns").resolve().as_uri()
    )
    experiment_name: str = "play_by_play_win_prob"
    metric: str = "r2"
    metric_higher_is_better: bool = True


@dataclass
class ModelSchemaConfig:
    """Feature and label column definitions."""

    required_input_feature_cols: List[str] = field(default_factory=lambda: [
        "game_id",
        "play_id",
        "qtr",
        "time",
        "total_home_score",
        "total_away_score",
        "home_team",
        "posteam",
        "down",
        "ydstogo",
        "yardline_100",
        "posteam_timeouts_remaining",
        "defteam_timeouts_remaining",
        "location",
    ])

    required_input_label_cols: List[str] = field(default_factory=lambda: [
        "game_id",
        "play_id",
        "result",
    ])

    key_cols: List[str] = field(default_factory=lambda: ["game_id", "play_id"])

    numeric_features: List[str] = field(default_factory=lambda: [
        "qtr",
        "total_home_score",
        "total_away_score",
        "score_diff",
        "down",
        "ydstogo",
        "yardline_100",
        "posteam_timeouts_remaining",
        "defteam_timeouts_remaining",
        "time_seconds",
        "time_seconds_total",
    ])

    categorical_features: List[str] = field(default_factory=lambda: [
        "posteam_is_home",
        "location",
    ])

    label_cols: List[str] = field(default_factory=lambda: ["win"])

    @property
    def all_feature_cols(self) -> List[str]:
        """Return all feature columns (numeric + categorical)."""
        return self.numeric_features + self.categorical_features


@dataclass
class PathConfig:
    """Data and checkpoint path configuration."""

    # Base directories
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    checkpoint_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "chk")

    # Raw data
    @property
    def raw_data_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def raw_play_by_play_2023(self) -> Path:
        return self.raw_data_dir / "play_by_play_2023.parquet"

    # Processed data
    @property
    def features_dir(self) -> Path:
        return self.data_dir / "features"

    @property
    def labels_dir(self) -> Path:
        return self.data_dir / "labels"

    @property
    def features_2023(self) -> Path:
        return self.features_dir / "play_by_play_2023.parquet"

    @property
    def labels_2023(self) -> Path:
        return self.labels_dir / "play_by_play_2023.parquet"

    # Live predictions
    @property
    def live_dir(self) -> Path:
        return self.data_dir / "live"

    @property
    def live_features_dir(self) -> Path:
        return self.live_dir / "features"

    @property
    def live_espn_dir(self) -> Path:
        return self.live_dir / "espn"

    # Checkpoints
    @property
    def nfl_predictions_checkpoint(self) -> Path:
        return self.checkpoint_dir / "nfl_predictions"


@dataclass
class RandomForestConfig:
    """RandomForestRegressor hyperparameters."""

    n_estimators: int = 100
    random_state: int = 42
    n_jobs: int = -1


@dataclass
class TrainingConfig:
    """Model training configuration."""

    test_size: float = 0.2
    random_state: int = 42
    model_config: RandomForestConfig = field(default_factory=RandomForestConfig)


@dataclass
class KafkaConfig:
    """Kafka producer/consumer configuration."""

    bootstrap_servers: List[str] = field(default_factory=lambda: ["localhost:19092"])
    topic_nfl_plays: str = "nfl_plays"
    stream_sim_sleep_seconds: float = 0.3  # Time between publishing plays
    poll_interval_seconds: int = 10


@dataclass
class ESPNAPIConfig:
    """ESPN API configuration."""

    default_game_id: str = "401772820"
    poll_interval_seconds: int = 10
    api_base_url: str = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"


@dataclass
class SparkConfig:
    """Spark streaming configuration."""

    app_name: str = "NFL Win Probability Predictor"
    shuffle_partitions: int = 4  # Reduce for local development
    log_level: str = "WARN"


@dataclass
class Settings:
    """
    Main settings object containing all configuration.

    Usage:
        from play_by_play.config.settings import settings

        # Access configuration
        mlflow_uri = settings.mlflow.tracking_uri
        features = settings.schema.all_feature_cols
        data_path = settings.paths.features_2023
    """

    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    schema: ModelSchemaConfig = field(default_factory=ModelSchemaConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    kafka: KafkaConfig = field(default_factory=KafkaConfig)
    espn: ESPNAPIConfig = field(default_factory=ESPNAPIConfig)
    spark: SparkConfig = field(default_factory=SparkConfig)


# Global settings instance
settings = Settings()

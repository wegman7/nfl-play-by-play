# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an NFL play-by-play win probability prediction project that combines machine learning with streaming data processing. It trains models on historical NFL play-by-play data and makes real-time predictions on simulated (or live) game data using Kafka and PySpark streaming.

## Environment Setup

This project uses a Python virtual environment located at `.venv/`:

```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Key Technologies

- **ML Framework**: scikit-learn with MLflow for experiment tracking
- **Streaming**: Kafka (via Docker), PySpark Structured Streaming
- **Data**: pandas, parquet files
- **Tracking**: MLflow with SQLite backend (`mlflow.db`)

## Common Commands

### Kafka Infrastructure

Start Kafka and Kafka UI:
```bash
docker compose down -v
docker compose up -d
```

- Kafka UI available at: http://localhost:8080
- Kafka broker exposed on: `localhost:19092` (external), `kafka:9092` (internal)

### Model Training

Train a win probability model:
```bash
python src/notebooks/train.py
```

This will:
- Load features and labels from `data/features/` and `data/labels/`
- Train a RandomForestRegressor
- Log the model, metrics, and test data to MLflow
- Track in experiment: `play_by_play_win_prob`

### Building Features and Labels

```bash
python src/notebooks/build_features.py  # Creates data/features/
python src/notebooks/build_labels.py    # Creates data/labels/
```

### Streaming Pipeline

1. Start the stream simulator (publishes plays to Kafka):
```bash
python src/streaming_notebooks/stream_sim.py
```

2. Start the live prediction consumer (PySpark reads from Kafka, predicts, writes to parquet):
```bash
python src/streaming_notebooks/predict_live.py
```

Live predictions are written to: `data/live/features/`

## Architecture

### Data Flow

```
Historical Data (nflfastR) → Feature Engineering → Model Training (MLflow)
                                                          ↓
Kafka Stream ← Stream Simulator                    Best Model
     ↓                                                    ↓
PySpark Consumer → Feature Engineering → Prediction → Parquet Output
```

### Directory Structure

- **`src/features/`**: Feature engineering logic (e.g., `play_by_play.py`)
- **`src/labels/`**: Label generation logic (e.g., `play_by_play.py`)
- **`src/utils/`**: Utility functions (e.g., `model_util.py` for loading best models from MLflow)
- **`src/notebooks/`**: Training and preprocessing scripts
- **`src/streaming_notebooks/`**: Kafka producer/consumer for streaming predictions
- **`data/raw/`**: Raw play-by-play parquet files
- **`data/features/`**: Processed feature parquet files
- **`data/labels/`**: Processed label parquet files
- **`data/live/`**: Real-time prediction output (gitignored)

### Feature Engineering Pipeline

Features are built via a functional pipeline pattern using `.pipe()`:

```python
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df
        .copy()
        .pipe(add_time_features)
        .pipe(add_score_features)
        .pipe(select_final_feature_columns)
    )
```

The same `build_features()` function from `src/features/play_by_play.py` is reused in both training and streaming contexts.

### Model Management

Models are tracked in MLflow with SQLite backend. The utility function `load_best_model_from_experiment()` in `src/utils/model_util.py` loads the best model from an experiment based on a metric (e.g., r2 score).

### Streaming Architecture

- **Producer** (`stream_sim.py`): Reads historical parquet data and publishes to Kafka topic `nfl_plays` at ~3 plays/sec
- **Consumer** (`predict_live.py`):
  - Uses PySpark Structured Streaming to read from Kafka
  - Applies feature engineering via `foreachBatch` with pandas UDFs
  - Loads best model from MLflow
  - Generates predictions and writes to parquet in append mode
  - Maintains checkpoint in `chk/nfl_predictions/`

## Feature Columns

**Numeric features:**
- `qtr`, `total_home_score`, `total_away_score`, `score_diff`, `down`, `ydstogo`, `yardline_100`, `posteam_timeouts_remaining`, `defteam_timeouts_remaining`, `time_seconds`

**Categorical features:**
- `home_team`, `posteam`, `location`

**Target label:**
- `win` (1 for home win, 0.5 for tie, 0 for away win)

## Data Sources

The project uses historical play-by-play data from nflfastR/nflverse. See README.md for notes on potential live data sources (ESPN API, SportsDataIO, etc.).

## MLflow Tracking

All experiments use:
- Tracking URI: `sqlite:///mlflow.db`
- Experiment name: `play_by_play_win_prob`

Models are logged with:
- Signature and input example
- Test set as artifacts (`eval_data/X_test.parquet`, `eval_data/y_test.parquet`)
- Metrics: r2 score

## Development Notes

- The project is in active development - see TODO section in README.md for planned improvements (live data integration, Airflow orchestration, pregame model, hyperparameter tuning)
- `.env` file exists but is gitignored (likely contains API keys or config)
- PySpark streaming uses `.foreachBatch()` to enable pandas/sklearn operations on micro-batches
- Feature engineering logic must be consistent between training and serving (achieved by importing the same `build_features()` function)

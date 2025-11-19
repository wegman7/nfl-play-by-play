from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import (
    FloatType,
    StructType,
    StructField,
    StringType,
    IntegerType,
)

import pandas as pd
import mlflow
import mlflow.sklearn

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.features.play_by_play import build_features

STREAM_FEATURES_DIR = ROOT / "data" / "live" / "features"

spark = (
    SparkSession.builder
    .appName("NFLPlayByPlayStreaming")
    .config("spark.sql.shuffle.partitions", "4")
    .config(
        "spark.jars.packages",
        "org.apache.spark:spark-sql-kafka-0-10_2.13:4.0.1",
    )
    .getOrCreate()
)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
EXPERIMENT_NAME = "play_by_play_win_prob"
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    raise RuntimeError(f"Experiment '{EXPERIMENT_NAME}' not found")

client = mlflow.tracking.MlflowClient()
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.r2 DESC"],
    max_results=1,
)
best_run = runs[0]
RUN_ID = best_run.info.run_id
model_uri = f"runs:/{RUN_ID}/model"
model = mlflow.sklearn.load_model(model_uri)

json_fields = [
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
    "location"
]

feature_cols = [
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
    "home_team",
    "posteam",
    "location",
]

raw = (
    spark.readStream
         .format("kafka")
         .option("kafka.bootstrap.servers", "localhost:19092")
         .option("subscribe", "nfl_plays")
         .option("startingOffsets", "earliest")   # use "earliest" while testing
         .option("failOnDataLoss", "false")
         .load()
)

schema = StructType([
    StructField("game_id", StringType()),
    StructField("play_id", IntegerType()),
    StructField("qtr", IntegerType()),
    StructField("time", StringType()),  # "MM:SS" or "HH:MM:SS"
    StructField("total_home_score", FloatType()),
    StructField("total_away_score", FloatType()),
    StructField("home_team", StringType()),
    StructField("posteam", StringType()),
    StructField("down", IntegerType()),
    StructField("ydstogo", FloatType()),
    StructField("yardline_100", FloatType()),
    StructField("posteam_timeouts_remaining", IntegerType()),
    StructField("defteam_timeouts_remaining", IntegerType()),
    StructField("location", StringType()),
])

parsed = (
    raw.select(
        F.col("key").cast("string").alias("kafka_key"),
        F.from_json(F.col("value").cast("string"), schema).alias("j"),
        "timestamp",
    )
    .select("kafka_key", "timestamp", "j.*")
)

def process_batch(batch_df, epoch_id: int):
    if batch_df.rdd.isEmpty():
        return

    # Convert Spark → pandas
    pdf = batch_df.toPandas()

    # Build your engineered features
    features = build_features(pdf)

    # Keep only rows that have *all* required model features
    good_features = features.dropna(subset=feature_cols)

    # Optional: see if anything was dropped
    num_bad = len(features) - len(good_features)
    if num_bad > 0:
        print(f"[epoch {epoch_id}] Skipping {num_bad} bad rows (NaNs in feature_cols)")
        # OPTIONAL: save these rows for debugging
        # bad_rows = features[features[feature_cols].isna().any(axis=1)]
        # bad_rows.to_parquet(f"./bad_rows_epoch_{epoch_id}.parquet", index=False)

    # If nothing valid to score, return early
    if good_features.empty:
        return

    # Run model ONLY on clean data
    X = good_features[feature_cols]
    preds = model.predict(X)
    good_features["win_prob"] = preds

    # Write results back to Spark for saving
    out_spark_df = spark.createDataFrame(good_features)
    (
        out_spark_df
        .write
        .mode("append")
        .parquet(str(STREAM_FEATURES_DIR))
    )

query = (
    parsed
    .writeStream
    .foreachBatch(process_batch)
    .outputMode("append")
    .option("checkpointLocation", "./chk/nfl_predictions")
    # .trigger(processingTime="10 seconds")
    .start()
)

spark.streams.awaitAnyTermination()
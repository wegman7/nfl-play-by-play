from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import (
    FloatType,
    StructType,
    StructField,
    StringType,
    IntegerType,
)

import pandas as pd

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.features.play_by_play import build_features
from src.utils.model_util import load_best_model_from_experiment

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

EXPERIMENT_NAME = "play_by_play_win_prob"
model, model_uri, RUN_ID = load_best_model_from_experiment(
    experiment_name=EXPERIMENT_NAME,
    tracking_uri="sqlite:///mlflow.db",
    metric="r2",
    higher_is_better=True,
)
print(f"Loaded model from {model_uri} (run_id={RUN_ID})")

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
    "location",
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

    pdf = batch_df.toPandas()
    features = build_features(pdf)

    good_features = features.dropna(subset=feature_cols)

    num_bad = len(features) - len(good_features)
    if num_bad > 0:
        print(f"[epoch {epoch_id}] Skipping {num_bad} bad rows (NaNs in feature_cols)")

    if good_features.empty:
        return

    X = good_features[feature_cols]
    preds = model.predict(X)
    good_features["win_prob"] = preds

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

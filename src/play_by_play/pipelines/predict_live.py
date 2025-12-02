from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import (
    FloatType,
    StructType,
    StructField,
    StringType,
    IntegerType,
)

import pandas as pd

from play_by_play.ml.features import build_features
from play_by_play.utils.model_util import load_best_model_from_experiment
from play_by_play.config.settings import settings


def run_predict_live():
    spark = (
        SparkSession.builder
        .appName(settings.spark.app_name)
        .config("spark.sql.shuffle.partitions", str(settings.spark.shuffle_partitions))
        .config(
            "spark.jars.packages",
            "org.apache.spark:spark-sql-kafka-0-10_2.13:4.0.1",
        )
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel(settings.spark.log_level)

    model, model_uri, RUN_ID = load_best_model_from_experiment(
        experiment_name=settings.mlflow.experiment_name,
        tracking_uri=settings.mlflow.tracking_uri,
        metric=settings.mlflow.metric,
        higher_is_better=settings.mlflow.metric_higher_is_better,
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

    raw = (
        spark.readStream
             .format("kafka")
             .option("kafka.bootstrap.servers", ",".join(settings.kafka.bootstrap_servers))
             .option("subscribe", settings.kafka.topic_nfl_plays)
             .option("startingOffsets", "earliest")
             .option("failOnDataLoss", "false")
             .load()
    )

    schema = StructType([
        StructField("game_id", StringType()),
        StructField("play_id", IntegerType()),
        StructField("qtr", IntegerType()),
        StructField("time", StringType()),
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

        good_features = features.dropna(subset=settings.schema.all_feature_cols)

        num_bad = len(features) - len(good_features)
        if num_bad > 0:
            print(f"[epoch {epoch_id}] Skipping {num_bad} bad rows (NaNs in feature_cols)")

        if good_features.empty:
            return

        X = good_features[settings.schema.all_feature_cols]
        preds = model.predict(X)
        good_features["win_prob"] = preds

        out_spark_df = spark.createDataFrame(good_features)
        (
            out_spark_df
            .write
            .mode("append")
            .parquet(str(settings.paths.live_features_dir))
        )

    query = (
        parsed
        .writeStream
        .foreachBatch(process_batch)
        .outputMode("append")
        .option("checkpointLocation", str(settings.paths.nfl_predictions_checkpoint))
        .start()
    )

    spark.streams.awaitAnyTermination()

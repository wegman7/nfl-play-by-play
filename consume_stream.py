# %%
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

spark = (
    SparkSession.builder
    .appName("NFLPlayByPlay")
    .config("spark.sql.shuffle.partitions", "4")  # friendlier locally
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.13:4.0.1")
    .getOrCreate()
)

# 1) Read from Kafka
raw = (
    spark.readStream
         .format("kafka")
         .option("kafka.bootstrap.servers", "localhost:19092")
         .option("subscribe", "nfl_plays")
         .option("startingOffsets", "latest")   # use "earliest" while testing
         .option("failOnDataLoss", "false")
         .load()
)

# 2) Parse JSON (value is bytes -> string -> columns)
schema = StructType([
    StructField("game_id", StringType()),
    StructField("play_id", IntegerType()),
    StructField("down", IntegerType()),
    StructField("distance", IntegerType()),
    StructField("yardline", IntegerType()),
    StructField("offense", StringType()),
    StructField("defense", StringType()),
    StructField("score_home", IntegerType()),
    StructField("score_away", IntegerType()),
    StructField("quarter", IntegerType()),
    StructField("clock_seconds", IntegerType()),
])

parsed = (
    raw.select(
        F.col("key").cast("string").alias("kafka_key"),
        F.from_json(F.col("value").cast("string"), schema).alias("j"),
        "timestamp"
    )
    .select("kafka_key", "timestamp", "j.*")
)

# ---- OR: to Kafka topic "nfl_predictions"
out = (parsed
    .select(
        F.col("game_id").alias("key"),  # optional: key by game
        F.to_json(F.struct(
            "game_id","play_id","offense","defense","down","distance","yardline",
            "score_home","score_away","quarter","clock_seconds","timestamp"
        )).alias("value")
    )
    .selectExpr("CAST(key AS STRING) as key", "CAST(value AS STRING) as value")
    .writeStream
    .format("kafka")
    .option("kafka.bootstrap.servers", "localhost:19092")
    .option("topic", "nfl_predictions")
    .option("checkpointLocation", "./chk/nfl_predictions")
    .outputMode("append")
    .format("console") # to view in console (will not send to Kafka)
    .start()
)

spark.streams.awaitAnyTermination()

# %%

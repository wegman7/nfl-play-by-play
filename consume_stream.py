from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("NFLPlayByPlay").getOrCreate()

df = (spark.readStream
         .format("kafka")
         .option("subscribe", "nfl_plays")
         .load())

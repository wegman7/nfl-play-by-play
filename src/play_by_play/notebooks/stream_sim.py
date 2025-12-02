from pathlib import Path
import sys

import time, json
from kafka import KafkaProducer
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

cols = [
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

df = pd.read_parquet("./data/raw/play_by_play_2023.parquet")
df = df.sort_values(["game_id", "play_id"])[cols].dropna()

producer = KafkaProducer(bootstrap_servers=["localhost:19092"],
                         value_serializer=lambda v: json.dumps(v).encode("utf-8"))

for _, row in df.iterrows():
    msg = {
        "game_id": str(row["game_id"]),
        "play_id": int(row["play_id"]),
        "qtr": int(row["qtr"]),
        "time": str(row["time"]),
        "total_home_score": float(row["total_home_score"]),
        "total_away_score": float(row["total_away_score"]),
        "home_team": str(row["home_team"]),
        "posteam": str(row["posteam"]),
        "down": int(row["down"]),
        "ydstogo": float(row["ydstogo"]),
        "yardline_100": float(row["yardline_100"]),
        "posteam_timeouts_remaining": int(row["posteam_timeouts_remaining"]),
        "defteam_timeouts_remaining": int(row["defteam_timeouts_remaining"]),
        "location": str(row["location"]),
    }

    producer.send("nfl_plays", msg)
    time.sleep(0.3)  # ~3 plays/sec (simulate live)

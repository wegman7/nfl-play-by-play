import time, json
from kafka import KafkaProducer
import pandas as pd

df = pd.read_parquet("play_by_play_2023.parquet")
df = df.sort_values(["game_id", "play_id"])

producer = KafkaProducer(bootstrap_servers=["localhost:9092"],
                         value_serializer=lambda v: json.dumps(v).encode("utf-8"))

for _, row in df.iterrows():
    msg = {
        "game_id": row.game_id,
        "play_id": int(row.play_id),
        "quarter": int(row.qtr),
        "down": int(row.down) if row.down == row.down else None,
        "yards_to_go": int(row.ydstogo),
        "epa": row.epa,
        "passer": row.passer_player_name,
        "rusher": row.rusher_player_name,
        "play_type": row.play_type,
        "desc": row.desc,
        "timestamp": str(row.game_date)
    }
    producer.send("nfl_plays", msg)
    time.sleep(0.3)  # ~3 plays/sec (simulate live)

---
description: Run live ESPN predictions with polling
---

Run the ESPN API polling script to generate live win probability predictions for an NFL game.

This script polls the ESPN API every 10 seconds (configurable) and generates predictions using the best MLflow model.

Execute:
```bash
python src/notebooks/predict_live_espn.py
```

The script uses these defaults:
- Game ID: 401772945
- Polling interval: 10 seconds
- Output directory: data/live/espn/
- Max iterations: unlimited (runs until Ctrl+C)

To customize, add arguments after the command:
- `--game-id GAME_ID`: Use a different ESPN game ID
- `--interval SECONDS`: Change polling interval
- `--max-iterations N`: Limit number of iterations (useful for testing)
- `--output-dir PATH`: Custom output directory

Examples:
- `/predict-espn --interval 30`: Poll every 30 seconds
- `/predict-espn --game-id 401547649 --max-iterations 5`: Test with different game

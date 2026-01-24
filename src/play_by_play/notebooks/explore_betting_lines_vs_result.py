# %%
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT))

# %%
from pathlib import Path
import sys
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT))

from src.play_by_play.ml.features import build_features
from src.play_by_play.ml.labels import build_labels
from src.play_by_play.ml.util import convert_posteam_to_home_pred
from src.play_by_play.config.settings import settings

# %%
data_dir = ROOT / "data" / "raw"
full_df = pd.DataFrame()
for i in range(1999, 2026):
    path = data_dir / f"play_by_play_{i}.parquet"
    df = pd.read_parquet(path)
    print(f"Year: {i}, Shape: {df.shape}")
    full_df = pd.concat([full_df, df], axis=0, ignore_index=True)

# %%
clean_df = (
    full_df[
        ['play_id', 'game_id', 'home_score', 'away_score', 'location', 'spread_line', 'total_line']
    ]
    .assign(score_diff=lambda df: df.home_score - df.away_score)
)


# %% find average score diff for all neutral location games
clean_df[['location', 'home_score', 'away_score', 'score_diff']][clean_df['location'] != 'Home'].groupby('location').mean().abs()

# %% group by game
game_df = clean_df.groupby('game_id').agg({
    'home_score': 'first',
    'away_score': 'first',
    'score_diff': 'first',
    'spread_line': 'first',
    'total_line': 'first',
})

line_diff_df = game_df.assign(
    spread_diff=lambda df: df.score_diff - df.spread_line,
    over_under_diff=lambda df: df.home_score + df.away_score - df.total_line,
)
line_diff_avg = line_diff_df[['spread_diff', 'over_under_diff']].mean()
line_diff_avg

# %%
cond_df = game_df[
    # game_df['spread_line'] > 16 # home favorite
    game_df['spread_line'] < -7  # away favorite
    # game_df['spread_line'].between(1, 3)  # favorite by 1-3 points
]


# %% find average diff between spread line and actual score diff
line_diff_cond_df = cond_df.assign(
    spread_diff=lambda df: df.score_diff - df.spread_line,
    over_under_diff=lambda df: df.home_score + df.away_score - df.total_line,
)
line_diff_cond_avg = line_diff_cond_df[['spread_diff', 'over_under_diff']].mean()
line_diff_cond_avg


# looks at:
# spread vs actual
    # home favorites
    # away favorites
    # favorites by margin (e.g., 1-3, 4-7, 8-10, 11+)
# over/under vs actual total score
# %%

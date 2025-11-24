"""
Utilities for fetching and processing ESPN NFL play-by-play data.
"""

import re
import requests
import pandas as pd


# ---------- timeout logic helpers ----------

# Team-charged timeout:
# "Timeout #2 by NYJ at 01:50."
TEAM_TIMEOUT_REGEX = re.compile(
    r"^Timeout\s+#\d+\s+by\s+([A-Z]{2,4})\b",
    flags=re.IGNORECASE,
)

# Official timeout (does NOT count against a team):
# "Official Timeout at 06:36."
OFFICIAL_TIMEOUT_REGEX = re.compile(
    r"^Official Timeout\b",
    flags=re.IGNORECASE,
)


def fetch_espn_game(game_id: str) -> dict:
    """
    Fetch ESPN gamepackageJSON for a single NFL game.
    Matches the endpoint you're already using.
    """
    url = f"https://cdn.espn.com/core/nfl/playbyplay?gameId={game_id}&xhr=1"
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.json()


def get_team_meta(game_json: dict) -> tuple[str, str, dict]:
    """
    Extract home/away team IDs and an abbrev -> team_id mapping
    from gamepackageJSON.
    """
    comps = game_json["gamepackageJSON"]["header"]["competitions"][0]["competitors"]

    home_id = None
    away_id = None
    abbrev_to_team_id: dict[str, str] = {}

    for c in comps:
        team = c["team"]
        tid = team["id"]
        abbrev = (team.get("abbreviation") or "").upper()
        if abbrev:
            abbrev_to_team_id[abbrev] = tid

        if c["homeAway"] == "home":
            home_id = tid
        elif c["homeAway"] == "away":
            away_id = tid

    if home_id is None or away_id is None:
        raise ValueError("Could not determine home/away team IDs from competitors")

    return home_id, away_id, abbrev_to_team_id


def get_all_plays(game_json: dict) -> list[dict]:
    """
    Flatten all plays from the drives structure into a single list.

    If your structure is slightly different, print out
    game_json["gamepackageJSON"].keys() and adjust this function only.
    """
    drives_obj = game_json["gamepackageJSON"]["drives"]

    # some games use "all", some use "previous" – this keeps it flexible
    if "all" in drives_obj:
        drives = drives_obj["all"]
    elif "previous" in drives_obj:
        drives = drives_obj["previous"]
    else:
        raise KeyError(
            "Could not find 'all' or 'previous' drives in gamepackageJSON['drives']"
        )

    plays: list[dict] = []
    for d in drives:
        for p in d.get("plays", []):
            plays.append(p)
    return plays


def _get_play_description(play: dict) -> str:
    """
    ESPN sticks text in slightly different fields depending on context.
    This normalizes to one description string.
    """
    return (
        play.get("text")
        or play.get("description")
        or play.get("shortDescription")
        or ""
    ).strip()


def is_timeout_play(play: dict) -> bool:
    """
    Returns True only for *team-charged* timeouts.
    Official timeouts return False and do not affect timeout counts.
    """
    desc = _get_play_description(play)

    # Official timeout? ignore
    if OFFICIAL_TIMEOUT_REGEX.search(desc):
        return False

    # Team-charged timeout?
    if TEAM_TIMEOUT_REGEX.search(desc):
        return True

    return False


def get_timeout_team_abbrev(play: dict) -> str | None:
    """
    Parse the team abbreviation from a team-charged timeout description.

    Example: "Timeout #2 by NYJ at 01:50." -> "NYJ"
    """
    desc = _get_play_description(play)
    m = TEAM_TIMEOUT_REGEX.search(desc)
    if not m:
        return None
    return m.group(1).upper()


def add_timeouts_remaining(
    plays: list[dict],
    home_id: str,
    away_id: str,
    abbrev_to_team_id: dict[str, str],
    timeouts_per_half: int = 3,
) -> list[dict]:
    """
    Walk plays in sequenceNumber order and reconstruct 'timeouts remaining'
    for each team at every play.

    - Resets timeouts at the start of the 2nd half (period >= 3).
    - Applies timeout AFTER recording the snapshot (so counts reflect state
      at play start).
    """
    timeouts: dict[str, int] = {
        home_id: timeouts_per_half,
        away_id: timeouts_per_half,
    }

    current_period = 1
    enriched: list[dict] = []

    # Ensure numeric order; sequenceNumber is a string in ESPN JSON
    sorted_plays = sorted(
        plays,
        key=lambda x: float(x.get("sequenceNumber", 0)),
    )

    for play in sorted_plays:
        period = play.get("period", {}).get("number", current_period)

        # Reset timeouts at start of 2nd half (Q3)
        if period >= 3 and current_period < 3:
            timeouts[home_id] = timeouts_per_half
            timeouts[away_id] = timeouts_per_half

        current_period = period

        # Snapshot BEFORE applying this play's timeout effect
        play_with_timeouts = {
            **play,
            "home_timeouts_remaining": timeouts[home_id],
            "away_timeouts_remaining": timeouts[away_id],
        }
        enriched.append(play_with_timeouts)

        # Now, if this play is a *team timeout*, decrement appropriate team
        if is_timeout_play(play):
            # First choice: parse abbrev from description ("Timeout #2 by NYJ...")
            abbrev = get_timeout_team_abbrev(play)
            tid: str | None = None

            if abbrev and abbrev in abbrev_to_team_id:
                tid = abbrev_to_team_id[abbrev]
            else:
                # Fallback: use start.team.id if abbrev mapping fails
                start_team = (play.get("start") or {}).get("team") or {}
                tid = start_team.get("id")

            if tid in timeouts:
                timeouts[tid] = max(0, timeouts[tid] - 1)

    return enriched


def espn_game_to_df_with_timeouts(game_id: str) -> pd.DataFrame:
    """
    High-level helper:
    - fetch JSON
    - extract plays + team IDs/meta
    - enrich with timeouts remaining
    - return a pandas DataFrame (one row per play)
    """
    game_json = fetch_espn_game(game_id)
    home_id, away_id, abbrev_to_team_id = get_team_meta(game_json)
    plays = get_all_plays(game_json)
    plays_enriched = add_timeouts_remaining(
        plays,
        home_id,
        away_id,
        abbrev_to_team_id,
    )

    # Flatten nested dicts into columns
    df = pd.json_normalize(plays_enriched)

    # (Optional) clean up some common useful columns
    rename_map = {
        "period.number": "qtr",
        "clock.displayValue": "clock",
        "start.down": "down",
        "start.distance": "ydstogo",
        "start.yardLine": "yardline_100_like",
        "homeScore": "total_home_score",
        "awayScore": "total_away_score",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    return df

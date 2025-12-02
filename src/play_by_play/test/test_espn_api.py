"""
Pytest tests for ESPN API utility functions.

Unit tests use mocked API responses.
Integration tests (marked with @pytest.mark.integration) hit the real ESPN API.
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch

from play_by_play.utils.espn_api_util import (
    fetch_espn_game,
    get_team_meta,
    get_all_plays,
    is_timeout_play,
    get_timeout_team_abbrev,
    add_timeouts_remaining,
    espn_game_to_df_with_timeouts,
)


# Required columns for feature engineering
REQUIRED_COLS = [
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


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def mock_espn_game_json():
    """Mock ESPN game JSON response with realistic structure."""
    return {
        "gamepackageJSON": {
            "header": {
                "competitions": [
                    {
                        "competitors": [
                            {
                                "homeAway": "home",
                                "team": {
                                    "id": "16",
                                    "abbreviation": "KC",
                                },
                            },
                            {
                                "homeAway": "away",
                                "team": {
                                    "id": "22",
                                    "abbreviation": "BUF",
                                },
                            },
                        ]
                    }
                ]
            },
            "drives": {
                "previous": [
                    {
                        "plays": [
                            {
                                "id": "4011234567890",
                                "sequenceNumber": "100",
                                "period": {"number": 1},
                                "clock": {"displayValue": "15:00"},
                                "homeScore": 0,
                                "awayScore": 0,
                                "start": {
                                    "team": {"id": "16"},
                                    "down": 1,
                                    "distance": 10,
                                    "yardLine": 75,
                                },
                                "text": "Patrick Mahomes pass to Travis Kelce for 5 yards",
                            },
                            {
                                "id": "4011234567891",
                                "sequenceNumber": "200",
                                "period": {"number": 1},
                                "clock": {"displayValue": "14:30"},
                                "homeScore": 0,
                                "awayScore": 0,
                                "start": {
                                    "team": {"id": "16"},
                                    "down": 2,
                                    "distance": 5,
                                    "yardLine": 70,
                                },
                                "text": "Patrick Mahomes rush for 3 yards",
                            },
                            {
                                "id": "4011234567892",
                                "sequenceNumber": "300",
                                "period": {"number": 1},
                                "clock": {"displayValue": "14:00"},
                                "homeScore": 0,
                                "awayScore": 0,
                                "start": {
                                    "team": {"id": "22"},
                                    "down": 1,
                                    "distance": 10,
                                    "yardLine": 75,
                                },
                                "text": "Timeout #1 by BUF at 14:00.",
                            },
                        ]
                    }
                ]
            },
        }
    }


@pytest.fixture
def sample_plays():
    """Sample plays for testing timeout logic."""
    return [
        {
            "id": "play1",
            "sequenceNumber": "100",
            "period": {"number": 1},
            "start": {"team": {"id": "16"}},
            "text": "First play",
        },
        {
            "id": "play2",
            "sequenceNumber": "200",
            "period": {"number": 1},
            "start": {"team": {"id": "22"}},
            "text": "Timeout #1 by BUF at 14:00.",
        },
        {
            "id": "play3",
            "sequenceNumber": "300",
            "period": {"number": 3},
            "start": {"team": {"id": "16"}},
            "text": "Third quarter play",
        },
    ]


# ============================================================================
# UNIT TESTS (MOCKED)
# ============================================================================


def test_get_team_meta(mock_espn_game_json):
    """Test extraction of team metadata from ESPN JSON."""
    home_id, away_id, abbrev_map = get_team_meta(mock_espn_game_json)

    assert home_id == "16"
    assert away_id == "22"
    assert abbrev_map["KC"] == "16"
    assert abbrev_map["BUF"] == "22"


def test_get_all_plays(mock_espn_game_json):
    """Test extraction of all plays from drives structure."""
    plays = get_all_plays(mock_espn_game_json)

    assert len(plays) == 3
    assert plays[0]["id"] == "4011234567890"
    assert plays[1]["id"] == "4011234567891"
    assert plays[2]["id"] == "4011234567892"


def test_is_timeout_play():
    """Test timeout detection logic."""
    # Team timeout
    timeout_play = {"text": "Timeout #1 by BUF at 14:00."}
    assert is_timeout_play(timeout_play) is True

    # Official timeout (should not count)
    official_timeout = {"text": "Official Timeout at 06:36."}
    assert is_timeout_play(official_timeout) is False

    # Regular play
    regular_play = {"text": "Patrick Mahomes pass complete"}
    assert is_timeout_play(regular_play) is False


def test_get_timeout_team_abbrev():
    """Test parsing team abbreviation from timeout description."""
    play = {"text": "Timeout #2 by NYJ at 01:50."}
    assert get_timeout_team_abbrev(play) == "NYJ"

    play = {"text": "Timeout #1 by KC at 05:00."}
    assert get_timeout_team_abbrev(play) == "KC"

    # Non-timeout play
    play = {"text": "Regular play"}
    assert get_timeout_team_abbrev(play) is None


def test_add_timeouts_remaining(sample_plays):
    """Test timeout tracking logic."""
    home_id = "16"
    away_id = "22"
    abbrev_map = {"KC": "16", "BUF": "22"}

    enriched = add_timeouts_remaining(sample_plays, home_id, away_id, abbrev_map)

    # First play: both teams should have 3 timeouts
    assert enriched[0]["home_timeouts_remaining"] == 3
    assert enriched[0]["away_timeouts_remaining"] == 3

    # Second play (before timeout applied): should still be 3 each
    assert enriched[1]["home_timeouts_remaining"] == 3
    assert enriched[1]["away_timeouts_remaining"] == 3

    # Third play (after BUF timeout, and Q3 reset): should reset to 3
    assert enriched[2]["home_timeouts_remaining"] == 3
    assert enriched[2]["away_timeouts_remaining"] == 3


@patch("play_by_play.utils.espn_api_util.fetch_espn_game")
def test_espn_game_to_df_with_timeouts_mocked(mock_fetch, mock_espn_game_json):
    """Test DataFrame creation with mocked API response."""
    mock_fetch.return_value = mock_espn_game_json

    df = espn_game_to_df_with_timeouts("401772945")

    # Check that all required columns are present
    for col in REQUIRED_COLS:
        assert col in df.columns, f"Missing required column: {col}"

    # Check row count
    assert len(df) == 3

    # Check game_id
    assert (df["game_id"] == "401772945").all()

    # Check home_team
    assert (df["home_team"] == "KC").all()

    # Check location logic
    # First two plays by KC (home) should be "Home"
    assert df.loc[0, "location"] == "Home"
    assert df.loc[1, "location"] == "Home"
    # Third play by BUF (away) should be "Away"
    assert df.loc[2, "location"] == "Away"

    # Check timeout columns exist
    assert "posteam_timeouts_remaining" in df.columns
    assert "defteam_timeouts_remaining" in df.columns


# ============================================================================
# INTEGRATION TESTS (REAL API)
# ============================================================================


@pytest.mark.integration
@pytest.mark.slow
def test_real_espn_api_integration():
    """Integration test hitting the real ESPN API."""
    game_id = "401772945"

    df = espn_game_to_df_with_timeouts(game_id)

    # Check all required columns are present
    missing_cols = [col for col in REQUIRED_COLS if col not in df.columns]
    assert not missing_cols, f"Missing columns: {missing_cols}"

    # Check we got some plays
    assert len(df) > 0, "No plays returned from API"

    # Check data types
    assert df["qtr"].dtype == "int64"
    assert df["total_home_score"].dtype == "int64"
    assert df["total_away_score"].dtype == "int64"
    assert df["down"].dtype == "int64"
    assert df["ydstogo"].dtype == "int64"
    assert df["yardline_100"].dtype == "int64"

    # Check game_id is correct
    assert (df["game_id"] == game_id).all()

    # Check timeout columns are numeric
    assert df["posteam_timeouts_remaining"].dtype == "int64"
    assert df["defteam_timeouts_remaining"].dtype == "int64"

    # Check timeout values are in valid range [0, 3]
    assert df["posteam_timeouts_remaining"].between(0, 3).all()
    assert df["defteam_timeouts_remaining"].between(0, 3).all()

    # Check location values are valid
    assert df["location"].isin(["Home", "Away"]).all()

    # Print summary
    print(f"\n✅ Integration test passed!")
    print(f"   - Plays extracted: {len(df)}")
    print(f"   - Home team: {df['home_team'].iloc[0]}")
    print(f"   - All {len(REQUIRED_COLS)} required columns present")


@pytest.mark.integration
@pytest.mark.slow
def test_real_api_timeout_logic():
    """Integration test specifically for timeout tracking."""
    game_id = "401772945"

    df = espn_game_to_df_with_timeouts(game_id)

    # Timeouts should never be negative
    assert (df["posteam_timeouts_remaining"] >= 0).all()
    assert (df["defteam_timeouts_remaining"] >= 0).all()

    # Timeouts should never exceed 3
    assert (df["posteam_timeouts_remaining"] <= 3).all()
    assert (df["defteam_timeouts_remaining"] <= 3).all()

    # First play of game should have 3 timeouts each
    first_play = df.iloc[0]
    assert first_play["posteam_timeouts_remaining"] == 3
    assert first_play["defteam_timeouts_remaining"] == 3

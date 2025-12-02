# Testing Guide

This directory contains pytest tests for the NFL play-by-play project.

## Test Types

### Unit Tests (Mocked)
- **Fast** - Run in ~0.2 seconds
- **Reliable** - No external dependencies
- **Frequent** - Run on every code change
- Uses mocked ESPN API responses

### Integration Tests (Real API)
- **Slow** - Run in ~1 second
- **Real data** - Hits actual ESPN API
- **Occasional** - Run before commits/deploys
- Marked with `@pytest.mark.integration`

## Running Tests

### All tests
```bash
pytest
```

### Unit tests only (fast, no API calls)
```bash
pytest -m "not integration"
```

### Integration tests only (hits real API)
```bash
pytest -m "integration"
```

### Run tests with verbose output
```bash
pytest -v
```

### Run specific test file
```bash
pytest src/tests/test_espn_api.py
```

### Run specific test function
```bash
pytest src/tests/test_espn_api.py::test_get_team_meta
```

### See print statements during tests
```bash
pytest -s
```

## Test Coverage

Current tests cover:
- ✅ Team metadata extraction (`get_team_meta`)
- ✅ Play extraction from drives (`get_all_plays`)
- ✅ Timeout detection logic (`is_timeout_play`)
- ✅ Timeout team parsing (`get_timeout_team_abbrev`)
- ✅ Timeout tracking across plays (`add_timeouts_remaining`)
- ✅ Full DataFrame creation with all required columns (`espn_game_to_df_with_timeouts`)
- ✅ Real API integration validation
- ✅ Real API timeout logic validation

## Required Columns

Tests validate that all feature engineering columns are present:
- `game_id`, `play_id`, `qtr`, `time`
- `total_home_score`, `total_away_score`
- `home_team`, `posteam`
- `down`, `ydstogo`, `yardline_100`
- `posteam_timeouts_remaining`, `defteam_timeouts_remaining`
- `location`

## CI/CD Usage

For continuous integration, run unit tests on every commit:
```bash
pytest -m "not integration" --tb=short
```

Run integration tests less frequently (e.g., before deployment):
```bash
pytest -m "integration"
```

## Adding New Tests

1. Create test functions in `test_espn_api.py` (or new test files)
2. Use `@pytest.mark.integration` for tests that hit external APIs
3. Use fixtures from `conftest.py` for shared test data
4. Mock external calls for unit tests using `@patch` decorator

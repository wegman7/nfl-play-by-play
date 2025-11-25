---
description: Run ESPN API tests (unit tests only, no integration)
---

Run pytest on the ESPN API test suite, excluding integration tests that hit the real ESPN API.

This runs only the fast unit tests with mocked data.

To run with additional pytest options, add them after the command (e.g., `/test-espn -v` for verbose).

To include integration tests, use: `pytest src/tests/test_espn_api.py`

"""Pytest configuration and fixtures for test suite."""

import sys
from pathlib import Path

# Add project root to Python path for imports
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

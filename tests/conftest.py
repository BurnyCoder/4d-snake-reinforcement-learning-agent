"""Shared pytest fixtures: keep SNAKE_* environment overrides from leaking between tests."""

import os

import pytest


@pytest.fixture(autouse=True)
def _clean_snake_env(monkeypatch):
    """Config.from_env writes SNAKE_* keys into os.environ; isolate every test from the others."""
    for key in [k for k in os.environ if k.startswith("SNAKE_")]:
        monkeypatch.delenv(key, raising=False)

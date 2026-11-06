"""Shared test helpers."""

import re

import pytest

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    """Remove ANSI styling so assertions match what a reader sees.

    Typer/Rich colorizes only when it detects a capable terminal: GitHub Actions
    has one, a bare local shell often does not. Rich also styles an option's
    leading dash separately —
    ``\\x1b[1;36m-\\x1b[0m\\x1b[1;36m-verbose\\x1b[0m`` — and styles numbers
    inside tables, so a literal ``"--version" in result.stdout`` or
    ``"60 predictions" in result.stdout`` can pass locally and fail in CI.

    Negative assertions are the worse case: ``"Usage:" not in result.stdout``
    passes vacuously against a colorized stream.

    Note the full local matrix (3.11-3.14) does not catch this — every run shares
    one terminal. Check with ``FORCE_COLOR=1 uv run pytest`` instead.
    """
    return _ANSI.sub("", text)


@pytest.fixture
def plain():
    """Return :func:`strip_ansi`, for asserting on rendered CLI output."""
    return strip_ansi


@pytest.fixture
def warnings_seen():
    """WARNING messages logged in this process while the test runs.

    loguru handlers do not reach spawned workers, so this sees only what the
    parent process logs.
    """
    from loguru import logger

    seen: list[str] = []
    handler = logger.add(lambda m: seen.append(m.record["message"]), level="WARNING")
    yield seen
    logger.remove(handler)

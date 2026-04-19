"""
tests/unit/test_tooling.py

Meta-tests that verify the project's tooling is correctly configured.
These test the development environment itself — not business logic.

Why this matters:
- Catches configuration drift (someone edits pyproject.toml incorrectly)
- Documents expected tool behavior
- Runs in CI so broken tooling is caught immediately
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.resolve()


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    """Run a subprocess command from the project root."""
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )


# ── Black ─────────────────────────────────────────────────────────────────────


def test_black_is_installed() -> None:
    """Black must be installed as a dev dependency."""
    result = run([sys.executable, "-m", "black", "--version"])
    assert result.returncode == 0
    assert "black" in result.stdout.lower()


def test_black_reports_clean_codebase() -> None:
    """
    All source files must be Black-formatted.
    If this fails: run 'python -m black src/ tests/' to fix.
    """
    result = run([sys.executable, "-m", "black", "--check", "src/", "tests/"])
    assert (
        result.returncode == 0
    ), f"Black found unformatted files:\n{result.stdout}\n{result.stderr}"


# ── Ruff ──────────────────────────────────────────────────────────────────────


def test_ruff_is_installed() -> None:
    """Ruff must be installed as a dev dependency."""
    result = run([sys.executable, "-m", "ruff", "--version"])
    assert result.returncode == 0
    assert "ruff" in result.stdout.lower()


def test_ruff_reports_clean_codebase() -> None:
    """
    All source files must pass Ruff linting.
    If this fails: run 'python -m ruff check src/ tests/ --fix' to fix.
    """
    result = run([sys.executable, "-m", "ruff", "check", "src/", "tests/"])
    assert (
        result.returncode == 0
    ), f"Ruff found issues:\n{result.stdout}\n{result.stderr}"


# ── Project structure ─────────────────────────────────────────────────────────


def test_pyproject_toml_exists() -> None:
    """pyproject.toml must exist at the project root."""
    assert (ROOT / "pyproject.toml").exists()


def test_required_source_modules_exist() -> None:
    """
    All planned source modules must exist.
    Fails immediately if someone accidentally deletes a folder.
    """
    required = [
        "src/tracker/capture/__init__.py",
        "src/tracker/detection/__init__.py",
        "src/tracker/tracking/__init__.py",
        "src/tracker/visualization/__init__.py",
        "src/tracker/processing/__init__.py",
        "src/tracker/config.py",
        "src/tracker/logger.py",
    ]
    for path in required:
        assert (ROOT / path).exists(), f"Required module missing: {path}"


def test_default_config_exists() -> None:
    """configs/default.yaml must exist — the app cannot start without it."""
    assert (
        ROOT / "configs" / "default.yaml"
    ).exists(), "configs/default.yaml is missing — application cannot start"


def test_pre_commit_config_exists() -> None:
    """.pre-commit-config.yaml must exist to enforce code quality."""
    assert (ROOT / ".pre-commit-config.yaml").exists()

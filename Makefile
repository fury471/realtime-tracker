# Realtime Tracker - developer shortcuts
# Windows: python -m make <target>  (if make is not installed)
# Linux/Mac/WSL: make <target>

.PHONY: install test test-fast lint format check clean

install:
	pip install -e ".[dev,api]"

test:
	python -m pytest tests/ -v

test-fast:
	python -m pytest tests/ -q --no-header

lint:
	python -m ruff check src/ tests/
	python -m black --check src/ tests/

format:
	python -m black src/ tests/
	python -m ruff check src/ tests/ --fix

check: format lint test
	@echo "All checks passed."

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

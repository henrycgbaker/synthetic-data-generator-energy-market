.PHONY: help install test test-all test-unit test-integration test-functional test-smoke test-slow test-coverage clean lint format pre-commit-install pre-commit-run

help:
	@echo "Available commands:"
	@echo "  make install              - Install package and dev dependencies"
	@echo "  make test                 - Run all tests (excluding slow)"
	@echo "  make test-all             - Run ALL tests including slow"
	@echo "  make test-unit            - Run unit tests only"
	@echo "  make test-integration     - Run integration tests only"
	@echo "  make test-functional      - Run functional tests only"
	@echo "  make test-smoke           - Run smoke tests (quick validation)"
	@echo "  make test-slow            - Run slow tests only"
	@echo "  make test-coverage        - Run tests with coverage report"
	@echo "  make lint                 - Run linting checks"
	@echo "  make format               - Format code with black and isort"
	@echo "  make pre-commit-install   - Install pre-commit hooks"
	@echo "  make pre-commit-run       - Run pre-commit on all files"
	@echo "  make clean                - Clean test artifacts"

install:
	pip install -e ".[dev]"

test:
	python -m pytest -m "not slow"

test-all:
	python -m pytest

test-unit:
	python -m pytest -m unit

test-integration:
	python -m pytest -m integration

test-functional:
	python -m pytest -m functional

test-smoke:
	python -m pytest -m smoke

test-slow:
	python -m pytest -m slow

test-coverage:
	python -m pytest --cov=synthetic_data_pkg --cov-report=html --cov-report=term

lint:
	ruff check synthetic_data_pkg/ tests/
	mypy synthetic_data_pkg/

format:
	black synthetic_data_pkg/ tests/
	isort synthetic_data_pkg/ tests/

pre-commit-install:
	pre-commit install

pre-commit-run:
	pre-commit run --all-files

clean:
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

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
	poetry install

test:
	poetry run pytest -m "not slow"

test-all:
	poetry run pytest

test-unit:
	poetry run pytest -m unit

test-integration:
	poetry run pytest -m integration

test-functional:
	poetry run pytest -m functional

test-smoke:
	poetry run pytest -m smoke

test-slow:
	poetry run pytest -m slow

test-coverage:
	poetry run pytest --cov=synthetic_data_pkg --cov-report=html --cov-report=term

lint:
	poetry run ruff check synthetic_data_pkg/ tests/
	poetry run mypy synthetic_data_pkg/

format:
	poetry run black synthetic_data_pkg/ tests/
	poetry run isort synthetic_data_pkg/ tests/

pre-commit-install:
	poetry run pre-commit install

pre-commit-run:
	poetry run pre-commit run --all-files

clean:
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

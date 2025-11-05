# Testing Guide

This document describes the testing infrastructure for the synthetic data generator.

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── unit/                    # Unit tests (isolated component testing)
│   ├── test_demand.py       # Tests for demand module
│   └── test_dists.py        # Tests for distributions module
├── integration/             # Integration tests (multiple components)
│   └── test_scenarios.py    # Tests for scenario execution
└── functional/              # Functional tests (end-to-end workflows)
    └── test_workflows.py    # Tests for complete user workflows
```

## Test Types

### Unit Tests (@pytest.mark.unit)
- Test individual functions and classes in isolation
- Fast, focused, and independent
- Mock external dependencies
- Example: Testing demand curve calculations

### Integration Tests (@pytest.mark.integration)
- Test multiple components working together
- Validate interfaces between modules
- Check data flow through the system
- Example: Running a complete scenario

### Functional Tests (@pytest.mark.functional)
- Test complete user workflows end-to-end
- Validate CLI, file I/O, and runner scripts
- Slow but comprehensive
- Example: Running scenario scripts and validating outputs

### Special Markers
- `@pytest.mark.smoke`: Quick validation tests (run on every commit)
- `@pytest.mark.slow`: Long-running tests (run less frequently)

## Running Tests

### Install test dependencies
```bash
pip install -e ".[dev]"
```

### Run all tests
```bash
make test
# or
pytest
```

### Run specific test types
```bash
make test-unit           # Unit tests only
make test-integration    # Integration tests only
make test-functional     # Functional tests only
make test-smoke          # Quick smoke tests
make test-slow           # Long-running tests
```

### Run tests with coverage
```bash
make test-coverage
# Opens htmlcov/index.html for detailed report
```

### Run specific test files
```bash
pytest tests/unit/test_demand.py
pytest tests/integration/test_scenarios.py::TestScenarioExecution
pytest tests/unit/test_demand.py::TestDemandCurve::test_daily_seasonality_flag_on
```

### Run tests in parallel (faster)
```bash
pytest -n auto
```

## Pre-commit Hooks

Pre-commit hooks automatically run checks before each commit.

### Install hooks
```bash
make pre-commit-install
# or
pre-commit install
```

### What runs on commit:
- Code formatting (black, isort)
- Linting (ruff)
- Type checking (mypy)
- Unit tests (fast validation)

### What runs on push:
- Smoke tests (quick integration checks)

### Run hooks manually
```bash
make pre-commit-run
# or
pre-commit run --all-files
```

### Skip hooks temporarily (use sparingly!)
```bash
git commit --no-verify
```

## Code Quality Tools

### Formatting
```bash
make format
# Formats code with black and isort
```

### Linting
```bash
make lint
# Runs ruff (linter) and mypy (type checker)
```

## Writing Tests

### Example unit test
```python
import pytest
from synthetic_data_pkg.demand import DemandCurve

@pytest.mark.unit
def test_demand_curve_initialization():
    """Test DemandCurve initializes correctly"""
    cfg = DemandConfig()
    demand = DemandCurve(cfg)
    assert demand.cfg == cfg
```

### Example integration test
```python
import pytest

@pytest.mark.integration
def test_scenario_runs_successfully(minimal_config):
    """Test that a minimal scenario runs without errors"""
    executor = ScenarioExecutor(minimal_config)
    df = executor.run()
    assert len(df) > 0
```

### Using fixtures
```python
def test_with_temp_directory(temp_output_dir):
    """Test uses temporary directory fixture"""
    output_path = temp_output_dir / "output.csv"
    # Temp directory is automatically cleaned up
```

## Coverage Goals

- **Minimum coverage**: 80% (enforced by pytest)
- **Target coverage**: 90%+
- Check coverage report: `make test-coverage` → open `htmlcov/index.html`

## Continuous Integration

Tests run automatically on:
- Every commit (via pre-commit hooks): unit tests + formatting
- Every push (via pre-commit hooks): smoke tests
- Pull requests: full test suite

## Troubleshooting

### Tests are slow
- Run only fast tests: `pytest -m "not slow"`
- Run in parallel: `pytest -n auto`
- Run specific markers: `make test-smoke`

### Coverage too low
- Check which lines are missing: `make test-coverage`
- View HTML report: `htmlcov/index.html`
- Add tests for uncovered code

### Pre-commit hooks failing
- Fix issues manually: `make format && make lint`
- Check specific errors: `pre-commit run --all-files`
- Update hooks: `pre-commit autoupdate`

### Import errors in tests
- Ensure package is installed: `pip install -e .`
- Check Python path is correct
- Run from project root directory

## Best Practices

1. **Write tests first** (TDD): Define expected behavior before implementation
2. **Keep tests isolated**: Each test should be independent
3. **Use fixtures**: Share common setup code via fixtures
4. **Test edge cases**: Zero, negative, extreme values
5. **Clear test names**: Describe what is being tested
6. **One assertion focus**: Test one thing at a time
7. **Fast unit tests**: Unit tests should complete in milliseconds
8. **Mock external dependencies**: Don't rely on external services
9. **Parametrize similar tests**: Use `@pytest.mark.parametrize` for variations
10. **Document complex tests**: Add docstrings explaining non-obvious tests

## Example Test Development Workflow

1. Write a failing test that describes desired behavior
2. Run the test: `pytest tests/unit/test_demand.py -k test_new_feature`
3. Implement the feature
4. Run the test again (should pass)
5. Run full test suite: `make test`
6. Check coverage: `make test-coverage`
7. Commit (pre-commit hooks run automatically)

## Additional Resources

- [pytest documentation](https://docs.pytest.org/)
- [pytest markers](https://docs.pytest.org/en/stable/how-to/mark.html)
- [pytest fixtures](https://docs.pytest.org/en/stable/how-to/fixtures.html)
- [pre-commit](https://pre-commit.com/)

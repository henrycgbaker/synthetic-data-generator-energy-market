## Quick orientation for AI coding assistants

This repository generates synthetic electricity market time series using YAML scenarios and a Pydantic-validated config → simulate → output pipeline.

Keep changes small and focused. Key facts to help you be productive:

- Entry point / CLI: `synth-data` is installed via `pyproject.toml` (`synthetic_data_pkg.cli:entrypoint`). Typical run: `synth-data generate configs/1_gas_crisis.yaml`.
- Main orchestrator: `synthetic_data_pkg/runner.py` (see `execute_scenario()`), which loads configs, builds regime schedules, runs `simulate_timeseries()` and saves outputs.
- Core loop & solver: `synthetic_data_pkg/simulate.py` — draws regime values hour-by-hour, builds supply/demand curves and calls `find_equilibrium()`.
- Configs: `configs/` contains YAML scenarios. The loader supports `extends` to deep-merge `_base_template.yaml`.

Project-specific conventions and checks (do not change without tests):

- Required variables: `fuel.gas` and `fuel.coal` must always be present in a config's `variables`.
- Regime planner `supply_regime_planner.mode` accepts `local_only`, `global`, `hybrid`. Behaviour and validation differ (see `CLAUDE.md` and README config guide).
- Renewable modelling: top-level `renewable_availability_mode` is either `weather_simulation` (AR(1) wind, sinusoidal solar) or `direct` (requires `avail.wind`/`avail.solar`).
- Planned outages only affect `nuclear`, `coal`, `gas` (not wind/solar).
- Distribution kinds: `const`, `normal`, `beta`, `uniform`, `linear`, `empirical` — follow README examples for parameters and bounds.

Developer workflows and commands

- Install: `poetry install`. Note: repository supports working inside a conda env (see `POETRY_SETUP.md`) — many devs use `conda activate synth_data` and then run commands directly.
- Make targets (wrap `poetry run`):
  - `make test` (fast tests, excludes `slow`), `make test-all` (all tests)
  - `make test-unit`, `make test-integration`, `make test-functional`
  - `make format`, `make lint`
- Pytest: tests are under `tests/unit`, `tests/functional`, `tests/integration`. Use `tests/conftest.py` for fixtures (notably `minimal_config`).

Editing guidance and examples to reference in PRs

- When changing config shapes, update Pydantic models in `synthetic_data_pkg/config.py` and corresponding fixtures in `tests/conftest.py`.
- When modifying regime logic, update `synthetic_data_pkg/regimes.py` and ensure `build_schedules()` behaviours for `local_only`/`global`/`hybrid` remain consistent.
- For changes to equilibrium solving or numeric methods, run integration tests (they check solver edge cases). Numerical stability is important — prefer small, well-tested changes.

Files to inspect for non-obvious behavior

- `synthetic_data_pkg/runner.py` — orchestration, logging
- `synthetic_data_pkg/cli.py` — CLI entrypoint mapping
- `synthetic_data_pkg/simulate.py` — core loop & equilibrium solver
- `synthetic_data_pkg/supply.py` and `synthetic_data_pkg/demand.py` — supply curve and demand curve construction
- `synthetic_data_pkg/config.py` — Pydantic config models and validation rules
- `tests/conftest.py` — canonical fixtures like `minimal_config`

Safety rules for edits

- Preserve existing configuration semantics (required variables, planner modes, renewable mode). If you must change semantics, update README.md and tests.
- Avoid introducing print()s — use the `logging` module (project moved to logging in recent commits).
- Keep outputs deterministic where tests expect it: respect `seed` / RNG handling used in tests.

If uncertain, ask for the preferred approach and which tests to update. After edits, run `make test-unit` and `make test-integration` locally and attach failing tracebacks if any.

Requested follow-up

Please review this draft: tell me if you'd like more examples (config snippets, common PR checklist) or stricter rules about refactors vs. bugfixes.

# Helia-Core Tester

Toolkit for CMSIS-NN testing: generate TFLite models, convert to C, build for FVP, and run the tests.

## Features

- End-to-end pipeline: generate → convert → build → run
- TFLite model generation via pytest
- AOT conversion to C via helia-aot
- FVP Corstone-300 build and execution
- Reports for test results

## Getting Started

### Prerequisites

You need `uv` for Python dependencies:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

You also need `git` for submodules (uv does not handle submodules).

### Setup

1. Initialize submodules:
```bash
git submodule update --init --recursive --depth 1
```

2. Install Python dependencies:
```bash
uv sync
```

3. Run the tool:
```bash
uv run helia_core_tester --help
```

### CI Setup

Use the CI helper script:
```bash
./scripts/setup_ci.sh
```

Options:
```bash
./scripts/setup_ci.sh --downloads-dir /custom/path
./scripts/setup_ci.sh --skip-build-deps
```

### Using uv for Dependency Management

Standard workflow:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git submodule update --init --recursive --depth 1
uv sync
uv run helia_core_tester --help
```

Note: submodules must be handled by git.

## Usage

### Basic Usage

```bash
uv run helia_core_tester full
uv run helia_core_tester full --cpu cortex-m4
```

Note: `build` requires `artifacts/generated_tests/tests.cmake`, which is created by `generate`.

### Subcommands

- `uv run helia_core_tester generate` — Generate TFLite models and C/H templates
- `uv run helia_core_tester build` — Build for FVP
- `uv run helia_core_tester run` — Run on FVP
- `uv run helia_core_tester full` — Full pipeline
- `uv run helia_core_tester clean` — Remove build artifacts
- `uv run helia_core_tester clean-all` — Remove all artifacts
- `uv run helia_core_tester doctor` — Preflight checks

### Advanced Usage

```bash
uv run helia_core_tester full --op conv2D --dtype S8 --limit 5
uv run helia_core_tester full --skip-generation --skip-conversion
uv run helia_core_tester full --dry-run
uv run helia_core_tester full --cpu cortex-m3 -v 2
uv run helia_core_tester build --opt "-O2" --jobs 8
```

### Command Line Options

Pipeline control:
- `--skip-generation`
- `--skip-conversion`
- `--skip-build`
- `--skip-run`

Generation filters:
- `--op OPERATOR`
- `--dtype DTYPE`
- `--limit N`

Build options:
- `--cpu CPU`
- `--opt LEVEL`
- `--jobs N`

Run options:
- `--timeout SECONDS`
- `--no-fail-fast`
- `--report-formats`

General options:
- `--verbosity, -v`
- `--dry-run`
- `--quiet, -q`
- `--log-file PATH`
- `--plan`

## Configuration

Defaults live in `helia_core_tester.toml` at the repo root. CLI flags override these values.

## Architecture

Python source: `helia_core_tester/`.

Core modules:
- `helia_core_tester.core.pipeline`: Pipeline orchestration
- `helia_core_tester.core.config`: Configuration and repo-root discovery
- `helia_core_tester.core.discovery`: Path discovery
- `helia_core_tester.core.steps`: Pipeline steps

CLI:
- `helia_core_tester.cli`: CLI entry point

FVP:
- `helia_core_tester.fvp.build_and_run_fvp`: Build and run

Scripts:
- `helia_core_tester/scripts/setup_dependencies.py`: Build dependency download

Generation:
- `helia_core_tester/generation/`: TFLite generation via pytest
- `helia_core_tester/generation/ops/`: Operators
- `helia_core_tester/generation/io/`: I/O utilities
- `helia_core_tester/generation/assets/descriptors/`: Descriptor schemas/examples
- `artifacts/generated_tests/manifest.json`: Generated test manifest
- `artifacts/generated_tests/tests.cmake`: CMake test list

Utilities and reporting:
- `helia_core_tester/utils/`: Command helpers
- `helia_core_tester/reporting/`: Result parsing and report generation

## Development

Generate TFLite models directly:
```bash
uv run pytest helia_core_tester/generation/test_ops.py::test_generation -v --op mean_int16
```

Code quality (optional):
```bash
black helia_core_tester/
flake8 helia_core_tester/
mypy helia_core_tester/
```

## Requirements

- `uv` (required)
- Python 3.8+
- pytest
- TensorFlow Lite
- helia-aot
- CMake
- ARM GCC toolchain
- FVP Corstone-300

## CI Reporting

See `CI_REPORTING.md`.

## CI/CD Integration

GitHub Actions in `.github/workflows/ci.yml`:
- checks out submodules
- runs `./scripts/setup_ci.sh`
- verifies dependencies and build tools
- tests the CLI

For other CI systems:
```bash
./scripts/setup_ci.sh
```

# Helia-Core Tester

A comprehensive toolkit for CMSIS-NN testing, model generation, and FVP simulation.

## Features

- **Complete Test Pipeline**: End-to-end workflow from model generation to FVP execution
- **TFLite Model Generation**: Generate TensorFlow Lite models using pytest
- **AOT Conversion**: Convert TFLite models to C modules using helia-aot
- **Test Runner Generation**: Automatically generate Unity test runners
- **FVP Building**: Build test executables for FVP Corstone-300 simulator
- **FVP Execution**: Run tests on FVP simulator with comprehensive reporting

## Getting Started

### Prerequisites

**uv is required** for Python dependency management. Install it first:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**git is required** for cloning submodules (uv cannot handle git submodules). Note: The Include and Source directories are accessed directly from the parent repository, not via submodule.

You can either manually set up dependencies or use the automated setup:

**Option 1: Automated setup**
```bash
# From the repository root (Tests/CodeGen) - uses uv sync
uv sync
```
1. **Initialize submodules** (uv cannot do this, git is required):
   ```bash
   git submodule update --init --recursive --depth 1
   ```

2. **Install Python dependencies with uv sync**:
   ```bash
   uv sync
   ```

**Run the tool**:
```bash
# Using uv run (recommended)
uv run cmsis-nn-tools --help

# Or activate the virtual environment first
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
cmsis-nn-tools --help
```

### CI Setup

For CI environments, use the `setup_ci.sh` script which:
- Requires `uv` to be installed (does not install it)
- Runs `uv sync` to set up the Python environment (`.venv` at repo root)
- Optionally runs build dependency setup (ARM GCC, CMSIS-5, Corstone300 FVP, etc.)
- Configures environment variables

```bash
# In CI: ensure uv is available, then run
./scripts/setup_ci.sh

# With custom downloads directory
./scripts/setup_ci.sh --downloads-dir /custom/path

# Skip certain steps if needed
./scripts/setup_ci.sh --skip-build-deps
```

The script is idempotent and safe to run multiple times. It will skip steps that are already completed.

### Using uv for Dependency Management 

This project uses `uv` for Python dependency management:

- **Faster**: uv is 10-100x faster than pip for dependency resolution
- **Better reproducibility**: Lock file (`uv.lock`) ensures exact dependency versions
- **Simplified setup**: `uv sync` handles everything
- **Standard workflow**: Use `uv run` to execute commands in the project environment

**Setup workflow:**
```bash
# 1. Install uv (one-time)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Initialize submodules (uv cannot do this, git is required) - shallow clone
git submodule update --init --recursive --depth 1

# 3. Sync dependencies (creates .venv and installs everything)
uv sync

# 4. Run commands using uv run
uv run cmsis-nn-tools --help
```

**Note**: uv cannot clone git submodules (that's a git feature), so we use `git submodule` commands for that. All Python dependencies are managed entirely by uv.

## Usage

### Basic Usage

```bash
# Run complete pipeline (default: cortex-m55)
uv run cmsis-nn-tools full

# Specify CPU
uv run cmsis-nn-tools full --cpu cortex-m4
```

### Subcommands

- `uv run cmsis-nn-tools generate` — Generate TFLite models and template C/H
- `uv run cmsis-nn-tools runners` — Generate Unity test runners
- `uv run cmsis-nn-tools build` — CMake build for FVP
- `uv run cmsis-nn-tools run` — Run tests on FVP
- `uv run cmsis-nn-tools full` — Run the full pipeline
- `uv run cmsis-nn-tools clean` — Remove build artifacts
- `uv run cmsis-nn-tools doctor` — Preflight checks

### Advanced Usage

```bash
# Run with specific filters
uv run cmsis-nn-tools full --op conv2D --dtype S8 --limit 5

# Skip certain steps
uv run cmsis-nn-tools full --skip-generation --skip-conversion

# Dry run to see what would be done
uv run cmsis-nn-tools full --dry-run

# Verbose output with custom CPU
uv run cmsis-nn-tools full --cpu cortex-m3 -v 2

# Custom optimization level and jobs
uv run cmsis-nn-tools build --opt "-O2" --jobs 8
```

### Command Line Options

#### Pipeline Control
- `--skip-generation`: Skip TFLite model generation
- `--skip-conversion`: Skip TFLite to C conversion
- `--skip-runners`: Skip test runner generation
- `--skip-build`: Skip FVP build
- `--skip-run`: Skip FVP test execution

#### Generation Filters
- `--op OPERATOR`: Generate only specific operator (e.g., 'Conv2D')
- `--dtype DTYPE`: Generate only specific dtype (e.g., 'S8', 'S16')
- `--limit N`: Limit number of models to generate

#### Build Options
- `--cpu CPU`: Target CPU (default: cortex-m55)
- `--opt LEVEL`: Optimization level (default: -Ofast)
- `--jobs N`: Parallel build jobs (default: auto)

#### Run Options
- `--timeout SECONDS`: Per-test timeout in seconds (0 = none)
- `--no-fail-fast`: Don't stop on first test failure

#### General Options
- `--verbosity, -v`: Output verbosity (0–3)
- `--dry-run`: Show what would be done without actually doing it
- `--quiet, -q`: Reduce output verbosity
- `--log-file PATH`: Log file path

## Architecture

All Python lives under `cmsis_nn_tools/`:

### Core Modules
- `cmsis_nn_tools.core.pipeline`: Main pipeline orchestration
- `cmsis_nn_tools.core.config`: Configuration management (repo-root discovery)
- `cmsis_nn_tools.core.discovery`: Path discovery (no __file__-based paths)
- `cmsis_nn_tools.core.steps`: Pipeline steps (generate, runners, build, run, clean)

### CLI
- `cmsis_nn_tools.cli`: Subcommand CLI entry point (`cmsis-nn-tools`)

### FVP
- `cmsis_nn_tools.fvp.build_and_run_fvp`: FVP build and test execution

### Scripts
- `cmsis_nn_tools/scripts/generate_test_runners.py`: Unity test runner generation
- `cmsis_nn_tools/scripts/setup_dependencies.py`: Build dependency download (invoked via `uv run`)

### Generation (TFLite)
- `cmsis_nn_tools/generation/tflite_generator/` contains:
  - `test_ops.py`, `conftest.py`: TFLite model generation via pytest
  - `tester/ops/`: Operator implementations
  - `tester/io/`: I/O utilities
  - `tester/descriptors/`: Test descriptor schemas and examples

### Utility and Reporting
- `cmsis_nn_tools/utils/`: Command execution utilities
- `cmsis_nn_tools/reporting/`: Test result parsing, storage, and report generation

## Development

### Generate TFLite models directly (pytest)

```bash
# From repo root (Tests/CodeGen)
uv run pytest cmsis_nn_tools/generation/tflite_generator/test_ops.py::test_generation -v --op mean_int16
```

### Code Quality (optional)

```bash
# From repo root
black cmsis_nn_tools/
flake8 cmsis_nn_tools/
mypy cmsis_nn_tools/
```

## Requirements

- **uv**: Fast Python package installer (required)
  - Install: `curl -LsSf https://astral.sh/uv/install.sh | sh`
  - All setup scripts require uv
- Python 3.8+
- pytest
- TensorFlow Lite
- helia-aot
- CMake
- ARM GCC toolchain
- FVP Corstone-300 simulator

## CI/CD Integration

This repository includes GitHub Actions workflows (`.github/workflows/ci.yml`) that:
- Checkout code with submodules
- Run the CI setup script
- Verify dependencies and build tools
- Test the CLI

The setup is designed to work seamlessly in CI environments. The `setup_ci.sh` script handles all dependency installation automatically.

For other CI systems, simply run:
```bash
./scripts/setup_ci.sh
```

The script will handle all setup steps including:
- Running `uv sync` (uv must be installed beforehand)
- Optionally downloading build dependencies (ARM GCC, CMSIS-5, Corstone300 FVP, etc.)
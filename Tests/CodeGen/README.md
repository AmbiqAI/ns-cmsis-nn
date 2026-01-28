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
# From the repository root - uses uv sync
uv sync

# Or run the setup script
python3 setup.py
```

**Option 2: Using CLI setup**
```bash
# From the repository root - uses uv (required)
python3 cmsis_nn_tools/cli.py --setup
```

**Option 3: Manual setup**
1. **Initialize submodules** (uv cannot do this, git is required):
   ```bash
   git submodule update --init --recursive --depth 1
   ```

2. **Install Python dependencies with uv sync**:
   ```bash
   # uv sync automatically creates .venv and installs dependencies
   uv sync
   
   # Or manually with uv pip install
   uv venv .venv
   uv pip install --python .venv/bin/python -e .
   ```

**Run the tool**:
```bash
# Using uv run
uv run python cmsis_nn_tools/cli.py --help

# Or activate the virtual environment first
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
python cmsis_nn_tools/cli.py --help
```

### CI Setup

For CI environments, use the `setup_ci.sh` script which:
- Automatically installs `uv` if not present
- Sets up Python virtual environment using `uv`
- Downloads and configures build dependencies (ARM GCC, CMSIS-5, Corstone300 FVP, etc.)
- Configures environment variables

```bash
# In CI scripts or GitHub Actions
python3 setup.py

# Or using the shell script
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
uv run python cmsis_nn_tools/cli.py --help
```

**Note**: uv cannot clone git submodules (that's a git feature), so we use `git submodule` commands for that. All Python dependencies are managed entirely by uv.

## Usage

### Basic Usage

```bash
# Run complete pipeline (default: cortex-m55)
python3 cmsis_nn_tools/cli.py

# Setup dependencies and run pipeline
python3 cmsis_nn_tools/cli.py --setup

# Specify CPU
python3 cmsis_nn_tools/cli.py --cpu cortex-m4
```

### Advanced Usage

```bash
# Run with specific filters
python3 cmsis_nn_tools/cli.py --op conv2D --dtype S8 --limit 5

# Skip certain steps
python3 cmsis_nn_tools/cli.py --skip-generation --skip-conversion

# Dry run to see what would be done
python3 cmsis_nn_tools/cli.py --dry-run

# Verbose output with custom CPU
python3 cmsis_nn_tools/cli.py --cpu cortex-m3 --verbose

# Custom optimization level and jobs
python3 cmsis_nn_tools/cli.py --opt "-O2" --jobs 8
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
- `--setup`: Install Python dependencies (requires uv)
- `--verbose, -v`: Show detailed output
- `--dry-run`: Show what would be done without actually doing it
- `--quiet, -q`: Reduce output verbosity
- `--log-file PATH`: Log file path

## Architecture

All Python lives under `cmsis_nn_tools/`:

### Core Modules
- `cmsis_nn_tools.core.pipeline`: Main pipeline orchestration
- `cmsis_nn_tools.core.config`: Configuration management
- `cmsis_nn_tools.core.logger`: Logging configuration

### Main Scripts
- `cmsis_nn_tools/cli.py`: Command-line interface entry point
- `cmsis_nn_tools/build_and_run_fvp.py`: FVP build and test execution

### Scripts
- `cmsis_nn_tools/scripts/generate_test_runners.py`: Unity test runner generation
- `cmsis_nn_tools/scripts/setup_dependencies.py`: Dependency download and setup
- `cmsis_nn_tools/scripts/collect_coverage.py`: Coverage data collection

### TFLite Generator
- `cmsis_nn_tools/tflite_generator/` contains:
  - `test_ops.py`, `conftest.py`: TFLite model generation via pytest
  - `tester/ops/`: Operator implementations
  - `tester/io/`: I/O utilities
  - `tester/descriptors/`: Test descriptor schemas and examples

### Utility Modules
- `cmsis_nn_tools/utils/command_runner.py`: Command execution utilities

### Reporting Modules
- `cmsis_nn_tools/reporting/`: Test result parsing, storage, and report generation

## Development

### Generate TFLite models directly (pytest)

```bash
cd cmsis_nn_tools/tflite_generator
pytest test_ops.py::test_generation -v --op mean_int16
```

### Code Quality (optional)

```bash
# Format
black cmsis_nn_tools/

# Lint
flake8 cmsis_nn_tools/

# Type check
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
- Installing uv (if not present)
- Setting up Python environment
- Downloading build dependencies (ARM GCC, CMSIS-5, Corstone300 FVP, etc.)
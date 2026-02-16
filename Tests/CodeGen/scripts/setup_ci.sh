#!/bin/bash
# CI Setup Script for CMSIS-AOT-Tester
# This script sets up the environment for CI builds using uv for Python dependencies

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the repository root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Default values (uv sync creates .venv at REPO_ROOT)
DOWNLOADS_DIR="${REPO_ROOT}/artifacts/downloads"
VENV_DIR="${REPO_ROOT}/.venv"
SKIP_BUILD_DEPS=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --downloads-dir)
            DOWNLOADS_DIR="$2"
            shift 2
            ;;
        --skip-build-deps)
            SKIP_BUILD_DEPS=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --downloads-dir DIR    Directory for downloads (default: ./artifacts/downloads)"
            echo "  --skip-build-deps      Skip build dependencies setup"
            echo "  --help                 Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}" >&2
            exit 1
            ;;
    esac
done

echo -e "${GREEN}=== CMSIS-AOT-Tester CI Setup ===${NC}"
echo "Repository root: ${REPO_ROOT}"
echo "Downloads directory: ${DOWNLOADS_DIR}"
echo ""

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Require uv (do not install it)
if ! command_exists uv; then
    echo -e "${RED}uv is required but not found.${NC}" >&2
    echo -e "${RED}Please install uv before running this script: curl -LsSf https://astral.sh/uv/install.sh | sh${NC}" >&2
    exit 1
fi
echo -e "${GREEN}✓ uv found${NC}"

# Create downloads directory
mkdir -p "${DOWNLOADS_DIR}"

# Setup Python environment with uv sync (creates .venv at REPO_ROOT, installs from src layout)
echo ""
echo -e "${GREEN}=== Setting up Python Environment ===${NC}"

cd "${REPO_ROOT}"
uv sync
echo -e "${GREEN}✓ Python dependencies installed with uv sync${NC}"

# Setup build dependencies (ARM GCC, CMSIS-5, etc.)
if [ "${SKIP_BUILD_DEPS}" = false ]; then
    echo ""
    echo -e "${GREEN}=== Setting up Build Dependencies ===${NC}"
    
    SETUP_SCRIPT="${REPO_ROOT}/helia_core_tester/scripts/setup_dependencies.py"
    
    if [ -f "${SETUP_SCRIPT}" ]; then
        (cd "${REPO_ROOT}" && uv run python -m helia_core_tester.scripts.setup_dependencies \
            --downloads-dir "${DOWNLOADS_DIR}" \
            --skip-python) || {
            echo -e "${YELLOW}Warning: Build dependencies setup had issues. Continuing...${NC}" >&2
        }
        
        echo -e "${GREEN}✓ Build dependencies setup complete${NC}"
    else
        echo -e "${YELLOW}Setup script not found at ${SETUP_SCRIPT}. Skipping build dependencies.${NC}"
    fi
else
    echo -e "${YELLOW}Skipping build dependencies setup${NC}"
fi

# Set up environment variables
echo ""
echo -e "${GREEN}=== Environment Configuration ===${NC}"

# Add .venv to PATH (created by uv sync)
if [ -d "${REPO_ROOT}/.venv/bin" ]; then
    export PATH="${REPO_ROOT}/.venv/bin:${PATH}"
elif [ -d "${REPO_ROOT}/.venv/Scripts" ]; then
    export PATH="${REPO_ROOT}/.venv/Scripts:${PATH}"
fi

# Add ARM GCC toolchain to PATH if it exists
ARM_GCC_PATH="${DOWNLOADS_DIR}/arm_gcc_download/bin"
if [ -d "${ARM_GCC_PATH}" ]; then
    export PATH="${ARM_GCC_PATH}:${PATH}"
    echo "ARM GCC toolchain added to PATH"
fi

# Create environment file for sourcing
ENV_FILE="${REPO_ROOT}/.ci_env"
cat > "${ENV_FILE}" <<EOF
# CI Environment Configuration
# Source this file to activate the environment: source .ci_env

export PATH="${REPO_ROOT}/.venv/bin:${PATH}"
export PATH="${ARM_GCC_PATH}:${PATH}"
EOF

echo -e "${GREEN}✓ Environment configured${NC}"
echo ""
echo -e "${GREEN}=== Setup Complete ===${NC}"
echo ""
echo "Virtual environment: ${REPO_ROOT}/.venv (from uv sync)"
echo "Downloads directory: ${DOWNLOADS_DIR}"
echo ""
echo "Run commands with: uv run helia_core_tester <command>"
echo "Or activate: source ${REPO_ROOT}/.venv/bin/activate"
echo "Or source: source ${ENV_FILE}"

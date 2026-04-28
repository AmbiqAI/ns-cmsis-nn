#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Validate a customer-facing ns-cmsis-nn binary package.

Usage:
  validate_binary_package.sh --package-dir <dir> --package-type <binary-sdk|zephyr-binary>
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
METADATA_HELPER="${REPO_ROOT}/scripts/release_package_metadata.py"
CMAKE_TEST="${REPO_ROOT}/cmake/package/tests/assert_variant_resolution.cmake"

PACKAGE_DIR=""
PACKAGE_TYPE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --package-dir)
      PACKAGE_DIR="${2:?missing value for --package-dir}"
      shift 2
      ;;
    --package-type)
      PACKAGE_TYPE="${2:?missing value for --package-type}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

[[ -n "${PACKAGE_DIR}" ]] || { echo "Missing --package-dir" >&2; exit 2; }
[[ -n "${PACKAGE_TYPE}" ]] || { echo "Missing --package-type" >&2; exit 2; }

python3 "${METADATA_HELPER}" validate-package \
  --package-dir "${PACKAGE_DIR}" \
  --package-type "${PACKAGE_TYPE}" \
  --require-binary-only

run_resolution_test() {
  local expected_library="$1"
  shift

  cmake -DPACKAGE_DIR="${PACKAGE_DIR}" \
    -DEXPECTED_LIBRARY="${expected_library}" \
    "$@" \
    -P "${CMAKE_TEST}"
}

run_unsupported_resolution_test() {
  if cmake -DPACKAGE_DIR="${PACKAGE_DIR}" \
    -DEXPECTED_LIBRARY="should-not-resolve.a" \
    "$@" \
    -P "${CMAKE_TEST}" >/dev/null 2>&1; then
    echo "Expected unsupported configuration to fail, but it succeeded" >&2
    exit 1
  fi
}

# Validate the supported package matrix.
run_resolution_test \
  "libns-cmsis-nn-cm0-cortex-m0-gcc-soft-none-ofast.a" \
  -DTEST_C_COMPILER_ID="GNU" \
  -DTEST_SYSTEM_PROCESSOR="cortex-m0"

run_resolution_test \
  "libns-cmsis-nn-cm4-cortex-m4-gcc-hard-fpv4-sp-d16-ofast.a" \
  -DTEST_ZEPHYR_TOOLCHAIN_VARIANT="gnuarmemb" \
  -DTEST_TRUE_SYMBOLS="CONFIG_CPU_CORTEX_M4;CONFIG_FPU"

run_resolution_test \
  "libns-cmsis-nn-cm55-cortex-m55-llvm-et-arm-hard-mve-fp-ofast.a" \
  -DTEST_ZEPHYR_TOOLCHAIN_VARIANT="llvm" \
  -DTEST_TRUE_SYMBOLS="CONFIG_CPU_CORTEX_M55;CONFIG_FPU"

run_resolution_test \
  "libns-cmsis-nn-cm55-cortex-m55-armclang-hard-mve-fp-ofast.a" \
  -DTEST_ZEPHYR_TOOLCHAIN_VARIANT="armclang" \
  -DTEST_TRUE_SYMBOLS="CONFIG_CPU_CORTEX_M55;CONFIG_FPU"

# Validate representative unsupported combinations from the matrix.
run_unsupported_resolution_test \
  -DTEST_ZEPHYR_TOOLCHAIN_VARIANT="gnuarmemb" \
  -DTEST_TRUE_SYMBOLS="CONFIG_CPU_CORTEX_M4"

run_unsupported_resolution_test \
  -DTEST_ZEPHYR_TOOLCHAIN_VARIANT="armclang" \
  -DTEST_TRUE_SYMBOLS="CONFIG_CPU_CORTEX_M4"
echo "Validated ${PACKAGE_TYPE} package at ${PACKAGE_DIR}"

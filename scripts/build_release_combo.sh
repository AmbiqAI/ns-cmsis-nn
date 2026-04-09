#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Build one release static library for packaging.

Usage:
  build_release_combo.sh --arch <cortex-m0|cortex-m4+fp|cortex-m55> \
                         --arch-label <cm0|cm4|cm55> \
                         --target-cpu <cortex-m0|cortex-m4|cortex-m55> \
                         --toolchain <gcc|armclang> \
                         --outdir <dir> \
                         [--build release]

Optional environment overrides:
  DOWNLOADS_DIR
  CMSIS_PATH
  ETHOS_U_CORE_PLATFORM_PATH
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CLEANUP_SCRIPT="${SCRIPT_DIR}/cleanup_release_combo.sh"

ARCH=""
ARCH_LABEL=""
TARGET_CPU=""
TOOLCHAIN=""
BUILD="release"
OUTDIR=""

DOWNLOADS_DIR="${DOWNLOADS_DIR:-${REPO_ROOT}/Tests/UnitTest/downloads}"
CMSIS_PATH="${CMSIS_PATH:-${DOWNLOADS_DIR}/CMSIS_5}"
ETHOS_U_CORE_PLATFORM_PATH="${ETHOS_U_CORE_PLATFORM_PATH:-${DOWNLOADS_DIR}/ethos-u-core-platform}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --arch)
      ARCH="${2:?missing value for --arch}"
      shift 2
      ;;
    --arch-label)
      ARCH_LABEL="${2:?missing value for --arch-label}"
      shift 2
      ;;
    --target-cpu)
      TARGET_CPU="${2:?missing value for --target-cpu}"
      shift 2
      ;;
    --toolchain)
      TOOLCHAIN="${2:?missing value for --toolchain}"
      shift 2
      ;;
    --build)
      BUILD="${2:?missing value for --build}"
      shift 2
      ;;
    --outdir)
      OUTDIR="${2:?missing value for --outdir}"
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

[[ -n "${ARCH}" ]] || { echo "Missing --arch" >&2; usage; exit 2; }
[[ -n "${ARCH_LABEL}" ]] || { echo "Missing --arch-label" >&2; usage; exit 2; }
[[ -n "${TARGET_CPU}" ]] || { echo "Missing --target-cpu" >&2; usage; exit 2; }
[[ -n "${TOOLCHAIN}" ]] || { echo "Missing --toolchain" >&2; usage; exit 2; }
[[ -n "${OUTDIR}" ]] || { echo "Missing --outdir" >&2; usage; exit 2; }
[[ "${BUILD}" == "release" ]] || { echo "Only --build release is supported for release artifacts" >&2; exit 2; }

if ! command -v arm-none-eabi-gcc >/dev/null 2>&1 && [[ -d "${DOWNLOADS_DIR}/arm_gcc_download/bin" ]]; then
  export PATH="${DOWNLOADS_DIR}/arm_gcc_download/bin:${PATH}"
fi

[[ -d "${CMSIS_PATH}" ]] || { echo "CMSIS_PATH not found: ${CMSIS_PATH}" >&2; exit 3; }
[[ -d "${ETHOS_U_CORE_PLATFORM_PATH}" ]] || { echo "ETHOS_U_CORE_PLATFORM_PATH not found: ${ETHOS_U_CORE_PLATFORM_PATH}" >&2; exit 3; }

case "${ARCH}" in
  cortex-m0)
    EXPECTED_TARGET_CPU="cortex-m0"
    EXPECTED_ARCH_LABEL="cm0"
    ;;
  cortex-m4+fp)
    EXPECTED_TARGET_CPU="cortex-m4"
    EXPECTED_ARCH_LABEL="cm4"
    ;;
  cortex-m55)
    EXPECTED_TARGET_CPU="cortex-m55"
    EXPECTED_ARCH_LABEL="cm55"
    ;;
  *)
    echo "Unsupported arch: ${ARCH}" >&2
    exit 2
    ;;
esac

[[ "${ARCH_LABEL}" == "${EXPECTED_ARCH_LABEL}" ]] || {
  echo "arch label mismatch for ${ARCH}: expected ${EXPECTED_ARCH_LABEL}, got ${ARCH_LABEL}" >&2
  exit 2
}

[[ "${TARGET_CPU}" == "${EXPECTED_TARGET_CPU}" ]] || {
  echo "target CPU mismatch for ${ARCH}: expected ${EXPECTED_TARGET_CPU}, got ${TARGET_CPU}" >&2
  exit 2
}

case "${TOOLCHAIN}" in
  gcc)
    TOOLCHAIN_FILE="${ETHOS_U_CORE_PLATFORM_PATH}/cmake/toolchain/arm-none-eabi-gcc.cmake"
    command -v arm-none-eabi-gcc >/dev/null 2>&1 || {
      echo "arm-none-eabi-gcc not found on PATH" >&2
      exit 3
    }
    ;;
  armclang)
    TOOLCHAIN_FILE="${ETHOS_U_CORE_PLATFORM_PATH}/cmake/toolchain/armclang.cmake"
    command -v armclang >/dev/null 2>&1 || {
      echo "armclang not found on PATH" >&2
      exit 3
    }
    ;;
  *)
    echo "Unsupported toolchain: ${TOOLCHAIN}" >&2
    exit 2
    ;;
esac

[[ -f "${TOOLCHAIN_FILE}" ]] || { echo "Toolchain file not found: ${TOOLCHAIN_FILE}" >&2; exit 3; }
[[ -f "${CLEANUP_SCRIPT}" ]] || { echo "Cleanup helper not found: ${CLEANUP_SCRIPT}" >&2; exit 3; }

resolve_strip_tool() {
  local candidate=""
  local path=""

  case "${TOOLCHAIN}" in
    gcc)
      for candidate in arm-none-eabi-strip llvm-strip llvm-strip-16 strip; do
        path="$(command -v "${candidate}" || true)"
        [[ -n "${path}" ]] || continue
        if "${path}" --help 2>&1 | grep -q -- '--strip-debug'; then
          printf '%s\n' "${path}"
          return 0
        fi
      done
      ;;
    armclang)
      for candidate in llvm-strip llvm-strip-16 strip arm-none-eabi-strip; do
        path="$(command -v "${candidate}" || true)"
        [[ -n "${path}" ]] || continue
        if "${path}" --help 2>&1 | grep -q -- '--strip-debug'; then
          printf '%s\n' "${path}"
          return 0
        fi
      done
      ;;
  esac

  return 1
}

STRIP_TOOL="$(resolve_strip_tool || true)"
[[ -n "${STRIP_TOOL}" ]] || { echo "No supported strip tool with --strip-debug found on PATH" >&2; exit 3; }

mkdir -p "$(dirname "${OUTDIR}")"
rm -rf "${OUTDIR}"
mkdir -p "${OUTDIR}"
OUTDIR="$(cd "${OUTDIR}" && pwd)"

if command -v ninja >/dev/null 2>&1; then
  CMAKE_GENERATOR="Ninja"
elif command -v make >/dev/null 2>&1; then
  CMAKE_GENERATOR="Unix Makefiles"
else
  echo "Neither ninja nor make is available on PATH" >&2
  exit 3
fi

LIB_OUT_DIR="${OUTDIR}/lib"
WORK_DIR="$(mktemp -d "${OUTDIR}/.release-work.XXXXXX")"
BUILD_COMPLETE=0
cleanup_work_dir() {
  rm -rf "${WORK_DIR}"
  if [[ "${BUILD_COMPLETE}" -ne 1 ]]; then
    rm -rf "${OUTDIR}"
  fi
}
trap cleanup_work_dir EXIT

BUILD_DIR="${WORK_DIR}/build"
LIB_IN="${BUILD_DIR}/libcmsis-nn.a"

mkdir -p "${LIB_OUT_DIR}"

cmake -S "${REPO_ROOT}" -B "${BUILD_DIR}" -G "${CMAKE_GENERATOR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN_FILE}" \
  -DTARGET_CPU="${TARGET_CPU}" \
  -DCMSIS_PATH="${CMSIS_PATH}"

cmake --build "${BUILD_DIR}" --target cmsis-nn --parallel

[[ -f "${LIB_IN}" ]] || { echo "Built archive not found: ${LIB_IN}" >&2; exit 4; }

LIB_NAME="libns-cmsis-nn-${ARCH_LABEL}-${TOOLCHAIN}.a"
FINAL_LIB="${LIB_OUT_DIR}/${LIB_NAME}"

cp "${LIB_IN}" "${FINAL_LIB}"
"${STRIP_TOOL}" --strip-debug "${FINAL_LIB}"

bash "${CLEANUP_SCRIPT}" \
  --outdir "${OUTDIR}" \
  --lib-name "${LIB_NAME}"

BUILD_COMPLETE=1
echo "Built and verified combo in ${OUTDIR}"

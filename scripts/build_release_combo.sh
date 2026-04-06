#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Build one release static library and prune the archive to the reviewed public API.

Usage:
  build_release_combo.sh --arch <cortex-m0|cortex-m4+fp|cortex-m55> \
                         --toolchain <gcc|armclang> \
                         --outdir <dir> \
                         [--build release]

Optional environment overrides:
  DOWNLOADS_DIR
  CMSIS_PATH
  ETHOS_U_CORE_PLATFORM_PATH
  PUBLIC_SYMBOLS
  PUBLIC_API_HEADER
  OBJCOPY
  NM
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ARCH=""
TOOLCHAIN=""
BUILD="release"
OUTDIR=""

DOWNLOADS_DIR="${DOWNLOADS_DIR:-${REPO_ROOT}/Tests/UnitTest/downloads}"
CMSIS_PATH="${CMSIS_PATH:-${DOWNLOADS_DIR}/CMSIS_5}"
ETHOS_U_CORE_PLATFORM_PATH="${ETHOS_U_CORE_PLATFORM_PATH:-${DOWNLOADS_DIR}/ethos-u-core-platform}"
PUBLIC_SYMBOLS="${PUBLIC_SYMBOLS:-${REPO_ROOT}/release/public_symbols.txt}"
PUBLIC_API_HEADER="${PUBLIC_API_HEADER:-${REPO_ROOT}/release/public_api.h}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --arch)
      ARCH="${2:?missing value for --arch}"
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
[[ -n "${TOOLCHAIN}" ]] || { echo "Missing --toolchain" >&2; usage; exit 2; }
[[ -n "${OUTDIR}" ]] || { echo "Missing --outdir" >&2; usage; exit 2; }
[[ "${BUILD}" == "release" ]] || { echo "Only --build release is supported for release artifacts" >&2; exit 2; }

if ! command -v arm-none-eabi-gcc >/dev/null 2>&1 && [[ -d "${DOWNLOADS_DIR}/arm_gcc_download/bin" ]]; then
  export PATH="${DOWNLOADS_DIR}/arm_gcc_download/bin:${PATH}"
fi

OBJCOPY="${OBJCOPY:-$(command -v arm-none-eabi-objcopy || true)}"
NM="${NM:-$(command -v arm-none-eabi-nm || true)}"
STRIP="${STRIP:-$(command -v arm-none-eabi-strip || true)}"

[[ -n "${OBJCOPY}" ]] || { echo "arm-none-eabi-objcopy not found on PATH" >&2; exit 3; }
[[ -n "${NM}" ]] || { echo "arm-none-eabi-nm not found on PATH" >&2; exit 3; }
[[ -f "${PUBLIC_SYMBOLS}" ]] || { echo "public symbols file not found: ${PUBLIC_SYMBOLS}" >&2; exit 3; }
[[ -f "${PUBLIC_API_HEADER}" ]] || { echo "public API header not found: ${PUBLIC_API_HEADER}" >&2; exit 3; }
[[ -d "${CMSIS_PATH}" ]] || { echo "CMSIS_PATH not found: ${CMSIS_PATH}" >&2; exit 3; }
[[ -d "${ETHOS_U_CORE_PLATFORM_PATH}" ]] || { echo "ETHOS_U_CORE_PLATFORM_PATH not found: ${ETHOS_U_CORE_PLATFORM_PATH}" >&2; exit 3; }

case "${ARCH}" in
  cortex-m0)
    TARGET_CPU="cortex-m0"
    ARCH_LABEL="cm0"
    ;;
  cortex-m4+fp)
    TARGET_CPU="cortex-m4"
    ARCH_LABEL="cm4"
    ;;
  cortex-m55)
    TARGET_CPU="cortex-m55"
    ARCH_LABEL="cm55"
    ;;
  *)
    echo "Unsupported arch: ${ARCH}" >&2
    exit 2
    ;;
esac

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

BUILD_DIR="${OUTDIR}/build"
STAGE_DIR="${OUTDIR}/stage"
STAGE_META_DIR="${STAGE_DIR}/meta"
STAGE_INCLUDE_DIR="${STAGE_DIR}/include/cmsis-nn"
STAGE_LIB_DIR="${STAGE_DIR}/lib"
LIB_IN="${BUILD_DIR}/libcmsis-nn.a"
PRUNED_LIB="${STAGE_LIB_DIR}/libcmsis-nn-public.a"
CANDIDATE="${STAGE_META_DIR}/public_symbols.candidate.txt"
RESOLVED="${STAGE_META_DIR}/public_symbols.resolved.txt"
REPORT="${STAGE_META_DIR}/public_symbol_validation.txt"
EXPORTS_OUT="${STAGE_META_DIR}/exported_symbols.txt"

mkdir -p "${STAGE_META_DIR}" "${STAGE_LIB_DIR}"

python3 "${REPO_ROOT}/scripts/generate_public_symbols.py" \
  --hdr-dir "${REPO_ROOT}/Include" \
  --out "${CANDIDATE}"

python3 "${REPO_ROOT}/scripts/validate_public_symbols.py" \
  --candidate "${CANDIDATE}" \
  --committed "${PUBLIC_SYMBOLS}" \
  --resolved-out "${RESOLVED}" \
  --report "${REPORT}"

python3 "${REPO_ROOT}/scripts/annotate_public_headers.py" \
  --symbols "${PUBLIC_SYMBOLS}" \
  --hdr-dir "${REPO_ROOT}/Include" \
  --out-dir "${STAGE_INCLUDE_DIR}" \
  --public-api "${PUBLIC_API_HEADER}"

cp "${PUBLIC_SYMBOLS}" "${STAGE_META_DIR}/public_symbols.txt"

cmake -S "${REPO_ROOT}" -B "${BUILD_DIR}" -G "${CMAKE_GENERATOR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN_FILE}" \
  -DTARGET_CPU="${TARGET_CPU}" \
  -DCMSIS_PATH="${CMSIS_PATH}" \
  -DCMSIS_NN_HIDE_INTERNAL_SYMBOLS=ON \
  -DCMSIS_NN_PUBLIC_HEADERS_DIR="${STAGE_INCLUDE_DIR}"

cmake --build "${BUILD_DIR}" --target cmsis-nn --parallel

[[ -f "${LIB_IN}" ]] || { echo "Built archive not found: ${LIB_IN}" >&2; exit 4; }

cp "${LIB_IN}" "${PRUNED_LIB}"
"${OBJCOPY}" --keep-global-symbols="${RESOLVED}" "${PRUNED_LIB}"
if [[ -n "${STRIP}" ]]; then
  "${STRIP}" --strip-debug "${PRUNED_LIB}"
else
  "${OBJCOPY}" --strip-debug "${PRUNED_LIB}"
fi

"${NM}" -g --defined-only "${PRUNED_LIB}" \
  | awk 'NF >= 3 && $NF !~ /:$/ { print $NF }' \
  | sort -u > "${EXPORTS_OUT}"

if ! diff -u "${RESOLVED}" "${EXPORTS_OUT}"; then
  echo "Exported symbol validation failed for ${PRUNED_LIB}" >&2
  exit 5
fi

LIB_NAME="libns-cmsis-nn-${ARCH_LABEL}-${TOOLCHAIN}.a"

cp "${PRUNED_LIB}" "${STAGE_LIB_DIR}/${LIB_NAME}"
cp "${CANDIDATE}" "${STAGE_META_DIR}/public_symbols.candidate.txt"
cp "${RESOLVED}" "${STAGE_META_DIR}/public_symbols.resolved.txt"
cp "${REPORT}" "${STAGE_META_DIR}/public_symbol_validation.txt"
cp "${EXPORTS_OUT}" "${STAGE_META_DIR}/exported_symbols.txt"

cat > "${STAGE_META_DIR}/combo.env" <<EOF
PRODUCT=ns-cmsis-nn
ARCH=${ARCH}
TOOLCHAIN=${TOOLCHAIN}
TARGET_CPU=${TARGET_CPU}
LIB_NAME=${LIB_NAME}
EOF

python3 - "${STAGE_META_DIR}/product_manifest.json" "ns-cmsis-nn" "${ARCH}" "${TOOLCHAIN}" "${LIB_NAME}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
manifest = {
    "schema": 1,
    "product": sys.argv[2],
    "arch": sys.argv[3],
    "toolchain": sys.argv[4],
    "library": sys.argv[5],
}
out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
PY

echo "Built and verified combo in ${OUTDIR}"

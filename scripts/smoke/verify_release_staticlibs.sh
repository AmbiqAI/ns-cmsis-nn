#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Verify generated ns-cmsis-nn static libraries by smoke-linking a tiny app.

Usage:
  verify_release_staticlibs.sh --lib <path> [--outdir <dir>]
  verify_release_staticlibs.sh --artifact-root <dir> [--outdir <dir>]

Exactly one of --lib or --artifact-root is required.

Optional environment overrides:
  DOWNLOADS_DIR
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SMOKE_SOURCE="${SCRIPT_DIR}/smoke_link_test.c"

LIB_PATH=""
ARTIFACT_ROOT=""
OUTDIR="${REPO_ROOT}/out/staticlib-smoke"
DOWNLOADS_DIR="${DOWNLOADS_DIR:-${REPO_ROOT}/Tests/UnitTest/downloads}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --lib)
      LIB_PATH="${2:?missing value for --lib}"
      shift 2
      ;;
    --artifact-root)
      ARTIFACT_ROOT="${2:?missing value for --artifact-root}"
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

[[ -f "${SMOKE_SOURCE}" ]] || { echo "Smoke source not found: ${SMOKE_SOURCE}" >&2; exit 3; }
[[ -d "${REPO_ROOT}/Include" ]] || { echo "Include directory not found: ${REPO_ROOT}/Include" >&2; exit 3; }

if [[ -n "${LIB_PATH}" && -n "${ARTIFACT_ROOT}" ]]; then
  echo "Pass only one of --lib or --artifact-root" >&2
  usage
  exit 2
fi

if [[ -z "${LIB_PATH}" && -z "${ARTIFACT_ROOT}" ]]; then
  echo "One of --lib or --artifact-root is required" >&2
  usage
  exit 2
fi

if ! command -v arm-none-eabi-gcc >/dev/null 2>&1 && [[ -x "${DOWNLOADS_DIR}/arm_gcc_download/bin/arm-none-eabi-gcc" ]]; then
  export PATH="${DOWNLOADS_DIR}/arm_gcc_download/bin:${PATH}"
fi

GNU_CC="$(command -v arm-none-eabi-gcc || true)"
[[ -n "${GNU_CC}" ]] || { echo "arm-none-eabi-gcc not found on PATH" >&2; exit 3; }

mkdir -p "${OUTDIR}"
OUTDIR="$(cd "${OUTDIR}" && pwd)"

metadata_for_archive() {
  local archive_path="$1"
  local metadata_path="${archive_path%.a}.json"

  [[ -f "${metadata_path}" ]] || {
    echo "Metadata not found for $(basename "${archive_path}"): ${metadata_path}" >&2
    exit 3
  }

  printf '%s\n' "${metadata_path}"
}

resolve_cpu_flags() {
  local archive_path="$1"
  local metadata_path="$2"
  local resolved

  resolved="$(python3 - "${archive_path}" "${metadata_path}" <<'PY'
import json
import sys
from pathlib import Path

archive_path = Path(sys.argv[1])
metadata_path = Path(sys.argv[2])

try:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
except Exception as exc:
    raise SystemExit(f"failed to read metadata {metadata_path}: {exc}")

expected_file = metadata.get("file")
if expected_file and expected_file != archive_path.name:
    raise SystemExit(
        f"metadata file mismatch for {archive_path.name}: expected {expected_file}"
    )

target_cpu = metadata.get("target_cpu")
float_abi = metadata.get("float_abi")
toolchain = metadata.get("toolchain")

if not target_cpu or not float_abi or not toolchain:
    raise SystemExit(
        f"metadata {metadata_path} must contain target_cpu, float_abi, and toolchain"
    )

print(f"{target_cpu}\t{float_abi}\t{toolchain}")
PY
)"

  IFS=$'\t' read -r TARGET_CPU FLOAT_ABI TOOLCHAIN <<< "${resolved}"

  case "${TARGET_CPU}:${FLOAT_ABI}" in
    cortex-m0:soft)
      CPU_FLAGS=(-mcpu=cortex-m0 -mthumb -mfloat-abi=soft)
      ;;
    cortex-m4:hard)
      CPU_FLAGS=(-mcpu=cortex-m4 -mthumb -mfloat-abi=hard -mfpu=fpv4-sp-d16)
      ;;
    cortex-m55:hard)
      CPU_FLAGS=(-mcpu=cortex-m55 -mthumb -mfloat-abi=hard)
      ;;
    *)
      echo "Unsupported static-library target ABI in ${metadata_path}: target_cpu=${TARGET_CPU}, float_abi=${FLOAT_ABI}" >&2
      exit 2
      ;;
  esac

  case "${TOOLCHAIN}" in
    gcc)
      CPU_FLAGS+=(-fshort-enums)
      ;;
    armclang|llvm-et-arm)
      CPU_FLAGS+=(-fno-short-enums)
      ;;
    *)
      echo "Unsupported static-library toolchain in ${metadata_path}: toolchain=${TOOLCHAIN}" >&2
      exit 2
      ;;
  esac
}

LIB_PATHS=()
if [[ -n "${LIB_PATH}" ]]; then
  [[ -f "${LIB_PATH}" ]] || { echo "Library not found: ${LIB_PATH}" >&2; exit 3; }
  LIB_PATHS+=("$(cd "$(dirname "${LIB_PATH}")" && pwd)/$(basename "${LIB_PATH}")")
else
  [[ -d "${ARTIFACT_ROOT}" ]] || { echo "Artifact root not found: ${ARTIFACT_ROOT}" >&2; exit 3; }
  while IFS= read -r found_path; do
    LIB_PATHS+=("${found_path}")
  done < <(find "${ARTIFACT_ROOT}" -type f -name 'libns-cmsis-nn-*.a' | sort)
fi

[[ "${#LIB_PATHS[@]}" -gt 0 ]] || {
  echo "No generated static libraries found" >&2
  exit 4
}

for archive_path in "${LIB_PATHS[@]}"; do
  lib_name="$(basename "${archive_path}")"
  metadata_path="$(metadata_for_archive "${archive_path}")"
  resolve_cpu_flags "${archive_path}" "${metadata_path}"

  smoke_dir="${OUTDIR}/${lib_name%.a}"
  mkdir -p "${smoke_dir}"

  smoke_obj="${smoke_dir}/smoke_link_test.o"
  smoke_rel="${smoke_dir}/smoke_link_test.rel"

  echo "Smoke-linking ${lib_name} for ${TARGET_CPU} (${TOOLCHAIN})"

  "${GNU_CC}" \
    "${CPU_FLAGS[@]}" \
    -O3 \
    -DNDEBUG \
    -ffunction-sections \
    -fdata-sections \
    -Wa,--noexecstack \
    -I"${REPO_ROOT}/Include" \
    -c "${SMOKE_SOURCE}" \
    -o "${smoke_obj}"

  "${GNU_CC}" \
    "${CPU_FLAGS[@]}" \
    -r \
    -Wl,--no-undefined \
    "${smoke_obj}" \
    -Wl,--whole-archive \
    "${archive_path}" \
    -Wl,--no-whole-archive \
    -o "${smoke_rel}"
done

echo "Verified ${#LIB_PATHS[@]} static libraries via smoke-linking"

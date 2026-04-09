#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Remove all release combo byproducts except the final static library.

Usage:
  cleanup_release_combo.sh --outdir <dir> --lib-name <filename>
EOF
}

OUTDIR=""
LIB_NAME=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --outdir)
      OUTDIR="${2:?missing value for --outdir}"
      shift 2
      ;;
    --lib-name)
      LIB_NAME="${2:?missing value for --lib-name}"
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

[[ -n "${OUTDIR}" ]] || { echo "Missing --outdir" >&2; usage; exit 2; }
[[ -n "${LIB_NAME}" ]] || { echo "Missing --lib-name" >&2; usage; exit 2; }

[[ -d "${OUTDIR}" ]] || { echo "Output directory not found: ${OUTDIR}" >&2; exit 3; }
OUTDIR="$(cd "${OUTDIR}" && pwd)"
LIB_DIR="${OUTDIR}/lib"
TARGET_LIB="${LIB_DIR}/${LIB_NAME}"

[[ -f "${TARGET_LIB}" ]] || { echo "Expected final library not found: ${TARGET_LIB}" >&2; exit 3; }

shopt -s dotglob nullglob

for path in "${OUTDIR}"/*; do
  [[ "${path}" == "${LIB_DIR}" ]] && continue
  rm -rf "${path}"
done

for path in "${LIB_DIR}"/*; do
  [[ "${path}" == "${TARGET_LIB}" ]] && continue
  rm -rf "${path}"
done

shopt -u dotglob nullglob

remaining_entries=("${OUTDIR}"/*)
[[ "${#remaining_entries[@]}" -eq 1 && "${remaining_entries[0]}" == "${LIB_DIR}" ]] || {
  echo "Cleanup left unexpected entries in ${OUTDIR}" >&2
  exit 4
}

remaining_libs=("${LIB_DIR}"/*)
[[ "${#remaining_libs[@]}" -eq 1 && "${remaining_libs[0]}" == "${TARGET_LIB}" ]] || {
  echo "Cleanup left unexpected files in ${LIB_DIR}" >&2
  exit 4
}

#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Build or package release static libraries for ns-cmsis-nn.

Usage:
  build_release_bundle.sh --tag <tag> --outdir <dir> [--artifact-root <dir>] [--skip-build]
                          [--archs cortex-m0,cortex-m4+fp,cortex-m55]
                          [--toolchains gcc,armclang,llvm-et-arm]
                          [--build release]
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

resolve_arch() {
  local arch="$1"

  case "${arch}" in
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
      echo "Unsupported arch: ${arch}" >&2
      exit 2
      ;;
  esac
}

TAG=""
OUTDIR=""
ARTIFACT_ROOT="${REPO_ROOT}/out/release-artifacts"
ARCHS="cortex-m0,cortex-m4+fp,cortex-m55"
TOOLCHAINS="gcc,armclang,llvm-et-arm"
BUILD="release"
SKIP_BUILD=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag)
      TAG="${2:?missing value for --tag}"
      shift 2
      ;;
    --outdir)
      OUTDIR="${2:?missing value for --outdir}"
      shift 2
      ;;
    --artifact-root)
      ARTIFACT_ROOT="${2:?missing value for --artifact-root}"
      shift 2
      ;;
    --archs)
      ARCHS="${2:?missing value for --archs}"
      shift 2
      ;;
    --toolchains)
      TOOLCHAINS="${2:?missing value for --toolchains}"
      shift 2
      ;;
    --build)
      BUILD="${2:?missing value for --build}"
      shift 2
      ;;
    --skip-build)
      SKIP_BUILD=1
      shift
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

[[ -n "${TAG}" ]] || { echo "Missing --tag" >&2; usage; exit 2; }
[[ -n "${OUTDIR}" ]] || { echo "Missing --outdir" >&2; usage; exit 2; }
[[ "${BUILD}" == "release" ]] || { echo "Only --build release is supported" >&2; exit 2; }

mkdir -p "${ARTIFACT_ROOT}" "${OUTDIR}"

IFS=',' read -r -a ARCH_ARRAY <<< "${ARCHS}"
IFS=',' read -r -a TOOLCHAIN_ARRAY <<< "${TOOLCHAINS}"

EXPECTED_LIBS=()
for arch in "${ARCH_ARRAY[@]}"; do
  resolve_arch "${arch}"
  for toolchain in "${TOOLCHAIN_ARRAY[@]}"; do
    EXPECTED_LIBS+=("libns-cmsis-nn-${ARCH_LABEL}-${toolchain}.a")
  done
done

if [[ "${SKIP_BUILD}" -eq 0 ]]; then
  for arch in "${ARCH_ARRAY[@]}"; do
    resolve_arch "${arch}"
    for toolchain in "${TOOLCHAIN_ARRAY[@]}"; do
      bash "${SCRIPT_DIR}/build_release_combo.sh" \
        --arch "${arch}" \
        --arch-label "${ARCH_LABEL}" \
        --target-cpu "${TARGET_CPU}" \
        --toolchain "${toolchain}" \
        --build "${BUILD}" \
        --outdir "${ARTIFACT_ROOT}/${arch}/${toolchain}/${BUILD}"
    done
  done
fi

ACTUAL_LIB_PATHS=()
while IFS= read -r lib_path; do
  ACTUAL_LIB_PATHS+=("${lib_path}")
done < <(find "${ARTIFACT_ROOT}" -type f -name 'libns-cmsis-nn-*.a' | sort)

[[ "${#ACTUAL_LIB_PATHS[@]}" -gt 0 ]] || {
  echo "No release static libraries found under ${ARTIFACT_ROOT}" >&2
  exit 4
}

[[ "${#ACTUAL_LIB_PATHS[@]}" -eq "${#EXPECTED_LIBS[@]}" ]] || {
  echo "Expected ${#EXPECTED_LIBS[@]} release static libraries under ${ARTIFACT_ROOT}, found ${#ACTUAL_LIB_PATHS[@]}" >&2
  exit 4
}

expected_list="$(mktemp)"
actual_list="$(mktemp)"
cleanup() {
  rm -f "${expected_list}" "${actual_list}"
}
trap cleanup EXIT

printf '%s\n' "${EXPECTED_LIBS[@]}" | sort > "${expected_list}"
for lib_path in "${ACTUAL_LIB_PATHS[@]}"; do
  basename "${lib_path}"
done | sort > "${actual_list}"

if ! diff -u "${expected_list}" "${actual_list}"; then
  echo "Release static library set does not match expected combos" >&2
  exit 4
fi

BUNDLE_DIR="${OUTDIR}/ns-cmsis-nn-staticlibs-${TAG}"
ZIP_PATH="${OUTDIR}/ns-cmsis-nn-staticlibs-${TAG}.zip"
rm -rf "${BUNDLE_DIR}" "${ZIP_PATH}"
mkdir -p "${BUNDLE_DIR}/lib"

for lib_name in "${EXPECTED_LIBS[@]}"; do
  LIB_MATCHES=()
  while IFS= read -r lib_path; do
    LIB_MATCHES+=("${lib_path}")
  done < <(find "${ARTIFACT_ROOT}" -type f -name "${lib_name}" | sort)

  [[ "${#LIB_MATCHES[@]}" -eq 1 ]] || {
    echo "Expected exactly one artifact for ${lib_name}, found ${#LIB_MATCHES[@]}" >&2
    exit 4
  }

  cp "${LIB_MATCHES[0]}" "${BUNDLE_DIR}/lib/"
done

python3 - "${BUNDLE_DIR}" "${TAG}" <<'PY'
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

bundle_dir = Path(sys.argv[1])
tag = sys.argv[2]

manifest = {
    "schema": 1,
    "product": "ns-cmsis-nn",
    "tag": tag,
    "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "libraries": [],
}

for path in sorted((bundle_dir / "lib").glob("*.a")):
    parts = path.stem.split("-")
    manifest["libraries"].append(
        {
            "file": path.name,
            "arch": parts[3],
            "toolchain": "-".join(parts[4:]),
        }
    )

(bundle_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
PY

(
  cd "${OUTDIR}"
  zip -rq "ns-cmsis-nn-staticlibs-${TAG}.zip" "ns-cmsis-nn-staticlibs-${TAG}"
)

echo "Wrote release bundles to ${OUTDIR}"

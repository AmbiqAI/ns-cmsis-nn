#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Build or package Helia-style release bundles for ns-cmsis-nn and helia-core.

Usage:
  build_release_bundle.sh --tag <tag> --outdir <dir> [--artifact-root <dir>] [--skip-build]
                          [--archs cortex-m4+fp,cortex-m55]
                          [--toolchains gcc,armclang]
                          [--build release]
                          [--visibility-mode single-facade|curated-tree|library-only]
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TAG=""
OUTDIR=""
ARTIFACT_ROOT="${REPO_ROOT}/out/release-artifacts"
ARCHS="cortex-m4+fp,cortex-m55"
TOOLCHAINS="gcc,armclang"
BUILD="release"
SKIP_BUILD=0
VISIBILITY_MODE="single-facade"

copy_single_facade_headers() {
  local source_dir="$1"
  local bundle_dir="$2"

  mkdir -p "${bundle_dir}/include/cmsis-nn/Internal"

  local required_headers=(
    "cmsis_nn.h"
    "public_api.h"
    "arm_nn_math_types.h"
    "arm_nn_types.h"
    "arm_nnfunctions.h"
    "arm_nnsupportfunctions.h"
    "Internal/arm_nn_compiler.h"
  )

  local header
  for header in "${required_headers[@]}"; do
    [[ -f "${source_dir}/include/cmsis-nn/${header}" ]] || {
      echo "Missing required single-facade header: ${header}" >&2
      exit 4
    }
    cp "${source_dir}/include/cmsis-nn/${header}" "${bundle_dir}/include/cmsis-nn/${header}"
  done
}

copy_headers_for_mode() {
  local source_dir="$1"
  local bundle_dir="$2"

  case "${VISIBILITY_MODE}" in
    single-facade)
      copy_single_facade_headers "${source_dir}" "${bundle_dir}"
      ;;
    curated-tree)
      mkdir -p "${bundle_dir}/include"
      cp -R "${source_dir}/include/cmsis-nn" "${bundle_dir}/include/"
      ;;
    library-only)
      ;;
    *)
      echo "Unsupported visibility mode: ${VISIBILITY_MODE}" >&2
      exit 2
      ;;
  esac
}

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
    --visibility-mode)
      VISIBILITY_MODE="${2:?missing value for --visibility-mode}"
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

[[ -n "${TAG}" ]] || { echo "Missing --tag" >&2; usage; exit 2; }
[[ -n "${OUTDIR}" ]] || { echo "Missing --outdir" >&2; usage; exit 2; }
[[ "${BUILD}" == "release" ]] || { echo "Only --build release is supported" >&2; exit 2; }

case "${VISIBILITY_MODE}" in
  single-facade|curated-tree|library-only)
    ;;
  *)
    echo "Unsupported visibility mode: ${VISIBILITY_MODE}" >&2
    exit 2
    ;;
esac

mkdir -p "${ARTIFACT_ROOT}" "${OUTDIR}"

if [[ "${SKIP_BUILD}" -eq 0 ]]; then
  IFS=',' read -r -a ARCH_ARRAY <<< "${ARCHS}"
  IFS=',' read -r -a TOOLCHAIN_ARRAY <<< "${TOOLCHAINS}"

  for arch in "${ARCH_ARRAY[@]}"; do
    for toolchain in "${TOOLCHAIN_ARRAY[@]}"; do
      bash "${SCRIPT_DIR}/build_release_combo.sh" \
        --arch "${arch}" \
        --toolchain "${toolchain}" \
        --build "${BUILD}" \
        --visibility-mode "${VISIBILITY_MODE}" \
        --outdir "${ARTIFACT_ROOT}/${arch}/${toolchain}/${BUILD}"
    done
  done
fi

declare -a PRODUCTS=("ns-cmsis-nn" "helia-core")

for product in "${PRODUCTS[@]}"; do
  PRODUCT_DIRS=()
  while IFS= read -r dir; do
    PRODUCT_DIRS+=("${dir}")
  done < <(find "${ARTIFACT_ROOT}" -type d -path "*/products/${product}" | sort)
  [[ "${#PRODUCT_DIRS[@]}" -gt 0 ]] || {
    echo "No product directories found for ${product} under ${ARTIFACT_ROOT}" >&2
    exit 4
  }

  BUNDLE_DIR="${OUTDIR}/${product}-${TAG}-${VISIBILITY_MODE}"
  ZIP_PATH="${OUTDIR}/${product}-${TAG}-${VISIBILITY_MODE}.zip"
  rm -rf "${BUNDLE_DIR}" "${ZIP_PATH}"
  mkdir -p "${BUNDLE_DIR}/lib"

  FIRST_DIR="${PRODUCT_DIRS[0]}"
  copy_headers_for_mode "${FIRST_DIR}" "${BUNDLE_DIR}"

  for dir in "${PRODUCT_DIRS[@]}"; do
    lib_path="$(find "${dir}/lib" -maxdepth 1 -type f -name '*.a' | sort | head -n1)"
    [[ -n "${lib_path}" ]] || {
      echo "Missing product archive under ${dir}/lib" >&2
      exit 4
    }
    cp "${lib_path}" "${BUNDLE_DIR}/lib/"
  done

  python3 - "${BUNDLE_DIR}" "${product}" "${TAG}" "${VISIBILITY_MODE}" <<'PY'
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

bundle_dir = Path(sys.argv[1])
product = sys.argv[2]
tag = sys.argv[3]
visibility_mode = sys.argv[4]
include_root = "include/cmsis-nn" if (bundle_dir / "include" / "cmsis-nn").is_dir() else None

manifest = {
    "schema": 1,
    "name": f"{product}-{tag}-{visibility_mode}",
    "product": product,
    "tag": tag,
    "visibility_mode": visibility_mode,
    "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "libraries": sorted(path.name for path in (bundle_dir / "lib").glob("*.a")),
    "include_root": include_root,
}
(bundle_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
PY

  (
    cd "${OUTDIR}"
    zip -rq "${product}-${TAG}-${VISIBILITY_MODE}.zip" "${product}-${TAG}-${VISIBILITY_MODE}"
  )
done

echo "Wrote release bundles to ${OUTDIR}"

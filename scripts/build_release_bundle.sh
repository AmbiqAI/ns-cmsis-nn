#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Build or package release static libraries for ns-cmsis-nn.

Usage:
  build_release_bundle.sh --tag <tag> --outdir <dir> [--artifact-root <dir>] [--skip-build]
                          [--archs cortex-m0,cortex-m4+fp,cortex-m55]
                          [--toolchains gcc,armclang]
                          [--build release]
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TAG=""
OUTDIR=""
ARTIFACT_ROOT="${REPO_ROOT}/out/release-artifacts"
ARCHS="cortex-m0,cortex-m4+fp,cortex-m55"
TOOLCHAINS="gcc,armclang"
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

if [[ "${SKIP_BUILD}" -eq 0 ]]; then
  IFS=',' read -r -a ARCH_ARRAY <<< "${ARCHS}"
  IFS=',' read -r -a TOOLCHAIN_ARRAY <<< "${TOOLCHAINS}"

  for arch in "${ARCH_ARRAY[@]}"; do
    for toolchain in "${TOOLCHAIN_ARRAY[@]}"; do
      bash "${SCRIPT_DIR}/build_release_combo.sh" \
        --arch "${arch}" \
        --toolchain "${toolchain}" \
        --build "${BUILD}" \
        --outdir "${ARTIFACT_ROOT}/${arch}/${toolchain}/${BUILD}"
    done
  done
fi

STAGE_DIRS=()
while IFS= read -r dir; do
  STAGE_DIRS+=("${dir}")
done < <(find "${ARTIFACT_ROOT}" -type d -path "*/stage" | sort)
[[ "${#STAGE_DIRS[@]}" -gt 0 ]] || {
  echo "No staged static libraries found under ${ARTIFACT_ROOT}" >&2
  exit 4
}

BUNDLE_DIR="${OUTDIR}/ns-cmsis-nn-staticlibs-${TAG}"
ZIP_PATH="${OUTDIR}/ns-cmsis-nn-staticlibs-${TAG}.zip"
rm -rf "${BUNDLE_DIR}" "${ZIP_PATH}"
mkdir -p "${BUNDLE_DIR}/lib"

for dir in "${STAGE_DIRS[@]}"; do
  lib_path="$(find "${dir}/lib" -maxdepth 1 -type f -name 'libns-cmsis-nn-*.a' | sort | head -n1)"
  [[ -n "${lib_path}" ]] || {
    echo "Missing static library under ${dir}/lib" >&2
    exit 4
  }
  cp "${lib_path}" "${BUNDLE_DIR}/lib/"
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
            "toolchain": parts[4],
        }
    )

(bundle_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
PY

(
  cd "${OUTDIR}"
  zip -rq "ns-cmsis-nn-staticlibs-${TAG}.zip" "ns-cmsis-nn-staticlibs-${TAG}"
)

echo "Wrote release bundles to ${OUTDIR}"

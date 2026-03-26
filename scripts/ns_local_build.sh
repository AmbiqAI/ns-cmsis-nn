#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Build customer release bundles locally in the same shape as the GitHub release workflows.

Usage:
  ns_local_build.sh --tag <tag> [--outdir <dir>]
                    [--archs cortex-m0,cortex-m4+fp,cortex-m55]
                    [--toolchains gcc,armclang]
                    [--visibility-modes single-facade,library-only]
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TAG="local"
OUTDIR="${REPO_ROOT}/out/ns_local_build"
ARCHS="cortex-m0,cortex-m4+fp,cortex-m55"
TOOLCHAINS="gcc,armclang"
VISIBILITY_MODES="single-facade,library-only"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag)
      TAG="${2:?missing value for --tag}"
      shift 2
      ;;
    -o|--outdir)
      OUTDIR="${2:?missing value for --outdir}"
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
    --visibility-modes)
      VISIBILITY_MODES="${2:?missing value for --visibility-modes}"
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

mkdir -p "${OUTDIR}"

IFS=',' read -r -a MODE_ARRAY <<< "${VISIBILITY_MODES}"
for mode in "${MODE_ARRAY[@]}"; do
  bash "${SCRIPT_DIR}/build_release_bundle.sh" \
    --tag "${TAG}" \
    --archs "${ARCHS}" \
    --toolchains "${TOOLCHAINS}" \
    --visibility-mode "${mode}" \
    --outdir "${OUTDIR}/${mode}"
done

echo "Wrote local release bundles to ${OUTDIR}"

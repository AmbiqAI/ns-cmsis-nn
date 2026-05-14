#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Build release static libraries locally in the same shape as the GitHub release workflow.

Usage:
  ns_local_build.sh [--tag <tag>] [--outdir <dir>]
                    [--archs cortex-m0,cortex-m4+fp,cortex-m55]
                    [--toolchains gcc,armclang,atfe]

Defaults:
  --tag local
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TAG="local"
OUTDIR="${REPO_ROOT}/out/ns_local_build"
ARCHS="cortex-m0,cortex-m4+fp,cortex-m55"
TOOLCHAINS="gcc,armclang,atfe"

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

bash "${SCRIPT_DIR}/build_release_bundle.sh" \
  --tag "${TAG}" \
  --archs "${ARCHS}" \
  --toolchains "${TOOLCHAINS}" \
  --outdir "${OUTDIR}"

bash "${SCRIPT_DIR}/build_zephyr_release_package.sh" \
  --tag "${TAG}" \
  --outdir "${OUTDIR}"

echo "Wrote local release packages to ${OUTDIR}"

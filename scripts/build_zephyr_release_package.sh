#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Build the customer-facing Zephyr binary package for ns-cmsis-nn.

Usage:
  build_zephyr_release_package.sh --tag <tag> --outdir <dir> [--artifact-root <dir>]
                                  [--archs cortex-m0,cortex-m4+fp,cortex-m55]
                                  [--toolchains gcc,armclang,atfe]
                                  [--source-commit <sha>] [--source-ref <ref>]
                                  [--workflow-name <name>] [--workflow-run-id <id>]
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RELEASE_METADATA_HELPER="${REPO_ROOT}/scripts/release_package_metadata.py"

TAG=""
OUTDIR=""
ARTIFACT_ROOT="${REPO_ROOT}/out/release-artifacts"
ARCHS="cortex-m0,cortex-m4+fp,cortex-m55"
TOOLCHAINS="gcc,armclang,atfe"
SOURCE_COMMIT="${SOURCE_COMMIT:-local}"
SOURCE_REF="${SOURCE_REF:-local}"
WORKFLOW_NAME="${WORKFLOW_NAME:-}"
WORKFLOW_RUN_ID="${WORKFLOW_RUN_ID:-}"

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
    --source-commit)
      SOURCE_COMMIT="${2:?missing value for --source-commit}"
      shift 2
      ;;
    --source-ref)
      SOURCE_REF="${2:?missing value for --source-ref}"
      shift 2
      ;;
    --workflow-name)
      WORKFLOW_NAME="${2:?missing value for --workflow-name}"
      shift 2
      ;;
    --workflow-run-id)
      WORKFLOW_RUN_ID="${2:?missing value for --workflow-run-id}"
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
[[ -f "${RELEASE_METADATA_HELPER}" ]] || { echo "Release metadata helper not found: ${RELEASE_METADATA_HELPER}" >&2; exit 3; }

mkdir -p "${OUTDIR}"
OUTDIR="$(cd "${OUTDIR}" && pwd)"

PACKAGE_DIR="${OUTDIR}/ns-cmsis-nn-zephyr-binary-${TAG}"
ZIP_PATH="${OUTDIR}/ns-cmsis-nn-zephyr-binary-${TAG}.zip"
rm -rf "${PACKAGE_DIR}" "${ZIP_PATH}"

python3 "${RELEASE_METADATA_HELPER}" assemble-zephyr-binary-package \
  --artifact-root "${ARTIFACT_ROOT}" \
  --outdir "${OUTDIR}" \
  --tag "${TAG}" \
  --archs "${ARCHS}" \
  --toolchains "${TOOLCHAINS}" \
  --source-commit "${SOURCE_COMMIT}" \
  --source-ref "${SOURCE_REF}" \
  --workflow-name "${WORKFLOW_NAME}" \
  --workflow-run-id "${WORKFLOW_RUN_ID}"

python3 "${RELEASE_METADATA_HELPER}" write-checksums \
  --package-dir "${PACKAGE_DIR}" \
  --output "${PACKAGE_DIR}/checksums.txt"

(
  cd "${OUTDIR}"
  zip -rq "ns-cmsis-nn-zephyr-binary-${TAG}.zip" "ns-cmsis-nn-zephyr-binary-${TAG}"
)

echo "Wrote Zephyr binary package to ${OUTDIR}"

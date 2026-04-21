#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Build a Zephyr source-module release package for ns-cmsis-nn.

Usage:
  build_zephyr_release_package.sh --tag <tag> --outdir <dir>
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RELEASE_METADATA_HELPER="${REPO_ROOT}/scripts/release_package_metadata.py"

TAG=""
OUTDIR=""

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

for path in \
  "${REPO_ROOT}/zephyr/module.yml" \
  "${REPO_ROOT}/zephyr/CMakeLists.txt" \
  "${REPO_ROOT}/zephyr/Kconfig" \
  "${REPO_ROOT}/Documentation/zephyr_module_release.md"; do
  [[ -f "${path}" ]] || { echo "Required Zephyr release file not found: ${path}" >&2; exit 3; }
done
for dir in "${REPO_ROOT}/Source" "${REPO_ROOT}/Include"; do
  [[ -d "${dir}" ]] || { echo "Required Zephyr release directory not found: ${dir}" >&2; exit 3; }
done

mkdir -p "${OUTDIR}"
OUTDIR="$(cd "${OUTDIR}" && pwd)"

PACKAGE_DIR="${OUTDIR}/ns-cmsis-nn-zephyr-module-${TAG}"
ZIP_PATH="${OUTDIR}/ns-cmsis-nn-zephyr-module-${TAG}.zip"
rm -rf "${PACKAGE_DIR}" "${ZIP_PATH}"

mkdir -p "${PACKAGE_DIR}/zephyr"
cp -R "${REPO_ROOT}/Source" "${PACKAGE_DIR}/Source"
cp -R "${REPO_ROOT}/Include" "${PACKAGE_DIR}/Include"
cp "${REPO_ROOT}/zephyr/module.yml" "${PACKAGE_DIR}/zephyr/module.yml"
cp "${REPO_ROOT}/zephyr/CMakeLists.txt" "${PACKAGE_DIR}/zephyr/CMakeLists.txt"
cp "${REPO_ROOT}/zephyr/Kconfig" "${PACKAGE_DIR}/zephyr/Kconfig"
cp "${REPO_ROOT}/Documentation/zephyr_module_release.md" "${PACKAGE_DIR}/README-ZEPHYR.md"

python3 "${RELEASE_METADATA_HELPER}" write-zephyr-manifest \
  --tag "${TAG}" \
  --output "${PACKAGE_DIR}/MANIFEST.json"

(
  cd "${OUTDIR}"
  zip -rq "ns-cmsis-nn-zephyr-module-${TAG}.zip" "ns-cmsis-nn-zephyr-module-${TAG}"
)

echo "Wrote Zephyr source-module package to ${OUTDIR}"

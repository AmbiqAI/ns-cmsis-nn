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
RELEASE_METADATA_HELPER="${REPO_ROOT}/scripts/release_package_metadata.py"

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

BUNDLE_DIR="${OUTDIR}/ns-cmsis-nn-staticlibs-${TAG}"
ZIP_PATH="${OUTDIR}/ns-cmsis-nn-staticlibs-${TAG}.zip"
[[ -f "${RELEASE_METADATA_HELPER}" ]] || { echo "Release metadata helper not found: ${RELEASE_METADATA_HELPER}" >&2; exit 4; }
rm -rf "${BUNDLE_DIR}" "${ZIP_PATH}"

python3 "${RELEASE_METADATA_HELPER}" assemble-staticlib-bundle \
  --artifact-root "${ARTIFACT_ROOT}" \
  --outdir "${OUTDIR}" \
  --tag "${TAG}" \
  --archs "${ARCHS}" \
  --toolchains "${TOOLCHAINS}"

(
  cd "${OUTDIR}"
  zip -rq "ns-cmsis-nn-staticlibs-${TAG}.zip" "ns-cmsis-nn-staticlibs-${TAG}"
)

echo "Wrote release bundles to ${OUTDIR}"

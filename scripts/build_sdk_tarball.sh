#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
# SPDX-License-Identifier: Apache-2.0
#
# Package a per-arch ns-cmsis-nn SDK tarball: the prebuilt static library
# plus the public headers, license, and a manifest describing the build.
#
# Layout produced:
#
#   ns-cmsis-nn-<cpu>-<toolchain>-<version>/
#     include/
#       arm_nn_math_types.h
#       arm_nn_tables.h
#       arm_nn_types.h
#       arm_nnfunctions.h
#       arm_nnsupportfunctions.h
#       Internal/arm_nn_compiler.h
#     lib/
#       libns-cmsis-nn.a
#       libns-cmsis-nn.a.sha256
#     cmake/                     # populated by step 2 (find_package config)
#     LICENSE
#     manifest.json
#
# Usage:
#   build_sdk_tarball.sh --target-cpu         cortex-m{0,4,55} \
#                        --version            <X.Y.Z>          \
#                        --staticlib          <path-to-.a>     \
#                        --outdir             <dir>            \
#                        [--toolchain         gcc|atfe|armclang|clang] \
#                        [--toolchain-version <toolchain-build-id>]

set -euo pipefail

TARGET_CPU=""
VERSION=""
STATICLIB=""
OUTDIR=""
TOOLCHAIN="gcc"
TOOLCHAIN_VERSION=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target-cpu)        TARGET_CPU="$2";        shift 2 ;;
    --version)           VERSION="$2";           shift 2 ;;
    --staticlib)         STATICLIB="$2";         shift 2 ;;
    --outdir)            OUTDIR="$2";            shift 2 ;;
    --toolchain)         TOOLCHAIN="$2";         shift 2 ;;
    --toolchain-version) TOOLCHAIN_VERSION="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$TARGET_CPU" ]] || { echo "--target-cpu required" >&2; exit 2; }
[[ -n "$VERSION"    ]] || { echo "--version required"    >&2; exit 2; }
[[ -n "$STATICLIB"  ]] || { echo "--staticlib required"  >&2; exit 2; }
[[ -n "$OUTDIR"     ]] || { echo "--outdir required"     >&2; exit 2; }

case "$TARGET_CPU" in
  cortex-m0)  ARCH_FLAGS="-mcpu=cortex-m0 -mthumb -mfloat-abi=soft" ;;
  cortex-m4)  ARCH_FLAGS="-mcpu=cortex-m4 -mthumb -mfpu=fpv4-sp-d16 -mfloat-abi=hard" ;;
  cortex-m55) ARCH_FLAGS="-mcpu=cortex-m55 -mthumb -mfloat-abi=hard" ;;
  *) echo "unsupported --target-cpu '$TARGET_CPU'" >&2; exit 2 ;;
esac

# Toolchain id is baked into both the tarball name and the manifest so a
# future multi-toolchain matrix (M3 #177) is a pure additive change.
# TOOLCHAIN_COMPILER mirrors CMake's CMAKE_C_COMPILER_ID values so the
# find_package() config can fail-fast on consumer/package compiler skew.
case "$TOOLCHAIN" in
  gcc)      TOOLCHAIN_COMPILER="GNU";      TOOLCHAIN_FULL_ID="gnu-arm-embedded" ;;
  atfe)     TOOLCHAIN_COMPILER="Clang";    TOOLCHAIN_FULL_ID="arm-toolchain-for-embedded" ;;
  armclang) TOOLCHAIN_COMPILER="ARMClang"; TOOLCHAIN_FULL_ID="arm-compiler"     ;;
  clang)    TOOLCHAIN_COMPILER="Clang";    TOOLCHAIN_FULL_ID="llvm-embedded-toolchain-for-arm" ;;
  *) echo "unsupported --toolchain '$TOOLCHAIN' (gcc|atfe|armclang|clang)" >&2; exit 2 ;;
esac

case "$OUTDIR" in
  ""|"/"|"."|"..") echo "refusing unsafe --outdir '$OUTDIR'" >&2; exit 2 ;;
esac

[[ -f "$STATICLIB" ]] || { echo "staticlib not found: $STATICLIB" >&2; exit 2; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

mkdir -p "$OUTDIR"

PKG_NAME="ns-cmsis-nn-${TARGET_CPU}-${TOOLCHAIN}-${VERSION}"
STAGE="$(mktemp -d "$OUTDIR/.stage.XXXXXX")"
trap 'rm -rf "$STAGE"' EXIT

PKG_ROOT="$STAGE/$PKG_NAME"
mkdir -p "$PKG_ROOT/include" "$PKG_ROOT/lib" "$PKG_ROOT/cmake"

# Headers
cp -a "$REPO_ROOT/Include/." "$PKG_ROOT/include/"

# Library — rename to canonical name inside the tarball; consumers don't
# need to know the per-arch suffix once they have a per-arch tarball.
cp "$STATICLIB" "$PKG_ROOT/lib/libns-cmsis-nn.a"
( cd "$PKG_ROOT/lib" && sha256sum libns-cmsis-nn.a > libns-cmsis-nn.a.sha256 )

# License
cp "$REPO_ROOT/LICENSE" "$PKG_ROOT/LICENSE"

# Manifest
lib_sha256="$(awk '{print $1}' "$PKG_ROOT/lib/libns-cmsis-nn.a.sha256")"
lib_size="$(stat -c %s "$PKG_ROOT/lib/libns-cmsis-nn.a" 2>/dev/null || stat -f %z "$PKG_ROOT/lib/libns-cmsis-nn.a")"
built_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# CMake package config — render @VAR@ placeholders into the staged tree.
render_template() {
  local src="$1" dst="$2"
  sed -e "s|@NS_CMSIS_NN_VERSION@|${VERSION}|g" \
      -e "s|@NS_CMSIS_NN_TARGET_CPU@|${TARGET_CPU}|g" \
      -e "s|@NS_CMSIS_NN_ARCH_FLAGS@|${ARCH_FLAGS}|g" \
      -e "s|@NS_CMSIS_NN_TOOLCHAIN@|${TOOLCHAIN}|g" \
      -e "s|@NS_CMSIS_NN_TOOLCHAIN_COMPILER@|${TOOLCHAIN_COMPILER}|g" \
      "$src" > "$dst"
}
render_template "$REPO_ROOT/cmake/templates/ns-cmsis-nn-config.cmake.in" \
                "$PKG_ROOT/cmake/ns-cmsis-nn-config.cmake"
render_template "$REPO_ROOT/cmake/templates/ns-cmsis-nn-config-version.cmake.in" \
                "$PKG_ROOT/cmake/ns-cmsis-nn-config-version.cmake"

cat > "$PKG_ROOT/manifest.json" <<EOF
{
  "schema_version": 1,
  "package": "ns-cmsis-nn",
  "version": "${VERSION}",
  "target_cpu": "${TARGET_CPU}",
  "toolchain": {
    "id": "${TOOLCHAIN}",
    "full_id": "${TOOLCHAIN_FULL_ID}",
    "compiler_id": "${TOOLCHAIN_COMPILER}",
    "version": "${TOOLCHAIN_VERSION}"
  },
  "abi": {
    "arch_flags": "${ARCH_FLAGS}"
  },
  "library": {
    "name": "libns-cmsis-nn.a",
    "sha256": "${lib_sha256}",
    "size_bytes": ${lib_size}
  },
  "built_at": "${built_at}"
}
EOF

# Tar it up. Use --owner=0 --group=0 for reproducibility-friendliness.
TARBALL="$OUTDIR/${PKG_NAME}.tar.gz"
tar -C "$STAGE" \
    --owner=0 --group=0 --sort=name \
    --mtime="${built_at}" \
    -czf "$TARBALL" "$PKG_NAME"
( cd "$OUTDIR" && sha256sum "${PKG_NAME}.tar.gz" > "${PKG_NAME}.tar.gz.sha256" )

echo ">>> packaged ${TARBALL}"
ls -la "$TARBALL" "$TARBALL.sha256"

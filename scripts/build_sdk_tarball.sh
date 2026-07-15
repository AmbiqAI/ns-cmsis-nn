#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
# SPDX-License-Identifier: Apache-2.0
#
# Package a per-arch ns-cmsis-nn SDK tarball: the prebuilt static library
# plus the public headers, license, and a manifest describing the build.
#
# The archive packaged here is a superset build: integer kernels are
# always present, and F32/F16 are included whenever the static library
# was configured with them. This script never creates separate int/float
# "variant" tarballs — one canonical archive per cpu/toolchain.
#
# Feature metadata ("features" in manifest.json, and the NS_CMSIS_NN_HAS_*
# variables rendered into cmake/ns-cmsis-nn-config.cmake) is sourced from
# the "<staticlib>.buildconfig.json" sidecar that build_staticlib.sh writes
# next to the archive it produces. That sidecar is the single source of
# truth for what was actually compiled in, so the manifest can never drift
# from the real archive contents. If a sidecar is not found (e.g. a
# hand-supplied .a with no sidecar), this script falls back to the
# explicit --enable-f32/--enable-f16/--enable-requantize-inline-asm flags
# passed on this script's own command line, and marks the manifest's
# feature metadata as unverified.
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
#                        [--toolchain-version <toolchain-build-id>] \
#                        [--enable-f32] [--enable-f16] \
#                        [--enable-requantize-inline-asm]
#
# --enable-f32/--enable-f16/--enable-requantize-inline-asm are only
# consulted as a fallback when "<staticlib>.buildconfig.json" is absent;
# when the sidecar is present its values are authoritative and these
# flags are cross-checked against it (a mismatch is a configuration bug
# and aborts the packaging step, rather than silently trusting whichever
# value differs).

set -euo pipefail

TARGET_CPU=""
VERSION=""
STATICLIB=""
OUTDIR=""
TOOLCHAIN="gcc"
TOOLCHAIN_VERSION=""
CLI_ENABLE_F32=0
CLI_ENABLE_F16=0
CLI_ENABLE_REQUANTIZE_INLINE_ASM=0
CLI_F32_SET=0
CLI_F16_SET=0
CLI_REQUANTIZE_SET=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target-cpu)        TARGET_CPU="$2";        shift 2 ;;
    --version)           VERSION="$2";           shift 2 ;;
    --staticlib)         STATICLIB="$2";         shift 2 ;;
    --outdir)            OUTDIR="$2";            shift 2 ;;
    --toolchain)         TOOLCHAIN="$2";         shift 2 ;;
    --toolchain-version) TOOLCHAIN_VERSION="$2"; shift 2 ;;
    --enable-f32) CLI_ENABLE_F32=1; CLI_F32_SET=1; shift ;;
    --enable-f16) CLI_ENABLE_F16=1; CLI_F16_SET=1; shift ;;
    --enable-requantize-inline-asm)
      CLI_ENABLE_REQUANTIZE_INLINE_ASM=1; CLI_REQUANTIZE_SET=1; shift ;;
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

# ----------------------------------------------------------------------------
# Resolve feature metadata: prefer the buildconfig.json sidecar written by
# build_staticlib.sh (single source of truth for what's actually in the
# archive); fall back to the CLI flags above, with a warning, when no
# sidecar is present.
# ----------------------------------------------------------------------------
BUILDCONFIG_SIDECAR="${STATICLIB}.buildconfig.json"
FEATURES_VERIFIED="false"

if [[ -f "$BUILDCONFIG_SIDECAR" ]]; then
  command -v jq >/dev/null || { echo "jq required to read $BUILDCONFIG_SIDECAR" >&2; exit 3; }
  SIDECAR_F32="$(jq -r '.arm_nn_enable_f32' "$BUILDCONFIG_SIDECAR")"
  SIDECAR_F16="$(jq -r '.arm_nn_enable_f16' "$BUILDCONFIG_SIDECAR")"
  SIDECAR_REQUANTIZE="$(jq -r '.cmsis_nn_use_requantize_inline_assembly' "$BUILDCONFIG_SIDECAR")"

  # Cross-check any explicitly-passed CLI flags against the sidecar: a
  # mismatch means the caller thinks the archive was built one way but it
  # was actually built another way, which is exactly the metadata drift
  # this design is meant to prevent.
  if [[ "$CLI_F32_SET" == "1" ]]; then
    cli_f32_bool="$([[ "$CLI_ENABLE_F32" == "1" ]] && echo true || echo false)"
    if [[ "$cli_f32_bool" != "$SIDECAR_F32" ]]; then
      echo "F32 mismatch: --enable-f32 says ${cli_f32_bool} but ${BUILDCONFIG_SIDECAR} says ${SIDECAR_F32}" >&2
      exit 2
    fi
  fi
  if [[ "$CLI_F16_SET" == "1" ]]; then
    cli_f16_bool="$([[ "$CLI_ENABLE_F16" == "1" ]] && echo true || echo false)"
    if [[ "$cli_f16_bool" != "$SIDECAR_F16" ]]; then
      echo "F16 mismatch: --enable-f16 says ${cli_f16_bool} but ${BUILDCONFIG_SIDECAR} says ${SIDECAR_F16}" >&2
      exit 2
    fi
  fi
  if [[ "$CLI_REQUANTIZE_SET" == "1" ]]; then
    cli_requantize_bool="$([[ "$CLI_ENABLE_REQUANTIZE_INLINE_ASM" == "1" ]] && echo true || echo false)"
    if [[ "$cli_requantize_bool" != "$SIDECAR_REQUANTIZE" ]]; then
      echo "requantize-inline-asm mismatch: --enable-requantize-inline-asm says ${cli_requantize_bool}" \
           "but ${BUILDCONFIG_SIDECAR} says ${SIDECAR_REQUANTIZE}" >&2
      exit 2
    fi
  fi

  FEATURE_F32="$SIDECAR_F32"
  FEATURE_F16="$SIDECAR_F16"
  FEATURE_REQUANTIZE="$SIDECAR_REQUANTIZE"
  FEATURES_VERIFIED="true"
  echo ">>> feature metadata sourced from ${BUILDCONFIG_SIDECAR} (verified)"
else
  FEATURE_F32="$([[ "$CLI_ENABLE_F32" == "1" ]] && echo true || echo false)"
  FEATURE_F16="$([[ "$CLI_ENABLE_F16" == "1" ]] && echo true || echo false)"
  FEATURE_REQUANTIZE="$([[ "$CLI_ENABLE_REQUANTIZE_INLINE_ASM" == "1" ]] && echo true || echo false)"
  echo ">>> WARNING: no ${BUILDCONFIG_SIDECAR} found; feature metadata is" \
       "UNVERIFIED and taken from CLI flags only (F32=${FEATURE_F32} F16=${FEATURE_F16}" \
       "requantize-inline-asm=${FEATURE_REQUANTIZE})" >&2
fi

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
      -e "s|@NS_CMSIS_NN_HAS_F32@|${FEATURE_F32}|g" \
      -e "s|@NS_CMSIS_NN_HAS_F16@|${FEATURE_F16}|g" \
      -e "s|@NS_CMSIS_NN_HAS_REQUANTIZE_INLINE_ASM@|${FEATURE_REQUANTIZE}|g" \
      "$src" > "$dst"
}
render_template "$REPO_ROOT/cmake/templates/ns-cmsis-nn-config.cmake.in" \
                "$PKG_ROOT/cmake/ns-cmsis-nn-config.cmake"
render_template "$REPO_ROOT/cmake/templates/ns-cmsis-nn-config-version.cmake.in" \
                "$PKG_ROOT/cmake/ns-cmsis-nn-config-version.cmake"

#
# schema_version 2 adds the "features" block below. Consumers reading an
# older (schema_version 1) manifest with no "features" key MUST treat the
# archive as integer-only (f32/f16/requantize_inline_asm all false) — this
# is the documented, conservative fallback for pre-existing packages built
# before float support existed.
cat > "$PKG_ROOT/manifest.json" <<EOF
{
  "schema_version": 2,
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
  "features": {
    "integer": true,
    "f32": ${FEATURE_F32},
    "f16": ${FEATURE_F16},
    "requantize_inline_asm": ${FEATURE_REQUANTIZE},
    "verified": ${FEATURES_VERIFIED}
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

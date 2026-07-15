#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
# SPDX-License-Identifier: Apache-2.0
#
# Build a single prebuilt cmsis-nn static library for one
# target CPU, strip it, sha256 it, and emit the final artefact at the
# canonical release name.
#
# Usage:
#   build_staticlib.sh --target-cpu cortex-m{0,4,55} \
#                      --version    <X.Y.Z>          \
#                      --outdir     <dir>             \
#                      [--toolchain gcc|atfe|armclang] \
#                      [--toolchain-root <dir>]        \
#                      [--enable-f32] [--enable-f16]   \
#                      [--enable-requantize-inline-asm]
#
# --enable-f32/--enable-f16/--enable-requantize-inline-asm pass explicit
# -DARM_NN_ENABLE_F32/F16=ON and -DCMSIS_NN_USE_REQUANTIZE_INLINE_ASSEMBLY=ON
# into the configure step — the archive's contents are always driven by
# what is explicitly requested here, never by CMake option() defaults.
# This project publishes F16 support for cortex-m55 only (matches the
# ARMV8_1_M_MVEF Kconfig restriction in zephyr/Kconfig); --enable-f16 is
# rejected for any other --target-cpu.
#
# A "<final_name>.buildconfig.json" sidecar is written next to the
# archive recording exactly which flags produced it. This is the single
# source of truth build_sdk_tarball.sh reads to populate the SDK
# manifest's "features" block, so the manifest can never drift from what
# was actually compiled into the archive.
#
# Outputs in <outdir>:
#   gcc:               libns-cmsis-nn-<target-cpu>-<version>.a(.buildconfig.json)
#   atfe / armclang:   libns-cmsis-nn-<toolchain>-<target-cpu>-<version>.a(.buildconfig.json)

set -euo pipefail

TARGET_CPU=""
VERSION=""
OUTDIR=""
TOOLCHAIN="gcc"
TOOLCHAIN_ROOT=""
ENABLE_F32=0
ENABLE_F16=0
ENABLE_REQUANTIZE_INLINE_ASM=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target-cpu) TARGET_CPU="${2:?}"; shift 2 ;;
    --version)    VERSION="${2:?}";    shift 2 ;;
    --outdir)     OUTDIR="${2:?}";     shift 2 ;;
    --toolchain)  TOOLCHAIN="${2:?}";  shift 2 ;;
    --toolchain-root) TOOLCHAIN_ROOT="${2-}"; shift 2 ;;
    --enable-f32) ENABLE_F32=1; shift ;;
    --enable-f16) ENABLE_F16=1; shift ;;
    --enable-requantize-inline-asm) ENABLE_REQUANTIZE_INLINE_ASM=1; shift ;;
    -h|--help)
      sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

for v in TARGET_CPU VERSION OUTDIR; do
  if [[ -z "${!v}" ]]; then
    echo "$v is required" >&2
    exit 2
  fi
done

case "${TARGET_CPU}" in
  cortex-m0|cortex-m4|cortex-m55) ;;
  *)
    echo "unsupported target-cpu '${TARGET_CPU}' (expect cortex-m{0,4,55})" >&2
    exit 2
    ;;
esac

if [[ "${ENABLE_F16}" == "1" && "${TARGET_CPU}" != "cortex-m55" ]]; then
  echo "--enable-f16 is only qualified for cortex-m55 (got --target-cpu ${TARGET_CPU});" \
       "this matches the ARMV8_1_M_MVEF restriction in zephyr/Kconfig." >&2
  exit 2
fi

case "${OUTDIR}" in
  /|"."|"..")
    echo "refusing to use unsafe OUTDIR: ${OUTDIR}" >&2
    exit 2
    ;;
esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
case "${TOOLCHAIN}" in
  gcc)
    toolchain_file="${repo_root}/cmake/toolchain/arm-none-eabi-gcc.cmake"
    strip_command="arm-none-eabi-strip"
    command -v arm-none-eabi-gcc   >/dev/null || { echo "arm-none-eabi-gcc not on PATH"   >&2; exit 3; }
    command -v arm-none-eabi-strip >/dev/null || { echo "arm-none-eabi-strip not on PATH" >&2; exit 3; }
    compiler_version="$(arm-none-eabi-gcc --version | head -n 1)"
    ;;
  atfe)
    [[ -n "${TOOLCHAIN_ROOT}" ]] || TOOLCHAIN_ROOT="${NS_CMSIS_NN_TOOLCHAIN_ROOT:-${NS_CMSIS_NN_ATFE_ROOT:-}}"
    [[ -n "${TOOLCHAIN_ROOT}" ]] || { echo "--toolchain-root required for atfe" >&2; exit 3; }
    TOOLCHAIN_ROOT="$(cd "${TOOLCHAIN_ROOT}" && pwd)"
    toolchain_file="${repo_root}/cmake/toolchain/atfe.cmake"
    strip_command="${TOOLCHAIN_ROOT}/bin/llvm-strip"
    compiler_version="$(${TOOLCHAIN_ROOT}/bin/clang --version | head -n 1)"
    export NS_CMSIS_NN_TOOLCHAIN_ROOT="${TOOLCHAIN_ROOT}"
    ;;
  armclang)
    [[ -n "${TOOLCHAIN_ROOT}" ]] || TOOLCHAIN_ROOT="${NS_CMSIS_NN_TOOLCHAIN_ROOT:-${NS_CMSIS_NN_ARMCLANG_ROOT:-}}"
    [[ -n "${TOOLCHAIN_ROOT}" ]] || { echo "--toolchain-root required for armclang" >&2; exit 3; }
    TOOLCHAIN_ROOT="$(cd "${TOOLCHAIN_ROOT}" && pwd)"
    toolchain_file="${repo_root}/cmake/toolchain/armclang.cmake"
    strip_command="$(command -v llvm-strip || true)"
    compiler_version="$(${TOOLCHAIN_ROOT}/bin/armclang --version | head -n 1)"
    export NS_CMSIS_NN_TOOLCHAIN_ROOT="${TOOLCHAIN_ROOT}"
    ;;
  *)
    echo "unsupported toolchain '${TOOLCHAIN}' (expect gcc|atfe|armclang)" >&2
    exit 2
    ;;
esac

[[ -f "${toolchain_file}" ]] || { echo "missing toolchain: ${toolchain_file}" >&2; exit 3; }
command -v sha256sum >/dev/null || { echo "sha256sum not on PATH" >&2; exit 3; }

mkdir -p "${OUTDIR}"
OUTDIR="$(cd "${OUTDIR}" && pwd)"

build_dir="$(mktemp -d "${OUTDIR}/.build.XXXXXX")"
trap 'rm -rf "${build_dir}"' EXIT

generator="Unix Makefiles"
command -v ninja >/dev/null && generator="Ninja"

echo ">>> configuring (${TOOLCHAIN}, ${TARGET_CPU}, F32=${ENABLE_F32} F16=${ENABLE_F16} requantize-asm=${ENABLE_REQUANTIZE_INLINE_ASM})"
cmake -S "${repo_root}" -B "${build_dir}" -G "${generator}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="${toolchain_file}" \
  -DNS_CMSIS_NN_TOOLCHAIN_ROOT="${TOOLCHAIN_ROOT}" \
  -DNS_CMSIS_NN_TARGET_CPU="${TARGET_CPU}" \
  -DARM_NN_ENABLE_F32="$([[ "${ENABLE_F32}" == "1" ]] && echo ON || echo OFF)" \
  -DARM_NN_ENABLE_F16="$([[ "${ENABLE_F16}" == "1" ]] && echo ON || echo OFF)" \
  -DCMSIS_NN_USE_REQUANTIZE_INLINE_ASSEMBLY="$([[ "${ENABLE_REQUANTIZE_INLINE_ASM}" == "1" ]] && echo ON || echo OFF)"

echo ">>> building"
cmake --build "${build_dir}" --target cmsis-nn --parallel

src_lib="${build_dir}/libcmsis-nn.a"
[[ -f "${src_lib}" ]] || { echo "expected archive not produced: ${src_lib}" >&2; exit 4; }

if [[ "${TOOLCHAIN}" == "gcc" ]]; then
  final_name="libns-cmsis-nn-${TARGET_CPU}-${VERSION}.a"
else
  final_name="libns-cmsis-nn-${TOOLCHAIN}-${TARGET_CPU}-${VERSION}.a"
fi
final_path="${OUTDIR}/${final_name}"

cp "${src_lib}" "${final_path}"
if [[ -n "${strip_command}" && -x "${strip_command}" ]]; then
  "${strip_command}" --strip-debug "${final_path}"
else
  echo ">>> strip command unavailable for ${TOOLCHAIN}; leaving archive unstripped"
fi

( cd "${OUTDIR}" && sha256sum "${final_name}" > "${final_name}.sha256" )

# Sidecar: single source of truth for what was actually compiled into this
# archive. build_sdk_tarball.sh reads this to populate the SDK manifest's
# "features" block, so the manifest can never claim a feature that wasn't
# actually passed into this configure step.
buildconfig_path="${final_path}.buildconfig.json"
cat > "${buildconfig_path}" <<EOF
{
  "target_cpu": "${TARGET_CPU}",
  "toolchain": "${TOOLCHAIN}",
  "version": "${VERSION}",
  "arm_nn_enable_f32": $([[ "${ENABLE_F32}" == "1" ]] && echo true || echo false),
  "arm_nn_enable_f16": $([[ "${ENABLE_F16}" == "1" ]] && echo true || echo false),
  "cmsis_nn_use_requantize_inline_assembly": $([[ "${ENABLE_REQUANTIZE_INLINE_ASM}" == "1" ]] && echo true || echo false)
}
EOF

echo ">>> done: ${final_path}"
echo ">>> compiler: ${compiler_version}"
ls -lh "${final_path}" "${final_path}.sha256" "${buildconfig_path}"

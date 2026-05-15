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
#                      --outdir     <dir>
#
# Outputs in <outdir>:
#   libns-cmsis-nn-<target-cpu>-<version>.a
#   libns-cmsis-nn-<target-cpu>-<version>.a.sha256
#
# Single toolchain on purpose: GNU Arm Embedded (arm-none-eabi-gcc).
# Consumers needing armclang/ATFE builds are welcome to compile from
# source.

set -euo pipefail

TARGET_CPU=""
VERSION=""
OUTDIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target-cpu) TARGET_CPU="${2:?}"; shift 2 ;;
    --version)    VERSION="${2:?}";    shift 2 ;;
    --outdir)     OUTDIR="${2:?}";     shift 2 ;;
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

case "${OUTDIR}" in
  /|"."|"..")
    echo "refusing to use unsafe OUTDIR: ${OUTDIR}" >&2
    exit 2
    ;;
esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
toolchain_file="${repo_root}/cmake/toolchain/arm-none-eabi-gcc.cmake"
[[ -f "${toolchain_file}" ]] || { echo "missing toolchain: ${toolchain_file}" >&2; exit 3; }

command -v arm-none-eabi-gcc   >/dev/null || { echo "arm-none-eabi-gcc not on PATH"   >&2; exit 3; }
command -v arm-none-eabi-strip >/dev/null || { echo "arm-none-eabi-strip not on PATH" >&2; exit 3; }
command -v sha256sum           >/dev/null || { echo "sha256sum not on PATH"           >&2; exit 3; }

mkdir -p "${OUTDIR}"
OUTDIR="$(cd "${OUTDIR}" && pwd)"

build_dir="$(mktemp -d "${OUTDIR}/.build.XXXXXX")"
trap 'rm -rf "${build_dir}"' EXIT

generator="Unix Makefiles"
command -v ninja >/dev/null && generator="Ninja"

echo ">>> configuring (${TARGET_CPU})"
cmake -S "${repo_root}" -B "${build_dir}" -G "${generator}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="${toolchain_file}" \
  -DNS_CMSIS_NN_TARGET_CPU="${TARGET_CPU}"

echo ">>> building"
cmake --build "${build_dir}" --target cmsis-nn --parallel

src_lib="${build_dir}/libcmsis-nn.a"
[[ -f "${src_lib}" ]] || { echo "expected archive not produced: ${src_lib}" >&2; exit 4; }

final_name="libns-cmsis-nn-${TARGET_CPU}-${VERSION}.a"
final_path="${OUTDIR}/${final_name}"

cp "${src_lib}" "${final_path}"
arm-none-eabi-strip --strip-debug "${final_path}"

( cd "${OUTDIR}" && sha256sum "${final_name}" > "${final_name}.sha256" )

echo ">>> done: ${final_path}"
ls -lh "${final_path}" "${final_path}.sha256"

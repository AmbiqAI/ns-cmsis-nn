#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
# SPDX-License-Identifier: Apache-2.0
#
# Smoke-link verification for a prebuilt cmsis-nn static library.
#
# Compiles scripts/smoke/staticlib_smoke.c against the given archive
# with the matching architecture flags. GCC and ATfE produce an ELF with
# a minimal bare-metal link; armclang uses a compile-plus-archive-symbol
# check because Arm's linker option dialect is intentionally different.
#
# Failure here means the published archive is missing symbols a
# consumer would expect, or its arch flags are mismatched.
#
# Usage:
#   smoke_staticlib.sh --target-cpu cortex-m{0,4,55} \
#                      --library <path/to/libns-cmsis-nn-*.a> \
#                      --outdir <dir> \
#                      [--toolchain gcc|atfe|armclang] \
#                      [--toolchain-root <dir>]

set -euo pipefail

TARGET_CPU=""
LIBRARY=""
OUTDIR=""
TOOLCHAIN="gcc"
TOOLCHAIN_ROOT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target-cpu) TARGET_CPU="${2:?}"; shift 2 ;;
    --library)    LIBRARY="${2:?}";    shift 2 ;;
    --outdir)     OUTDIR="${2:?}";     shift 2 ;;
    --toolchain)  TOOLCHAIN="${2:?}";  shift 2 ;;
    --toolchain-root) TOOLCHAIN_ROOT="${2:?}"; shift 2 ;;
    *)
      echo "unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

for v in TARGET_CPU LIBRARY OUTDIR; do
  if [[ -z "${!v}" ]]; then echo "$v required" >&2; exit 2; fi
done

case "${TARGET_CPU}" in
  cortex-m0)  arch_flags=(-mcpu=cortex-m0  -mthumb -mfloat-abi=soft) ;;
  cortex-m4)  arch_flags=(-mcpu=cortex-m4  -mthumb -mfpu=fpv4-sp-d16 -mfloat-abi=hard) ;;
  cortex-m55) arch_flags=(-mcpu=cortex-m55 -mthumb -mfloat-abi=hard) ;;
  *)
    echo "unsupported target-cpu '${TARGET_CPU}'" >&2
    exit 2
    ;;
esac

[[ -f "${LIBRARY}" ]] || { echo "library not found: ${LIBRARY}" >&2; exit 3; }

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
smoke_src="${repo_root}/scripts/smoke/staticlib_smoke.c"
[[ -f "${smoke_src}" ]] || { echo "smoke source missing: ${smoke_src}" >&2; exit 3; }

mkdir -p "${OUTDIR}"
OUTDIR="$(cd "${OUTDIR}" && pwd)"

elf="${OUTDIR}/staticlib_smoke_${TOOLCHAIN}_${TARGET_CPU}.elf"
obj="${OUTDIR}/staticlib_smoke_${TOOLCHAIN}_${TARGET_CPU}.o"
check_mode="link"

case "${TOOLCHAIN}" in
  gcc)
    compiler="arm-none-eabi-gcc"
    nm="arm-none-eabi-nm"
    size="arm-none-eabi-size"
    link_flags=(-nostartfiles --specs=nosys.specs)
    command -v "${compiler}" >/dev/null || { echo "${compiler} not on PATH" >&2; exit 3; }
    command -v "${nm}"       >/dev/null || { echo "${nm} not on PATH"       >&2; exit 3; }
    command -v "${size}"     >/dev/null || { echo "${size} not on PATH"     >&2; exit 3; }
    ;;
  atfe)
    [[ -n "${TOOLCHAIN_ROOT}" ]] || TOOLCHAIN_ROOT="${NS_CMSIS_NN_TOOLCHAIN_ROOT:-${NS_CMSIS_NN_ATFE_ROOT:-}}"
    [[ -n "${TOOLCHAIN_ROOT}" ]] || { echo "--toolchain-root required for atfe" >&2; exit 3; }
    compiler="${TOOLCHAIN_ROOT}/bin/clang"
    nm="${TOOLCHAIN_ROOT}/bin/llvm-nm"
    size="${TOOLCHAIN_ROOT}/bin/llvm-size"
    link_flags=(--target=arm-none-eabi -nostartfiles -nostdlib)
    ;;
  armclang)
    [[ -n "${TOOLCHAIN_ROOT}" ]] || TOOLCHAIN_ROOT="${NS_CMSIS_NN_TOOLCHAIN_ROOT:-${NS_CMSIS_NN_ARMCLANG_ROOT:-}}"
    [[ -n "${TOOLCHAIN_ROOT}" ]] || { echo "--toolchain-root required for armclang" >&2; exit 3; }
    compiler="${TOOLCHAIN_ROOT}/bin/armclang"
    nm="$(command -v llvm-nm || true)"
    size="$(command -v llvm-size || true)"
    arch_flags=(--target=arm-arm-none-eabi "${arch_flags[@]}")
    link_flags=(-nostartfiles -nostdlib)
    check_mode="archive-symbols"
    ;;
  *)
    echo "unsupported toolchain '${TOOLCHAIN}' (expect gcc|atfe|armclang)" >&2
    exit 2
    ;;
esac

[[ -x "${compiler}" || -n "$(command -v "${compiler}" 2>/dev/null || true)" ]] || { echo "compiler not found: ${compiler}" >&2; exit 3; }
[[ -n "${nm}" ]] || { echo "llvm-nm not found for ${TOOLCHAIN}" >&2; exit 3; }
[[ -n "${size}" ]] || { echo "llvm-size not found for ${TOOLCHAIN}" >&2; exit 3; }

if [[ "${check_mode}" == "link" ]]; then
  echo ">>> smoke-linking ${TOOLCHAIN}/${TARGET_CPU} against $(basename "${LIBRARY}")"
  "${compiler}" \
    "${arch_flags[@]}" \
    "${link_flags[@]}" \
    -Wl,--gc-sections \
    -Wl,--entry=ns_cmsis_nn_smoke_refs \
    -Wl,--unresolved-symbols=ignore-all \
    -Wl,--whole-archive "${LIBRARY}" -Wl,--no-whole-archive \
    -o "${elf}" \
    "${smoke_src}"
  nm_input="${elf}"
else
  echo ">>> smoke-checking ${TOOLCHAIN}/${TARGET_CPU} against $(basename "${LIBRARY}")"
  "${compiler}" "${arch_flags[@]}" -c "${smoke_src}" -o "${obj}"
  nm_input="${LIBRARY}"
fi

# Sanity: ELF must contain at least one symbol from each referenced group.
required_syms=(
  arm_relu_q7
  arm_softmax_s8
  arm_convolve_s8
  arm_fully_connected_s8
  arm_max_pool_s8
  arm_avgpool_s8
  arm_elementwise_add_s8
  arm_elementwise_mul_s8
)
missing=()
nm_out="$(${nm} --defined-only "${nm_input}")"
for s in "${required_syms[@]}"; do
  if ! grep -qE "[[:space:]]${s}\$" <<<"${nm_out}"; then
    missing+=("${s}")
  fi
done
if (( ${#missing[@]} > 0 )); then
  echo "smoke check is missing expected symbols:" >&2
  printf '  %s\n' "${missing[@]}" >&2
  exit 4
fi

if [[ "${check_mode}" == "link" ]]; then
  size_out="$(${size} "${elf}")"
  echo ">>> ${elf}"
  echo "${size_out}"
else
  bytes="$(stat -c %s "${LIBRARY}" 2>/dev/null || stat -f %z "${LIBRARY}")"
  echo ">>> ${obj}"
  echo ">>> archive bytes: ${bytes}"
fi
echo ">>> smoke OK (${TOOLCHAIN}, ${TARGET_CPU})"

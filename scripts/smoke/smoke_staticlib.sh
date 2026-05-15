#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
# SPDX-License-Identifier: Apache-2.0
#
# Smoke-link verification for a prebuilt cmsis-nn static library.
#
# Compiles scripts/smoke/staticlib_smoke.c against the given archive
# with the matching architecture flags and produces an ELF. We use a
# minimal hosted-bare-metal link (`-nostartfiles` + `--specs=nosys.specs`)
# so we never need a board's startup/linker glue — we just want the
# linker to resolve our symbol references.
#
# Failure here means the published archive is missing symbols a
# consumer would expect, or its arch flags are mismatched.
#
# Usage:
#   smoke_staticlib.sh --target-cpu cortex-m{0,4,55} \
#                      --library <path/to/libns-cmsis-nn-*.a> \
#                      --outdir <dir>

set -euo pipefail

TARGET_CPU=""
LIBRARY=""
OUTDIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target-cpu) TARGET_CPU="${2:?}"; shift 2 ;;
    --library)    LIBRARY="${2:?}";    shift 2 ;;
    --outdir)     OUTDIR="${2:?}";     shift 2 ;;
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

elf="${OUTDIR}/staticlib_smoke_${TARGET_CPU}.elf"

echo ">>> smoke-linking ${TARGET_CPU} against $(basename "${LIBRARY}")"
arm-none-eabi-gcc \
  "${arch_flags[@]}" \
  -nostartfiles \
  --specs=nosys.specs \
  -Wl,--gc-sections \
  -Wl,--entry=ns_cmsis_nn_smoke_refs \
  -Wl,--whole-archive "${LIBRARY}" -Wl,--no-whole-archive \
  -o "${elf}" \
  "${smoke_src}"

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
nm_out="$(arm-none-eabi-nm --defined-only "${elf}")"
for s in "${required_syms[@]}"; do
  if ! grep -qE "[[:space:]]${s}\$" <<<"${nm_out}"; then
    missing+=("${s}")
  fi
done
if (( ${#missing[@]} > 0 )); then
  echo "smoke ELF is missing expected symbols:" >&2
  printf '  %s\n' "${missing[@]}" >&2
  exit 4
fi

size_out="$(arm-none-eabi-size "${elf}")"
echo ">>> ${elf}"
echo "${size_out}"
echo ">>> smoke OK (${TARGET_CPU})"

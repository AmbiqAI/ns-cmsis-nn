#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
# SPDX-License-Identifier: Apache-2.0
#
# Smoke-link verification for a prebuilt cmsis-nn static library.
#
# Compiles scripts/smoke/staticlib_smoke.c against the given archive
# with the matching architecture flags and links a minimal bare-metal
# ELF. All three toolchains link for real; armclang goes through armlink,
# whose option dialect differs from the GNU/LLD one (see link_style).
#
# Failure here means the published archive is missing symbols a
# consumer would expect, or its arch flags are mismatched.
#
# Usage:
#   smoke_staticlib.sh --target-cpu cortex-m{0,4,55} \
#                      --library <path/to/libns-cmsis-nn-*.a> \
#                      --outdir <dir> \
#                      [--toolchain gcc|atfe|armclang] \
#                      [--toolchain-root <dir>] \
#                      [--no-strict]
#
# The link is STRICT by default: every object in the archive must
# resolve. Previously this check passed --gc-sections and
# --unresolved-symbols=ignore-all, which meant kernels the smoke source
# does not call were discarded, or their undefined symbols ignored,
# before the linker ever had to resolve them. That made the check green
# on an archive containing `__ARM_undef` -- the symbol older GCCs emit
# when an MVE _Generic intrinsic fails to dispatch
# (AmbiqAI/ns-cmsis-nn#305) -- which no consumer could ever supply.
#
# --no-strict restores the old lenient behaviour. It is an escape hatch
# for local debugging; no CI leg uses it. As of the strict-by-default
# change every gcc and atfe leg (cortex-m0/m4/m55) links clean, so any
# new strict failure is a real unresolved symbol, not a known exemption.
#
# armclang used to be exempt: its check_mode was "archive-symbols", which
# ran `nm --defined-only` over the .a instead of invoking a linker, so it
# could not observe a broken archive symbol index or an unresolved
# reference at all (AmbiqAI/ns-cmsis-nn#291). It now performs a real
# link. Two dialect differences are handled explicitly:
#
#   whole-archive  armlink has no --whole-archive. Every member is
#                  extracted and placed on the link line instead, which
#                  is the exact equivalent and keeps the guarantee that
#                  EVERY object must resolve -- not just the ones the
#                  smoke source happens to reference.
#   dead-strip     armlink's unused-section removal is on by default;
#                  --no_remove is the analogue of omitting --gc-sections.
#
# armlink already treats an unresolved reference as an error (L6218E),
# so strictness needs no extra flag there -- only the two above, which
# stop the link from quietly discarding the very objects under test.

set -euo pipefail

TARGET_CPU=""
LIBRARY=""
OUTDIR=""
TOOLCHAIN="gcc"
TOOLCHAIN_ROOT=""
STRICT=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target-cpu) TARGET_CPU="${2:?}"; shift 2 ;;
    --library)    LIBRARY="${2:?}";    shift 2 ;;
    --outdir)     OUTDIR="${2:?}";     shift 2 ;;
    --toolchain)  TOOLCHAIN="${2:?}";  shift 2 ;;
    --toolchain-root) TOOLCHAIN_ROOT="${2-}"; shift 2 ;;
    --strict)     STRICT=1;            shift 1 ;;
    --no-strict)  STRICT=0;            shift 1 ;;
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
# Linker option dialect: "gnu" for the GNU ld / LLD flag spelling that
# gcc and ATfE share, "armlink" for Arm Compiler's.
link_style="gnu"
# Archive extractor, only needed by the armlink path.
ar_tool=""
# Libraries the archive legitimately depends on, appended AFTER the
# objects so the linker can resolve backwards into them.
post_link_libs=()

case "${TOOLCHAIN}" in
  gcc)
    compiler="arm-none-eabi-gcc"
    nm="arm-none-eabi-nm"
    size="arm-none-eabi-size"
    link_flags=(-nostartfiles --specs=nosys.specs)
    # GCC links libc but not libm. The archive genuinely calls floorf,
    # roundf and round (arm_resize_nearest_neighbor_s8/s16,
    # arm_quantize_f32_s8/s16), so a strict link needs libm on the line.
    # These are standard libm symbols every consumer already links --
    # unlike __ARM_undef, which nothing can supply.
    post_link_libs=(-lm)
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
    # armar cannot extract to a chosen directory; llvm-ar reads the same
    # ELF archive and is already installed alongside llvm-nm/llvm-size.
    ar_tool="$(command -v llvm-ar || true)"
    [[ -n "${ar_tool}" ]] || { echo "llvm-ar not found for armclang" >&2; exit 3; }
    arch_flags=(--target=arm-arm-none-eabi "${arch_flags[@]}")
    # No -nostdlib: the archive genuinely calls floorf/roundf/round, and
    # Arm Compiler's C library supplies them. Suppressing the library
    # scan would turn those into false unresolved-symbol failures, the
    # armlink equivalent of the -lm the gcc path appends.
    link_flags=(-nostartfiles)
    link_style="armlink"
    ;;
  *)
    echo "unsupported toolchain '${TOOLCHAIN}' (expect gcc|atfe|armclang)" >&2
    exit 2
    ;;
esac

[[ -x "${compiler}" || -n "$(command -v "${compiler}" 2>/dev/null || true)" ]] || { echo "compiler not found: ${compiler}" >&2; exit 3; }
[[ -n "${nm}" ]] || { echo "llvm-nm not found for ${TOOLCHAIN}" >&2; exit 3; }
[[ -n "${size}" ]] || { echo "llvm-size not found for ${TOOLCHAIN}" >&2; exit 3; }

if (( STRICT )); then
  echo ">>> strict smoke-linking ${TOOLCHAIN}/${TARGET_CPU} against $(basename "${LIBRARY}")"
else
  echo ">>> lenient smoke-linking ${TOOLCHAIN}/${TARGET_CPU} against $(basename "${LIBRARY}")"
fi

if [[ "${link_style}" == "gnu" ]]; then
  if (( STRICT )); then
    # Every object in the archive must resolve. --gc-sections is dropped
    # too: with it, kernels the smoke source does not call are discarded
    # before their undefined references are ever checked.
    resolve_flags=()
  else
    # shellcheck disable=SC2054  # commas are part of the -Wl, linker flags
    resolve_flags=(-Wl,--gc-sections -Wl,--unresolved-symbols=ignore-all)
  fi
  "${compiler}" \
    "${arch_flags[@]}" \
    "${link_flags[@]}" \
    "${resolve_flags[@]}" \
    -Wl,--entry=ns_cmsis_nn_smoke_refs \
    -Wl,--whole-archive "${LIBRARY}" -Wl,--no-whole-archive \
    -o "${elf}" \
    "${smoke_src}" \
    "${post_link_libs[@]+"${post_link_libs[@]}"}"
else
  # armlink dialect. Two deliberate differences from the GNU path:
  #
  #  1. There is no --whole-archive, so every member is extracted and
  #     named on the link line. Without this armlink would pull in only
  #     the members the smoke source references and an unresolved symbol
  #     in any other kernel would go unseen -- which is most of the
  #     archive, and exactly the gap #291 describes.
  #  2. --no_remove replaces "omit --gc-sections": armlink removes
  #     unused sections by default, which would discard those same
  #     members again after we went to the trouble of extracting them.
  #
  # armlink already errors on an unresolved reference (L6218E), so
  # strictness needs no third flag; --unresolved maps a dangling
  # reference onto a real symbol and is the lenient escape hatch.
  members_dir="${OUTDIR}/members_${TOOLCHAIN}_${TARGET_CPU}"
  rm -rf "${members_dir}"
  mkdir -p "${members_dir}"
  ( cd "${members_dir}" && "${ar_tool}" x "${LIBRARY}" )

  members=()
  while IFS= read -r m; do members+=("$m"); done \
    < <(find "${members_dir}" -name '*.o' | sort)

  listed="$("${ar_tool}" t "${LIBRARY}" | grep -c '\.o$' || true)"
  if (( ${#members[@]} == 0 )); then
    echo "no objects extracted from ${LIBRARY}" >&2
    exit 3
  fi
  if (( ${#members[@]} != listed )); then
    # Duplicate member basenames would silently overwrite on extract and
    # quietly shrink the link line, weakening the check without failing.
    echo "extracted ${#members[@]} objects but archive lists ${listed}" >&2
    exit 3
  fi
  echo ">>> linking ${#members[@]} extracted archive members"

  # shellcheck disable=SC2054  # commas are part of the -Wl, linker flags
  if (( STRICT )); then
    resolve_flags=(-Wl,--no_remove)
  else
    resolve_flags=(-Wl,--unresolved=ns_cmsis_nn_smoke_refs)
  fi
  "${compiler}" \
    "${arch_flags[@]}" \
    "${link_flags[@]}" \
    "${resolve_flags[@]}" \
    -Wl,--entry=ns_cmsis_nn_smoke_refs \
    -o "${elf}" \
    "${smoke_src}" \
    "${members[@]}" \
    "${post_link_libs[@]+"${post_link_libs[@]}"}"
fi
nm_input="${elf}"

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

size_out="$(${size} "${elf}")"
echo ">>> ${elf}"
echo "${size_out}"
echo ">>> smoke OK (${TOOLCHAIN}, ${TARGET_CPU})"

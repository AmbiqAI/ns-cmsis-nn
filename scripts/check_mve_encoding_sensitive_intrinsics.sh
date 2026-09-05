#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
#
# Keep the MVE intrinsics that old assemblers mis-encode out of the kernels.
# A sweep of the MVE instruction space over gas 2.39 to 2.45 found exactly two
# families whose encoding differs between releases: the Q-register form of
# VCVTB/VCVTT.F16<->F32, and the saturating narrowing shifts VQSHRN/VQSHRUN.
# Both assemble without a diagnostic and produce a valid but wrong instruction,
# so nothing downstream of the assembler can catch a new use.
# See AmbiqAI/ns-cmsis-nn#427 and AmbiqAI/ns-cmsis-nn#437.

set -euo pipefail

(($# > 0)) || exit 0

# awk reads a `name=value` operand as a variable assignment, so a relative
# path containing `=` would be skipped silently. `./` in front makes every
# relative operand a file; awk strips the one prefix back off when reporting.
operands=()
for path in "$@"; do
  case "${path}" in
    /*) operands+=("${path}") ;;
    *) operands+=("./${path}") ;;
  esac
done

hits=$(
  awk '
    BEGIN {
      # The wrappers that are allowed to name the conversions, removed from
      # the line before the raw form is looked for; their names contain it.
      wrapper = "arm_nn_vcvt[bt]q_(f16_f32|f32_f16)"
      cvt     = "(__arm_)?vcvt[bt]q(_[mx])?_(f16_f32|f32_f16)"
      shift   = "(__arm_)?vqshr(u)?n[bt]q[A-Za-z0-9_]*"
    }
    function report(line, pat, advice,   pos, start) {
      pos = 1
      while (match(substr(line, pos), pat)) {
        start = pos + RSTART - 1
        printf "%s:%d: %s: %s\n", name, FNR, substr(line, start, RLENGTH), advice
        pos = start + RLENGTH
      }
    }
    {
      name = FILENAME
      sub(/^\.\//, "", name)
      # The helper is where the raw conversions live; everything else routes
      # through it.
      if (name ~ /(^|\/)Include\/Internal\/arm_nn_vcvt_f16\.h$/) {
        next
      }
      line = $0
      gsub(wrapper, "", line)
      report(line, cvt, \
        "gas below binutils 2.43 mis-encodes this; call the arm_nn_ wrapper in " \
        "Include/Internal/arm_nn_vcvt_f16.h instead (AmbiqAI/ns-cmsis-nn#427)")
      report($0, shift, \
        "gas below binutils 2.43 assembles this as its rounding variant; no " \
        "wrapper exists yet, so it cannot be used here (AmbiqAI/ns-cmsis-nn#437)")
    }
  ' "${operands[@]}"
)

if [[ -n "${hits}" ]]; then
  printf '%s\n' "${hits}" >&2
  exit 1
fi

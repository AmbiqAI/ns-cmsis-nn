#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
#
# Fail on a direct call to the MVE half<->single conversion intrinsics, which
# the assembler in the older gated GCC releases mis-encodes; kernels reach them
# through the wrappers instead. See AmbiqAI/ns-cmsis-nn#427.

set -euo pipefail

(($# > 0)) || exit 0

# The wrappers end in the same characters as the intrinsics they replace, so an
# unanchored match would flag every use of the fix as the defect. Only a name
# with no identifier character in front of it is a finding.
pattern='(^|[^A-Za-z0-9_])vcvt[bt]q_f(16_f32|32_f16)($|[^A-Za-z0-9_])'

if hits="$(grep -nEH -- "${pattern}" "$@")"; then
  {
    printf '%s\n' "${hits}"
    cat <<'EOF'

Call the arm_nn_vcvtbq_f16_f32 / arm_nn_vcvttq_f16_f32 /
arm_nn_vcvtbq_f32_f16 / arm_nn_vcvttq_f32_f16 wrappers from
Include/Internal/arm_nn_vcvt_f16_fixup.h instead. The gas shipped with the
older gated Arm GNU releases encodes the Q-register form of these four
conversions wrongly, which builds and links clean and then faults or reads
the wrong registers at run time. See AmbiqAI/ns-cmsis-nn#427.
EOF
  } >&2
  exit 1
fi

/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

/* One function per shape of MML_SHAPES, each the inline body called with its literals, named for MML_LIST. Included
 * by mat_mult_nt_t_f32_literal_library.c and mat_mult_nt_t_f32_literal_c11.c, which the suite compiles under
 * different flag lists. */

#include "../../../../Include/Internal/arm_nn_mat_mult_nt_t_flt.h"
#include "mat_mult_nt_t_f32_literal.h"
#include <float.h>

#define MML_MIN_ARG(x) (x)
#define MML_MAX_ARG(x) (x)
#define MML_MIN_FULL(x) (-FLT_MAX)
#define MML_MAX_FULL(x) (FLT_MAX)
#define MML_MIN_RELU(x) (0.0f)
#define MML_MAX_RELU(x) (FLT_MAX)

#define MML_PASTE(a, b) a##b
#define MML_EXPAND(a, b) MML_PASTE(a, b)
#define MML_DEFINE_IN(list, l, r, k, o, clamp)                                                                         \
    arm_cmsis_nn_status MML_NAME(list, l, r, k, o, clamp)(const float32_t *lhs,                                        \
                                                          const float32_t *rhs,                                        \
                                                          const float32_t *bias,                                       \
                                                          float32_t *dst,                                              \
                                                          float32_t act_min,                                           \
                                                          float32_t act_max)                                           \
    {                                                                                                                  \
        (void)act_min;                                                                                                 \
        (void)act_max;                                                                                                 \
        return arm_nn_mat_mult_nt_t_f32_inline(                                                                        \
            lhs, rhs, bias, dst, l, r, k, o, MML_MIN_##clamp(act_min), MML_MAX_##clamp(act_max));                      \
    }
#define MML_DEFINE_LIST(list, l, r, k, o, clamp) MML_DEFINE_IN(list, l, r, k, o, clamp)
#define MML_DEFINE(l, r, k, o, clamp) MML_DEFINE_LIST(MML_LIST, l, r, k, o, clamp)

#if ARM_NN_ENABLE_F32
MML_SHAPES(MML_DEFINE)
#endif

#if defined(__GNUC__) && !defined(__clang__) && defined(__STRICT_ANSI__) && !defined(__FAST_MATH__) &&                 \
    (!defined(ARM_MATH_MVEF) || defined(ARM_MATH_AUTOVECTORIZE))
const int MML_EXPAND(MML_EXPAND(mml_, MML_LIST), _exact) = 1;
#else
const int MML_EXPAND(MML_EXPAND(mml_, MML_LIST), _exact) = 0;
#endif

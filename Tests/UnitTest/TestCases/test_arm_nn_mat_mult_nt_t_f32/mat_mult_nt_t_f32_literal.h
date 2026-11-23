/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */
#pragma once

/* The shapes the literal-extent cases compile arm_nn_mat_mult_nt_t_f32_inline at, as an emitted operator does: lhs
 * rows, rhs rows, rhs columns and the output row stride as literals, and the clamp bounds either passed at run time
 * (ARG) or literal (FULL: -FLT_MAX to FLT_MAX; RELU: 0 to FLT_MAX). The first rows are the extents of the farm probes'
 * float FULLY_CONNECTED operators; the rest cover each leg's blocks and remainders. mat_mult_nt_t_f32_literal_body.h
 * defines one function per shape for each flag list the suite compiles it under (MML_LIST). */

#include "arm_nnsupportfunctions.h"

#define MML_SHAPES(X)                                                                                                  \
    X(1, 4, 9, 4, FULL)        /* affine */                                                                            \
    X(1, 32, 16, 32, RELU)     /* mlp, first layer */                                                                  \
    X(1, 16, 32, 16, RELU)     /* mlp, second layer */                                                                 \
    X(1, 3, 16, 3, FULL)       /* mlp, output layer */                                                                 \
    X(1, 6, 16, 6, FULL)       /* vote */                                                                              \
    X(1, 1, 16, 1, FULL)       /* forest_r */                                                                          \
    X(1, 3, 3, 3, FULL)        /* svc, second product */                                                               \
    X(1, 3, 738, 3, FULL)      /* svc, first product */                                                                \
    X(1, 2048, 16, 2048, FULL) /* knn */                                                                               \
    X(1, 4, 9, 4, ARG)                                                                                                 \
    X(1, 1, 1, 1, ARG)                                                                                                 \
    X(1, 1, 5, 1, ARG)                                                                                                 \
    X(1, 3, 4, 3, ARG)                                                                                                 \
    X(1, 4, 1, 4, ARG)                                                                                                 \
    X(1, 5, 9, 5, ARG)                                                                                                 \
    X(1, 7, 17, 9, ARG)                                                                                                \
    X(1, 8, 4, 8, ARG)                                                                                                 \
    X(1, 9, 30, 9, ARG)                                                                                                \
    X(1, 12, 5, 12, ARG)                                                                                               \
    X(1, 13, 33, 13, ARG)                                                                                              \
    X(1, 17, 16, 17, ARG)                                                                                              \
    X(1, 17, 64, 20, ARG)                                                                                              \
    X(1, 32, 16, 32, ARG)                                                                                              \
    X(1, 16, 32, 16, ARG)                                                                                              \
    X(1, 6, 16, 6, ARG)                                                                                                \
    X(2, 5, 9, 5, ARG)                                                                                                 \
    X(2, 8, 31, 8, ARG)                                                                                                \
    X(3, 11, 7, 14, ARG)                                                                                               \
    X(3, 4, 64, 4, ARG)

#define MML_NAME(list, l, r, k, o, clamp) mml_##list##_##l##_##r##_##k##_##o##_##clamp

typedef arm_cmsis_nn_status (*mml_fn)(const float32_t *lhs,
                                      const float32_t *rhs,
                                      const float32_t *bias,
                                      float32_t *dst,
                                      float32_t act_min,
                                      float32_t act_max);

#define MML_DECLARE(l, r, k, o, clamp)                                                                                 \
    arm_cmsis_nn_status MML_NAME(library, l, r, k, o, clamp)(const float32_t *lhs,                                     \
                                                             const float32_t *rhs,                                     \
                                                             const float32_t *bias,                                    \
                                                             float32_t *dst,                                           \
                                                             float32_t act_min,                                        \
                                                             float32_t act_max);                                       \
    arm_cmsis_nn_status MML_NAME(c11, l, r, k, o, clamp)(const float32_t *lhs,                                         \
                                                         const float32_t *rhs,                                         \
                                                         const float32_t *bias,                                        \
                                                         float32_t *dst,                                               \
                                                         float32_t act_min,                                            \
                                                         float32_t act_max);
MML_SHAPES(MML_DECLARE)
#undef MML_DECLARE

/* 1 where the TU compiled under that list holds the scalar leg under GCC in an ISO C mode without fast-math, where
 * neither the body nor the reference contracts or reorders a sum, so every output equals the reference's bits. */
extern const int mml_library_exact;
extern const int mml_c11_exact;

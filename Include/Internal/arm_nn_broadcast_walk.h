/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_nn_broadcast_walk.h
 * Description:  NHWC broadcast validation and traversal shared by the
 *               elementwise minimum/maximum, comparison, add, sub, mul,
 *               squared-difference and PReLU kernels.
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#ifndef ARM_NN_BROADCAST_WALK_H
#define ARM_NN_BROADCAST_WALK_H

#include "arm_nn_types.h"

#include <stdint.h>

/**
 * @brief Check that one NHWC dimension of two operands broadcasts to the output dimension.
 *
 * TensorFlow Lite broadcast rules: both inputs must be at least 1 (an empty tensor is rejected rather
 * than treated as a no-op), they must be equal or one of them must be 1, and the output dimension
 * must be the larger of the two.
 */
static inline int32_t arm_nn_broadcast_dim_valid(const int32_t dim_1, const int32_t dim_2, const int32_t dim_out)
{
    if (dim_1 < 1 || dim_2 < 1)
    {
        return 0;
    }
    if (dim_1 != dim_2 && dim_1 != 1 && dim_2 != 1)
    {
        return 0;
    }
    return dim_out == ((dim_1 > dim_2) ? dim_1 : dim_2);
}

/**
 * @brief Check that two NHWC operands broadcast to the given output shape.
 *
 * Every kernel that uses ARM_NN_BROADCAST_WALK_NHWC must reject arguments that fail this check,
 * since the walk indexes each input by its own dims and writes the output by the output dims.
 */
static inline int32_t
arm_nn_broadcast_dims_valid(const cmsis_nn_dims *dims_1, const cmsis_nn_dims *dims_2, const cmsis_nn_dims *dims_out)
{
    return arm_nn_broadcast_dim_valid(dims_1->n, dims_2->n, dims_out->n) &&
        arm_nn_broadcast_dim_valid(dims_1->h, dims_2->h, dims_out->h) &&
        arm_nn_broadcast_dim_valid(dims_1->w, dims_2->w, dims_out->w) &&
        arm_nn_broadcast_dim_valid(dims_1->c, dims_2->c, dims_out->c);
}

/**
 * @brief Walk an NHWC broadcast of two operands, calling a contiguous kernel on each run.
 *
 * Each input is indexed by its own dims: a dimension of 1 has stride 0 and is broadcast, any
 * other dimension equals the output dimension and strides normally. The longest contiguous
 * run whose shapes agree is handed to the caller's kernels, so the common cases (identical
 * shapes, a single scalar, per-batch, per-row, per-channel) each cost one call per run.
 *
 * @param IN_TYPE   element type of the inputs
 * @param OUT_TYPE  element type of the output
 * @param in_1      const IN_TYPE *  first input
 * @param dims_1    const cmsis_nn_dims *  dims of in_1
 * @param in_2      const IN_TYPE *  second input
 * @param dims_2    const cmsis_nn_dims *  dims of in_2
 * @param out       OUT_TYPE *  output, sized by dims_out
 * @param dims_out  const cmsis_nn_dims *  broadcast output dims (see arm_nn_broadcast_dims_valid)
 * @param FULL      FULL(const IN_TYPE *a, const IN_TYPE *b, OUT_TYPE *o, int32_t n)
 *                  elementwise kernel over n elements of a and b
 * @param SCALAR_1  SCALAR_1(const IN_TYPE *scalar, const IN_TYPE *vec, OUT_TYPE *o, int32_t n)
 *                  kernel where *scalar is one element of in_1 broadcast against n elements of in_2
 * @param SCALAR_2  SCALAR_2(const IN_TYPE *scalar, const IN_TYPE *vec, OUT_TYPE *o, int32_t n)
 *                  kernel where *scalar is one element of in_2 broadcast against n elements of in_1;
 *                  note the operands arrive in reversed order, so an asymmetric kernel must swap them
 *
 * Preconditions: arm_nn_broadcast_dims_valid(dims_1, dims_2, dims_out) is non-zero.
 */
#define ARM_NN_BROADCAST_WALK_NHWC(                                                                                    \
    IN_TYPE, OUT_TYPE, in_1, dims_1, in_2, dims_2, out, dims_out, FULL, SCALAR_1, SCALAR_2)                            \
    do                                                                                                                 \
    {                                                                                                                  \
        const int32_t bw_c_1 = (dims_1)->c;                                                                            \
        const int32_t bw_c_2 = (dims_2)->c;                                                                            \
        const int32_t bw_row_1 = (dims_1)->w * bw_c_1;                                                                 \
        const int32_t bw_row_2 = (dims_2)->w * bw_c_2;                                                                 \
        const int32_t bw_plane_1 = (dims_1)->h * bw_row_1;                                                             \
        const int32_t bw_plane_2 = (dims_2)->h * bw_row_2;                                                             \
        const int32_t bw_total_1 = (dims_1)->n * bw_plane_1;                                                           \
        const int32_t bw_total_2 = (dims_2)->n * bw_plane_2;                                                           \
        OUT_TYPE *bw_out = (out);                                                                                      \
                                                                                                                       \
        if ((dims_1)->n == (dims_2)->n && (dims_1)->h == (dims_2)->h && (dims_1)->w == (dims_2)->w &&                  \
            bw_c_1 == bw_c_2)                                                                                          \
        {                                                                                                              \
            FULL((in_1), (in_2), bw_out, bw_total_1);                                                                  \
            break;                                                                                                     \
        }                                                                                                              \
        if (bw_total_1 == 1)                                                                                           \
        {                                                                                                              \
            SCALAR_1((in_1), (in_2), bw_out, bw_total_2);                                                              \
            break;                                                                                                     \
        }                                                                                                              \
        if (bw_total_2 == 1)                                                                                           \
        {                                                                                                              \
            SCALAR_2((in_2), (in_1), bw_out, bw_total_1);                                                              \
            break;                                                                                                     \
        }                                                                                                              \
                                                                                                                       \
        const int32_t bw_n_stride_1 = ((dims_1)->n == 1) ? 0 : bw_plane_1;                                             \
        const int32_t bw_n_stride_2 = ((dims_2)->n == 1) ? 0 : bw_plane_2;                                             \
        const int32_t bw_h_stride_1 = ((dims_1)->h == 1) ? 0 : bw_row_1;                                               \
        const int32_t bw_h_stride_2 = ((dims_2)->h == 1) ? 0 : bw_row_2;                                               \
        const int32_t bw_w_stride_1 = ((dims_1)->w == 1) ? 0 : bw_c_1;                                                 \
        const int32_t bw_w_stride_2 = ((dims_2)->w == 1) ? 0 : bw_c_2;                                                 \
        const int32_t bw_plane_same =                                                                                  \
            ((dims_1)->h == (dims_2)->h) && ((dims_1)->w == (dims_2)->w) && (bw_c_1 == bw_c_2);                        \
        const int32_t bw_row_same = ((dims_1)->w == (dims_2)->w) && (bw_c_1 == bw_c_2);                                \
                                                                                                                       \
        for (int32_t bw_n = 0; bw_n < (dims_out)->n; bw_n++)                                                           \
        {                                                                                                              \
            const IN_TYPE *bw_p_1 = (in_1) + bw_n * bw_n_stride_1;                                                     \
            const IN_TYPE *bw_p_2 = (in_2) + bw_n * bw_n_stride_2;                                                     \
                                                                                                                       \
            if (bw_plane_same)                                                                                         \
            {                                                                                                          \
                FULL(bw_p_1, bw_p_2, bw_out, bw_plane_1);                                                              \
                bw_out += bw_plane_1;                                                                                  \
                continue;                                                                                              \
            }                                                                                                          \
            for (int32_t bw_h = 0; bw_h < (dims_out)->h; bw_h++)                                                       \
            {                                                                                                          \
                const IN_TYPE *bw_r_1 = bw_p_1 + bw_h * bw_h_stride_1;                                                 \
                const IN_TYPE *bw_r_2 = bw_p_2 + bw_h * bw_h_stride_2;                                                 \
                                                                                                                       \
                if (bw_row_same)                                                                                       \
                {                                                                                                      \
                    FULL(bw_r_1, bw_r_2, bw_out, bw_row_1);                                                            \
                    bw_out += bw_row_1;                                                                                \
                }                                                                                                      \
                else if (bw_row_1 == 1)                                                                                \
                {                                                                                                      \
                    SCALAR_1(bw_r_1, bw_r_2, bw_out, bw_row_2);                                                        \
                    bw_out += bw_row_2;                                                                                \
                }                                                                                                      \
                else if (bw_row_2 == 1)                                                                                \
                {                                                                                                      \
                    SCALAR_2(bw_r_2, bw_r_1, bw_out, bw_row_1);                                                        \
                    bw_out += bw_row_1;                                                                                \
                }                                                                                                      \
                else                                                                                                   \
                {                                                                                                      \
                    for (int32_t bw_w = 0; bw_w < (dims_out)->w; bw_w++)                                               \
                    {                                                                                                  \
                        const IN_TYPE *bw_q_1 = bw_r_1 + bw_w * bw_w_stride_1;                                         \
                        const IN_TYPE *bw_q_2 = bw_r_2 + bw_w * bw_w_stride_2;                                         \
                                                                                                                       \
                        if (bw_c_1 == bw_c_2)                                                                          \
                        {                                                                                              \
                            FULL(bw_q_1, bw_q_2, bw_out, bw_c_1);                                                      \
                            bw_out += bw_c_1;                                                                          \
                        }                                                                                              \
                        else if (bw_c_1 == 1)                                                                          \
                        {                                                                                              \
                            SCALAR_1(bw_q_1, bw_q_2, bw_out, bw_c_2);                                                  \
                            bw_out += bw_c_2;                                                                          \
                        }                                                                                              \
                        else                                                                                           \
                        {                                                                                              \
                            SCALAR_2(bw_q_2, bw_q_1, bw_out, bw_c_1);                                                  \
                            bw_out += bw_c_1;                                                                          \
                        }                                                                                              \
                    }                                                                                                  \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
    } while (0)

#endif /* ARM_NN_BROADCAST_WALK_H */

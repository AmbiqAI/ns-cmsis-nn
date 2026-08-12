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
 * Title:        arm_strided_slice_common.h
 * Description:  Shared strided-slice template for CMSIS-NN
 *
 * $Date:        12 August 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#ifndef ARM_STRIDED_SLICE_COMMON_H
#define ARM_STRIDED_SLICE_COMMON_H

#include "arm_nn_types.h"

/*
 * StridedSlice is pure data movement, so every dtype variant shares one
 * element-indexed implementation; only the scalar type and the copy helper
 * differ. MEMCPY_FUNC must accept an element count; arm_memcpy_s8's
 * byte-count contract coincides with this for 1-byte elements.
 */
#define ARM_STRIDED_SLICE_DEFINE(FUNC_NAME, SCALAR_T, MEMCPY_FUNC)                                                     \
    /* Inline helper: choose < or > based on stride sign */                                                            \
    static inline bool FUNC_NAME##_stride_continue(int32_t idx, int32_t stop, int32_t stride)                          \
    {                                                                                                                  \
        return (stride > 0) ? (idx < stop) : (idx > stop);                                                             \
    }                                                                                                                  \
                                                                                                                       \
    arm_cmsis_nn_status FUNC_NAME(const SCALAR_T *input_data,                                                          \
                                  SCALAR_T *output_data,                                                               \
                                  const cmsis_nn_dims *const input_dims,                                               \
                                  const cmsis_nn_dims *const begin_dims,                                               \
                                  const cmsis_nn_dims *const stride_dims,                                              \
                                  const cmsis_nn_dims *const output_dims)                                              \
    {                                                                                                                  \
        const int32_t in_h = input_dims->h, in_w = input_dims->w, in_c = input_dims->c;                                \
        const int32_t b_n = begin_dims->n, b_h = begin_dims->h, b_w = begin_dims->w, b_c = begin_dims->c;              \
        const int32_t s_n = stride_dims->n, s_h = stride_dims->h, s_w = stride_dims->w, s_c = stride_dims->c;          \
        const int32_t o_n = output_dims->n, o_h = output_dims->h, o_w = output_dims->w, o_c = output_dims->c;          \
                                                                                                                       \
        const int32_t plane_elems = in_h * in_w * in_c; /* H x W x C */                                                \
        const int32_t slice_elems = in_w * in_c;        /* W x C */                                                    \
        const int32_t row_elems = in_c;                 /* C */                                                        \
                                                                                                                       \
        /* Case 1: whole-slab copy: only slice N, full H/W/C */                                                        \
        if (s_n == 1 && s_h == 1 && s_w == 1 && s_c == 1 && b_h == 0 && o_h == in_h && b_w == 0 && o_w == in_w &&      \
            b_c == 0 && o_c == in_c)                                                                                   \
        {                                                                                                              \
            int32_t offset = b_n * plane_elems;                                                                        \
            int32_t total = o_n * plane_elems;                                                                         \
            MEMCPY_FUNC(output_data, input_data + offset, total);                                                      \
            return ARM_CMSIS_NN_SUCCESS;                                                                               \
        }                                                                                                              \
                                                                                                                       \
        /* Case 2: per-batch slice: slice N & H, full W/C */                                                           \
        if (s_h == 1 && s_w == 1 && s_c == 1 && b_w == 0 && o_w == in_w && b_c == 0 && o_c == in_c)                    \
        {                                                                                                              \
            int32_t batch_block = o_h * slice_elems;                                                                   \
            for (int32_t n = 0; n < o_n; n++)                                                                          \
            {                                                                                                          \
                int32_t start = (b_n + n * s_n) * plane_elems + b_h * slice_elems;                                     \
                MEMCPY_FUNC(output_data + n * batch_block, input_data + start, batch_block);                           \
            }                                                                                                          \
            return ARM_CMSIS_NN_SUCCESS;                                                                               \
        }                                                                                                              \
                                                                                                                       \
        /* Case 3: per-row slice: stride in N/H, full channels, partial width */                                       \
        if (s_w == 1 && s_c == 1 && b_c == 0 && o_c == in_c)                                                           \
        {                                                                                                              \
            int32_t out_idx = 0;                                                                                       \
            int32_t row_block = o_w * in_c;                                                                            \
            for (int32_t n = 0; n < o_n; n++)                                                                          \
            {                                                                                                          \
                int32_t base_n = (b_n + n * s_n) * plane_elems;                                                        \
                for (int32_t h = 0; h < o_h; h++)                                                                      \
                {                                                                                                      \
                    int32_t start = base_n + (b_h + h * s_h) * slice_elems + b_w * in_c;                               \
                    MEMCPY_FUNC(output_data + out_idx, input_data + start, row_block);                                 \
                    out_idx += row_block;                                                                              \
                }                                                                                                      \
            }                                                                                                          \
            return ARM_CMSIS_NN_SUCCESS;                                                                               \
        }                                                                                                              \
                                                                                                                       \
        /* Case 4: per-pixel channel slice */                                                                          \
        if (s_c == 1)                                                                                                  \
        {                                                                                                              \
            int32_t out_idx = 0;                                                                                       \
            for (int32_t n = 0; n < o_n; n++)                                                                          \
            {                                                                                                          \
                int32_t base_n = (b_n + n * s_n) * plane_elems;                                                        \
                for (int32_t h = 0; h < o_h; h++)                                                                      \
                {                                                                                                      \
                    int32_t base_h = base_n + (b_h + h * s_h) * slice_elems;                                           \
                    for (int32_t w = 0; w < o_w; w++)                                                                  \
                    {                                                                                                  \
                        int32_t start = base_h + (b_w + w * s_w) * row_elems + b_c;                                    \
                        MEMCPY_FUNC(output_data + out_idx, input_data + start, o_c);                                   \
                        out_idx += o_c;                                                                                \
                    }                                                                                                  \
                }                                                                                                      \
            }                                                                                                          \
            return ARM_CMSIS_NN_SUCCESS;                                                                               \
        }                                                                                                              \
                                                                                                                       \
        /* General case: truly arbitrary strides (including negative) */                                               \
        {                                                                                                              \
            const int32_t stop0 = b_n + s_n * o_n;                                                                     \
            const int32_t stop1 = b_h + s_h * o_h;                                                                     \
            const int32_t stop2 = b_w + s_w * o_w;                                                                     \
            const int32_t stop3 = b_c + s_c * o_c;                                                                     \
                                                                                                                       \
            int32_t out_idx = 0;                                                                                       \
            for (int32_t off0 = b_n; FUNC_NAME##_stride_continue(off0, stop0, s_n); off0 += s_n)                       \
            {                                                                                                          \
                for (int32_t off1 = b_h; FUNC_NAME##_stride_continue(off1, stop1, s_h); off1 += s_h)                   \
                {                                                                                                      \
                    for (int32_t off2 = b_w; FUNC_NAME##_stride_continue(off2, stop2, s_w); off2 += s_w)               \
                    {                                                                                                  \
                        for (int32_t off3 = b_c; FUNC_NAME##_stride_continue(off3, stop3, s_c); off3 += s_c)           \
                        {                                                                                              \
                            const int32_t idx = off0 * plane_elems + off1 * slice_elems + off2 * row_elems + off3;     \
                            output_data[out_idx++] = input_data[idx];                                                  \
                        }                                                                                              \
                    }                                                                                                  \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
                                                                                                                       \
        return ARM_CMSIS_NN_SUCCESS;                                                                                   \
    }

#endif /* ARM_STRIDED_SLICE_COMMON_H */

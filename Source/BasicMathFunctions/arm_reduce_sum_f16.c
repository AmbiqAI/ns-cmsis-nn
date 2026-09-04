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
 * Title:        arm_reduce_sum_f16
 * Description:  Sum reduction operator for float16 tensors
 *
 * $Date:        12 August 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

#if ARM_NN_ENABLE_F16

/**
 *  @ingroup Public
 */

/**
 * @addtogroup Reduction
 * @{
 */

// Generic fallback: reduce any axis-mask combination.
// Accumulation is float32: a float16 accumulator would lose precision
// linearly with the reduction count and can overflow (max 65504).
static arm_cmsis_nn_status arm_reduce_sum_generic_f16(const float16_t *input_data,
                                                      const cmsis_nn_dims *input_dims,
                                                      const cmsis_nn_dims *axis_dims,
                                                      float16_t *output_data,
                                                      const cmsis_nn_dims *output_dims)
{
    const int32_t H = input_dims->h;
    const int32_t W = input_dims->w;
    const int32_t C = input_dims->c;

    const int32_t out_N = output_dims->n;
    const int32_t out_H = output_dims->h;
    const int32_t out_W = output_dims->w;
    const int32_t out_C = output_dims->c;

    const int32_t N_limit = axis_dims->n ? input_dims->n : 1;
    const int32_t H_limit = axis_dims->h ? H : 1;
    const int32_t W_limit = axis_dims->w ? W : 1;
    const int32_t C_limit = axis_dims->c ? C : 1;

    for (int32_t n = 0; n < out_N; ++n)
        for (int32_t h = 0; h < out_H; ++h)
            for (int32_t w = 0; w < out_W; ++w)
                for (int32_t c = 0; c < out_C; ++c)
                {
                    float32_t acc = 0.0f;

                    for (int32_t ni = 0; ni < N_limit; ++ni)
                        for (int32_t hi = 0; hi < H_limit; ++hi)
                            for (int32_t wi = 0; wi < W_limit; ++wi)
                                for (int32_t ci = 0; ci < C_limit; ++ci)
                                {
                                    const int32_t idx_n = axis_dims->n ? ni : n;
                                    const int32_t idx_h = axis_dims->h ? hi : h;
                                    const int32_t idx_w = axis_dims->w ? wi : w;
                                    const int32_t idx_c = axis_dims->c ? ci : c;

                                    const int32_t flat_index = ((idx_n * H + idx_h) * W + idx_w) * C + idx_c;
                                    acc += (float32_t)input_data[flat_index];
                                }

                    const int32_t out_index = ((n * out_H + h) * out_W + w) * out_C + c;
                    output_data[out_index] = (float16_t)acc;
                }

    return ARM_CMSIS_NN_SUCCESS;
}

// Fast path: reduced axes form a contiguous suffix -> row sums
static arm_cmsis_nn_status arm_reduce_sum_flatten_last_dims_f16(const float16_t *input_data,
                                                                float16_t *output_data,
                                                                int32_t outer_size,
                                                                int32_t inner_size)
{
    for (int32_t i = 0; i < outer_size; ++i)
    {
        const float16_t *row = &input_data[i * inner_size];

    #if defined(ARM_MATH_MVE_FLOAT16) && defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
        // Widen each half vector to float32 before accumulating; predicated
        // loads zero inactive lanes, the identity for summation.
        float32x4_t vacc_lo = vdupq_n_f32(0.0f);
        float32x4_t vacc_hi = vdupq_n_f32(0.0f);
        for (int32_t j = 0; j < inner_size; j += 8)
        {
            const mve_pred16_t p = vctp16q((uint32_t)(inner_size - j));
            const float16x8_t v = vld1q_z(&row[j], p);
            vacc_lo = vaddq(vacc_lo, vcvtbq_f32_f16(v));
            vacc_hi = vaddq(vacc_hi, vcvttq_f32_f16(v));
        }
        const float32_t acc = arm_nn_vec_reduce_add_f32(vaddq(vacc_lo, vacc_hi));
        output_data[i] = (float16_t)acc;
    #else
        float32_t acc = 0.0f;
        for (int32_t j = 0; j < inner_size; ++j)
        {
            acc += (float32_t)row[j];
        }
        output_data[i] = (float16_t)acc;
    #endif
    }

    return ARM_CMSIS_NN_SUCCESS;
}

/*
 * float16 sum over the specified axes. Accumulates in float32 and rounds
 * once to float16 (torch CPU semantics), avoiding float16 overflow and
 * linear precision loss. No requantization or clamp; NaN and Inf
 * propagate. Vector and scalar builds may differ in final ulps because
 * float accumulation order differs.
 *
 * Refer header file for details.
 *
 */
arm_cmsis_nn_status arm_reduce_sum_f16(const float16_t *input_data,
                                       const cmsis_nn_dims *input_dims,
                                       const cmsis_nn_dims *axis_dims,
                                       float16_t *output_data,
                                       const cmsis_nn_dims *output_dims)
{
    if (!input_data || !input_dims || !axis_dims || !output_data || !output_dims)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    int32_t in_dims[4] = {input_dims->n, input_dims->h, input_dims->w, input_dims->c};
    int32_t axis_arr[4] = {axis_dims->n ? 1 : 0, axis_dims->h ? 1 : 0, axis_dims->w ? 1 : 0, axis_dims->c ? 1 : 0};

    const int32_t suffix_start = arm_reduce_get_flatten_suffix_start_from_arrays(in_dims, axis_arr);

    if (suffix_start >= 0)
    {
        int32_t outer_size = 1, inner_size = 1;

        for (int32_t d = 0; d < suffix_start; ++d)
            outer_size *= in_dims[d];
        for (int32_t d = suffix_start; d < 4; ++d)
            inner_size *= in_dims[d];

        return arm_reduce_sum_flatten_last_dims_f16(input_data, output_data, outer_size, inner_size);
    }

    return arm_reduce_sum_generic_f16(input_data, input_dims, axis_dims, output_data, output_dims);
}

/**
 * @} end of Reduction group
 */

#endif /* ARM_NN_ENABLE_F16 */

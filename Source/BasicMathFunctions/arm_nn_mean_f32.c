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
 * Title:        arm_nn_mean_f32
 * Description:  Mean reduction operator for float32 tensors
 *
 * $Date:        2 September 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

#if ARM_NN_ENABLE_F32

/**
 *  @ingroup Public
 */

/**
 * @addtogroup Reduction
 * @{
 */

static arm_cmsis_nn_status arm_mean_generic_f32(const float32_t *input_data,
                                                const cmsis_nn_dims *input_dims,
                                                const cmsis_nn_dims *axis_dims,
                                                float32_t *output_data,
                                                const cmsis_nn_dims *output_dims,
                                                int32_t reduction_count)
{
    const int32_t input_h = input_dims->h;
    const int32_t input_w = input_dims->w;
    const int32_t input_c = input_dims->c;

    const int32_t n_limit = axis_dims->n ? input_dims->n : 1;
    const int32_t h_limit = axis_dims->h ? input_h : 1;
    const int32_t w_limit = axis_dims->w ? input_w : 1;
    const int32_t c_limit = axis_dims->c ? input_c : 1;

    for (int32_t n = 0; n < output_dims->n; ++n)
        for (int32_t h = 0; h < output_dims->h; ++h)
            for (int32_t w = 0; w < output_dims->w; ++w)
                for (int32_t c = 0; c < output_dims->c; ++c)
                {
                    float32_t sum = 0.0f;

                    for (int32_t ni = 0; ni < n_limit; ++ni)
                        for (int32_t hi = 0; hi < h_limit; ++hi)
                            for (int32_t wi = 0; wi < w_limit; ++wi)
                                for (int32_t ci = 0; ci < c_limit; ++ci)
                                {
                                    const int32_t input_n = axis_dims->n ? ni : n;
                                    const int32_t input_h_index = axis_dims->h ? hi : h;
                                    const int32_t input_w_index = axis_dims->w ? wi : w;
                                    const int32_t input_c_index = axis_dims->c ? ci : c;
                                    const int32_t input_index =
                                        ((input_n * input_h + input_h_index) * input_w + input_w_index) * input_c +
                                        input_c_index;
                                    sum += input_data[input_index];
                                }

                    const int32_t output_index = ((n * output_dims->h + h) * output_dims->w + w) * output_dims->c + c;
                    output_data[output_index] = sum / (float32_t)reduction_count;
                }

    return ARM_CMSIS_NN_SUCCESS;
}

// The divisor is the caller-computed reduction count rather than inner_size:
// the two are equal only through arm_reduce_get_flatten_suffix_start_from_arrays's
// current guarantee that every suffix dim is reduced or size 1, and mean kernels
// are the only users of that helper with a divisor to get wrong (PR #294 review,
// item 3 -- inherited from arm_nn_mean_f16).
static arm_cmsis_nn_status arm_mean_flatten_last_dims_f32(const float32_t *input_data,
                                                          float32_t *output_data,
                                                          int32_t outer_size,
                                                          int32_t inner_size,
                                                          int32_t reduction_count)
{
    for (int32_t i = 0; i < outer_size; ++i)
    {
        const float32_t *row = &input_data[i * inner_size];

    #if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
        // Predicated loads zero inactive lanes, which is the identity for
        // summation, so the tail folds into the vector loop.
        float32x4_t vacc = vdupq_n_f32(0.0f);
        for (int32_t j = 0; j < inner_size; j += 4)
        {
            const mve_pred16_t p = vctp32q((uint32_t)(inner_size - j));
            vacc = vaddq(vacc, vld1q_z(&row[j], p));
        }
        const float32_t sum = arm_nn_vec_reduce_add_f32(vacc);
    #else
        float32_t sum = 0.0f;
        for (int32_t j = 0; j < inner_size; ++j)
        {
            sum += row[j];
        }
    #endif

        output_data[i] = sum / (float32_t)reduction_count;
    }

    return ARM_CMSIS_NN_SUCCESS;
}

/*
 * float32 mean over the specified axes. Accumulation and the single divide
 * are in float32: unlike arm_nn_mean_f16 there is no cheap wider
 * accumulator on M-profile, so rounding error grows with the reduction
 * length (matching arm_reduce_sum_f32). NaN and Inf propagate. Vector and
 * scalar builds may differ in final ulps because float accumulation order
 * differs.
 *
 * Refer header file for details.
 *
 */
arm_cmsis_nn_status arm_nn_mean_f32(const float32_t *input_data,
                                    const cmsis_nn_dims *input_dims,
                                    const cmsis_nn_dims *axis_dims,
                                    float32_t *output_data,
                                    const cmsis_nn_dims *output_dims)
{
    if (!input_data || !input_dims || !axis_dims || !output_data || !output_dims)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    const int32_t input_shape[4] = {input_dims->n, input_dims->h, input_dims->w, input_dims->c};
    const int32_t output_shape[4] = {output_dims->n, output_dims->h, output_dims->w, output_dims->c};
    const int32_t axis_mask[4] = {
        axis_dims->n ? 1 : 0, axis_dims->h ? 1 : 0, axis_dims->w ? 1 : 0, axis_dims->c ? 1 : 0};
    int64_t input_size = 1;
    int64_t reduction_count = 1;

    for (int32_t dimension = 0; dimension < 4; ++dimension)
    {
        if (input_shape[dimension] < 1)
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }

        const int32_t expected_output_dimension = axis_mask[dimension] ? 1 : input_shape[dimension];
        if (output_shape[dimension] != expected_output_dimension || input_size > INT32_MAX / input_shape[dimension])
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
        input_size *= input_shape[dimension];

        if (axis_mask[dimension])
        {
            if (reduction_count > INT32_MAX / input_shape[dimension])
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            reduction_count *= input_shape[dimension];
        }
    }

    const int32_t suffix_start = arm_reduce_get_flatten_suffix_start_from_arrays(input_shape, axis_mask);
    if (suffix_start >= 0)
    {
        int32_t outer_size = 1;
        int32_t inner_size = 1;

        for (int32_t dimension = 0; dimension < suffix_start; ++dimension)
        {
            outer_size *= input_shape[dimension];
        }
        for (int32_t dimension = suffix_start; dimension < 4; ++dimension)
        {
            inner_size *= input_shape[dimension];
        }

        return arm_mean_flatten_last_dims_f32(
            input_data, output_data, outer_size, inner_size, (int32_t)reduction_count);
    }

    return arm_mean_generic_f32(input_data, input_dims, axis_dims, output_data, output_dims, (int32_t)reduction_count);
}

/**
 * @} end of Reduction group
 */

#endif /* ARM_NN_ENABLE_F32 */

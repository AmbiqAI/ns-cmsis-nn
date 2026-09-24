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
 * Title:        arm_reduce_sum_f32
 * Description:  Sum reduction operator for float32 tensors
 *
 * $Date:        12 August 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
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

// Generic fallback: reduce any axis-mask combination
static arm_cmsis_nn_status arm_reduce_sum_generic_f32_legacy(const float32_t *input_data,
                                                             const cmsis_nn_dims *input_dims,
                                                             const cmsis_nn_dims *axis_dims,
                                                             float32_t *output_data,
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
                                    acc += input_data[flat_index];
                                }

                    const int32_t out_index = ((n * out_H + h) * out_W + w) * out_C + c;
                    output_data[out_index] = acc;
                }

    return ARM_CMSIS_NN_SUCCESS;
}

// Refs #484.
static arm_cmsis_nn_status arm_reduce_sum_generic_f32(const float32_t *input_data,
                                                      const cmsis_nn_dims *input_dims,
                                                      const cmsis_nn_dims *axis_dims,
                                                      float32_t *output_data,
                                                      const cmsis_nn_dims *output_dims)
{
    const int32_t dims[4] = {input_dims->n, input_dims->h, input_dims->w, input_dims->c};
    const int32_t axes[4] = {axis_dims->n, axis_dims->h, axis_dims->w, axis_dims->c};
    const int32_t out[4] = {output_dims->n, output_dims->h, output_dims->w, output_dims->c};
    int32_t count = 1;
    for (int32_t d = 0; d < 4; ++d)
    {
        if (dims[d] < 1 || out[d] != (axes[d] ? 1 : dims[d]) || dims[d] > INT32_MAX / count)
        {
            return arm_reduce_sum_generic_f32_legacy(input_data, input_dims, axis_dims, output_data, output_dims);
        }
        count *= dims[d];
    }

    const uint32_t stride_w = (uint32_t)dims[3];
    const uint32_t stride_h = (uint32_t)dims[2] * stride_w;
    const uint32_t stride_n = (uint32_t)dims[1] * stride_h;
    const uint32_t step_n = axes[0] ? stride_n : 0;
    const uint32_t step_h = axes[1] ? stride_h : 0;
    const uint32_t step_w = axes[2] ? stride_w : 0;
    const int32_t limit_n = axes[0] ? dims[0] : 1;
    const int32_t limit_h = axes[1] ? dims[1] : 1;
    const int32_t limit_w = axes[2] ? dims[2] : 1;
    const int32_t limit_c = axes[3] ? dims[3] : 1;

    for (int32_t n = 0; n < out[0]; ++n)
        for (int32_t h = 0; h < out[1]; ++h)
            for (int32_t w = 0; w < out[2]; ++w)
                for (int32_t c = 0; c < out[3]; ++c)
                {
                    const uint32_t base =
                        (uint32_t)n * stride_n + (uint32_t)h * stride_h + (uint32_t)w * stride_w + (uint32_t)c;
                    // Preserve scalar order at -Ofast. Refs #484.
                    volatile float32_t acc = 0.0f;
                    uint32_t n_offset = base;
                    for (int32_t ni = 0; ni < limit_n; ++ni, n_offset += step_n)
                    {
                        uint32_t h_offset = n_offset;
                        for (int32_t hi = 0; hi < limit_h; ++hi, h_offset += step_h)
                        {
                            uint32_t w_offset = h_offset;
                            for (int32_t wi = 0; wi < limit_w; ++wi, w_offset += step_w)
                            {
                                const float32_t *row = input_data + w_offset;
                                for (int32_t ci = 0; ci < limit_c; ++ci)
                                {
                                    acc += row[ci];
                                }
                            }
                        }
                    }
                    *output_data++ = acc;
                }
    return ARM_CMSIS_NN_SUCCESS;
}

    #if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
// Channel-preserving spatial reductions. Refs #484.
static void arm_reduce_sum_spatial_mve_f32(const float32_t *input_data,
                                           int32_t outer,
                                           int32_t reduction,
                                           int32_t inner,
                                           float32_t *output_data)
{
    for (int32_t row = 0; row < outer; ++row)
    {
        for (int32_t c = 0; c < inner; c += 4)
        {
            const mve_pred16_t p = vctp32q((uint32_t)(inner - c));
            float32x4_t sum = vdupq_n_f32(0.0f);
            for (int32_t r = 0; r < reduction; ++r)
            {
                sum = vaddq(sum, vld1q_z(input_data + r * inner + c, p));
            }
            vstrwq_p(output_data + c, sum, p);
        }
        input_data += reduction * inner;
        output_data += inner;
    }
}

// Keep scalar FP controls and sequential accumulation under -Ofast. Refs #484.
static void arm_reduce_sum_spatial_scalar_f32(const float32_t *input_data,
                                              int32_t outer,
                                              int32_t reduction,
                                              int32_t inner,
                                              float32_t *output_data)
{
    for (int32_t row = 0; row < outer; ++row)
    {
        for (int32_t c = 0; c < inner; ++c)
        {
            float32_t sum = 0.0f;
            for (int32_t r = 0; r < reduction; ++r)
            {
                const float32_t value = input_data[r * inner + c];
                __ASM volatile("vadd.f32 %0, %0, %1" : "+t"(sum) : "t"(value));
            }
            output_data[c] = sum;
        }
        input_data += reduction * inner;
        output_data += inner;
    }
}
    #else
// Portable spatial accumulation. Refs #484.
static void arm_reduce_sum_spatial_portable_f32(const float32_t *input_data,
                                                float32_t *output_data,
                                                int32_t outer,
                                                int32_t reduction,
                                                int32_t inner)
{
    for (int32_t row = 0; row < outer; ++row)
    {
        for (int32_t c = 0; c < inner; ++c)
        {
            volatile float32_t sum = 0.0f;
            for (int32_t r = 0; r < reduction; ++r)
            {
                sum += input_data[r * inner + c];
            }
            output_data[c] = sum;
        }
        input_data += reduction * inner;
        output_data += inner;
    }
}
    #endif

// Fast path: reduced axes form a contiguous suffix -> row sums
static arm_cmsis_nn_status arm_reduce_sum_flatten_last_dims_f32(const float32_t *input_data,
                                                                float32_t *output_data,
                                                                int32_t outer_size,
                                                                int32_t inner_size)
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
        output_data[i] = arm_nn_vec_reduce_add_f32(vacc);
    #else
        float32_t acc = 0.0f;
        for (int32_t j = 0; j < inner_size; ++j)
        {
            acc += row[j];
        }
        output_data[i] = acc;
    #endif
    }

    return ARM_CMSIS_NN_SUCCESS;
}

/*
 * float32 sum over the specified axes. No requantization or clamp; NaN and
 * Inf propagate. Vector and scalar builds may differ in final ulps because
 * float accumulation order differs.
 *
 * Refer header file for details.
 *
 */
arm_cmsis_nn_status arm_reduce_sum_f32(const float32_t *input_data,
                                       const cmsis_nn_dims *input_dims,
                                       const cmsis_nn_dims *axis_dims,
                                       float32_t *output_data,
                                       const cmsis_nn_dims *output_dims)
{
    if (!input_data || !input_dims || !axis_dims || !output_data || !output_dims)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    if (!axis_dims->n && !axis_dims->c && (axis_dims->h || axis_dims->w) && input_dims->n > 0 && input_dims->h > 0 &&
        input_dims->w > 0 && input_dims->c > 0 && output_dims->n == input_dims->n &&
        output_dims->h == (axis_dims->h ? 1 : input_dims->h) && output_dims->w == (axis_dims->w ? 1 : input_dims->w) &&
        output_dims->c == input_dims->c)
    {
        const int32_t sizes[4] = {input_dims->n, input_dims->h, input_dims->w, input_dims->c};
        int32_t elements = 1;
        bool fits = true;
        for (int32_t i = 0; i < 4; ++i)
        {
            if (sizes[i] > INT32_MAX / elements)
            {
                fits = false;
                break;
            }
            elements *= sizes[i];
        }
        if (fits)
        {
            const int32_t outer = axis_dims->h ? input_dims->n : input_dims->n * input_dims->h;
            const int32_t reduction = (axis_dims->h ? input_dims->h : 1) * (axis_dims->w ? input_dims->w : 1);
            const int32_t inner = axis_dims->w ? input_dims->c : input_dims->w * input_dims->c;
    #if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
            uint32_t fpscr;
            __ASM volatile("vmrs %0, fpscr" : "=r"(fpscr));
            // MVE uses round-to-nearest and flush-to-zero. Refs #484.
            if ((fpscr & ((1u << 24) | (3u << 22))) == (1u << 24))
            {
                arm_reduce_sum_spatial_mve_f32(input_data, outer, reduction, inner, output_data);
            }
            else
            {
                arm_reduce_sum_spatial_scalar_f32(input_data, outer, reduction, inner, output_data);
            }
    #else
            arm_reduce_sum_spatial_portable_f32(input_data, output_data, outer, reduction, inner);
    #endif
            return ARM_CMSIS_NN_SUCCESS;
        }
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

        return arm_reduce_sum_flatten_last_dims_f32(input_data, output_data, outer_size, inner_size);
    }

    return arm_reduce_sum_generic_f32(input_data, input_dims, axis_dims, output_data, output_dims);
}

/**
 * @} end of Reduction group
 */

#endif /* ARM_NN_ENABLE_F32 */

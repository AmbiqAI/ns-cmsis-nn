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
static arm_cmsis_nn_status arm_reduce_sum_generic_f16_legacy(const float16_t *input_data,
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

// Refs #484.
static arm_cmsis_nn_status arm_reduce_sum_generic_f16(const float16_t *input_data,
                                                      const cmsis_nn_dims *input_dims,
                                                      const cmsis_nn_dims *axis_dims,
                                                      float16_t *output_data,
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
            return arm_reduce_sum_generic_f16_legacy(input_data, input_dims, axis_dims, output_data, output_dims);
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
                                const float16_t *row = input_data + w_offset;
                                for (int32_t ci = 0; ci < limit_c; ++ci)
                                {
                                    acc += (float32_t)row[ci];
                                }
                            }
                        }
                    }
                    *output_data++ = (float16_t)acc;
                }
    return ARM_CMSIS_NN_SUCCESS;
}

    #if defined(ARM_MATH_MVE_FLOAT16) && defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
// Spatial reductions preserve the innermost unreduced elements. Refs #484.
static void arm_reduce_sum_spatial_mve_f16(const float16_t *input_data,
                                           float16_t *output_data,
                                           int32_t outer,
                                           int32_t reduction,
                                           int32_t inner)
{
    for (int32_t row = 0; row < outer; ++row)
    {
        for (int32_t c = 0; c < inner; c += 8)
        {
            const mve_pred16_t p = vctp16q((uint32_t)(inner - c));
            float32x4_t even = vdupq_n_f32(0.0f);
            float32x4_t odd = vdupq_n_f32(0.0f);
            for (int32_t r = 0; r < reduction; ++r)
            {
                const float16x8_t value = vld1q_z(input_data + r * inner + c, p);
                even = vaddq(even, arm_nn_vcvtbq_f32_f16(value));
                odd = vaddq(odd, arm_nn_vcvttq_f32_f16(value));
            }
            float16x8_t result = arm_nn_vcvtbq_f16_f32(vdupq_n_f16(0.0f), even);
            result = arm_nn_vcvttq_f16_f32(result, odd);
            vstrhq_p(output_data + c, result, p);
        }
        input_data += reduction * inner;
        output_data += inner;
    }
}

// Scalar instructions retain directed rounding and FP32 accumulation. Refs #484.
static void arm_reduce_sum_spatial_scalar_f16(const float16_t *input_data,
                                              float16_t *output_data,
                                              int32_t outer,
                                              int32_t reduction,
                                              int32_t inner)
{
    for (int32_t row = 0; row < outer; ++row)
    {
        for (int32_t c = 0; c < inner; ++c)
        {
            float32_t sum = 0.0f;
            for (int32_t r = 0; r < reduction; ++r)
            {
                float32_t value;
                __ASM volatile("vcvtb.f32.f16 %0, %1" : "=t"(value) : "t"(input_data[r * inner + c]));
                __ASM volatile("vadd.f32 %0, %0, %1" : "+t"(sum) : "t"(value));
            }
            float16_t result;
            __ASM volatile("vcvtb.f16.f32 %0, %1" : "=t"(result) : "t"(sum));
            output_data[c] = result;
        }
        input_data += reduction * inner;
        output_data += inner;
    }
}
    #else
        #if !defined(__ARM_FP16_FORMAT_ALTERNATIVE)
// Portable spatial accumulation. Refs #484.
static void arm_reduce_sum_spatial_portable_f16(const float16_t *input_data,
                                                float16_t *output_data,
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
                sum += (float32_t)input_data[r * inner + c];
            }
            output_data[c] = (float16_t)sum;
        }
        input_data += reduction * inner;
        output_data += inner;
    }
}
        #endif
    #endif

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
            vacc_lo = vaddq(vacc_lo, arm_nn_vcvtbq_f32_f16(v));
            vacc_hi = vaddq(vacc_hi, arm_nn_vcvttq_f32_f16(v));
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

    if (!axis_dims->n && !axis_dims->c && (axis_dims->h || axis_dims->w) && output_dims->n == input_dims->n &&
        output_dims->c == input_dims->c && output_dims->h == (axis_dims->h ? 1 : input_dims->h) &&
        output_dims->w == (axis_dims->w ? 1 : input_dims->w))
    {
        int32_t elements = 1;
        bool bounded = true;
        for (int32_t d = 0; d < 4; ++d)
        {
            if (in_dims[d] <= 0 || in_dims[d] > INT32_MAX / elements)
            {
                bounded = false;
                break;
            }
            elements *= in_dims[d];
        }
        if (bounded)
        {
    #if defined(ARM_MATH_MVE_FLOAT16) && defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
            uint32_t fpscr;
            __ASM volatile("vmrs %0, fpscr" : "=r"(fpscr));
            // Keep alternative-half handling on the existing path. Refs #484.
            if (!(fpscr & (1u << 26)))
            {
                const int32_t outer = axis_dims->h ? input_dims->n : input_dims->n * input_dims->h;
                const int32_t reduction = (axis_dims->h ? input_dims->h : 1) * (axis_dims->w ? input_dims->w : 1);
                const int32_t inner = axis_dims->w ? input_dims->c : input_dims->w * input_dims->c;
                if (!(fpscr & (3u << 22)))
                {
                    arm_reduce_sum_spatial_mve_f16(input_data, output_data, outer, reduction, inner);
                }
                else
                {
                    arm_reduce_sum_spatial_scalar_f16(input_data, output_data, outer, reduction, inner);
                }
                return ARM_CMSIS_NN_SUCCESS;
            }
    #elif !defined(__ARM_FP16_FORMAT_ALTERNATIVE)
        #if defined(__ARM_FP) && (__ARM_FP & 4)
            uint32_t fpscr;
            __ASM volatile("vmrs %0, fpscr" : "=r"(fpscr));
            if (!(fpscr & (1u << 26)))
        #endif
            {
                const int32_t outer = axis_dims->h ? input_dims->n : input_dims->n * input_dims->h;
                const int32_t reduction = (axis_dims->h ? input_dims->h : 1) * (axis_dims->w ? input_dims->w : 1);
                const int32_t inner = axis_dims->w ? input_dims->c : input_dims->w * input_dims->c;
                arm_reduce_sum_spatial_portable_f16(input_data, output_data, outer, reduction, inner);
                return ARM_CMSIS_NN_SUCCESS;
            }
    #endif
        }
    }

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

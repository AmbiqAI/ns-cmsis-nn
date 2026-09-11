/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#ifndef ARM_NN_REDUCE_EXTREMA_FLT_H
#define ARM_NN_REDUCE_EXTREMA_FLT_H

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

#if ARM_NN_FLOAT_API_ENABLED

typedef struct
{
    int32_t dims[4];
    int32_t axes[4];
    int32_t out[4];
    int32_t stride[4];
    int32_t limit[4];
    int32_t step[4];
    int32_t input_count;
    int32_t output_count;
    int32_t outer;
    int32_t reduction;
    int32_t inner;
    bool copy;
    bool contiguous;
} arm_nn_extrema_plan;

static inline bool arm_nn_extrema_count(const int32_t dims[4], int32_t size, int32_t *count)
{
    *count = 1;
    for (int32_t d = 0; d < 4; ++d)
    {
        if (dims[d] == 0)
        {
            *count = 0;
            return true;
        }
    }
    for (int32_t d = 0; d < 4; ++d)
    {
        if (dims[d] > (INT32_MAX / size) / *count)
        {
            return false;
        }
        *count *= dims[d];
    }
    return true;
}

static inline bool arm_nn_extrema_prepare(const void *input,
                                          const cmsis_nn_dims *input_dims,
                                          const cmsis_nn_dims *axis_dims,
                                          void *output,
                                          const cmsis_nn_dims *output_dims,
                                          int32_t size,
                                          arm_nn_extrema_plan *plan)
{
    if (!input_dims || !axis_dims || !output_dims)
    {
        return false;
    }
    const int32_t dims[4] = {input_dims->n, input_dims->h, input_dims->w, input_dims->c};
    const int32_t axes[4] = {axis_dims->n, axis_dims->h, axis_dims->w, axis_dims->c};
    const int32_t out[4] = {output_dims->n, output_dims->h, output_dims->w, output_dims->c};
    int32_t first = 4;
    int32_t last = -1;
    for (int32_t d = 0; d < 4; ++d)
    {
        if (dims[d] < 0 || (axes[d] != 0 && axes[d] != 1) || out[d] != (axes[d] ? 1 : dims[d]))
        {
            return false;
        }
        plan->dims[d] = dims[d];
        plan->axes[d] = axes[d];
        plan->out[d] = out[d];
        if (axes[d])
        {
            if (first == 4)
            {
                first = d;
            }
            last = d;
        }
    }
    if (!arm_nn_extrema_count(dims, size, &plan->input_count) ||
        !arm_nn_extrema_count(out, size, &plan->output_count) || (plan->input_count && !input) ||
        (plan->output_count && !output))
    {
        return false;
    }
    plan->copy = first == 4;
    if (!plan->input_count || !plan->output_count || plan->copy)
    {
        return true;
    }
    int32_t stride = 1;
    for (int32_t d = 3; d >= 0; --d)
    {
        plan->stride[d] = stride;
        plan->limit[d] = axes[d] ? dims[d] : 1;
        plan->step[d] = axes[d] ? stride : 0;
        stride *= dims[d];
    }
    plan->contiguous = true;
    if (arm_reduce_get_flatten_suffix_start_from_arrays(dims, axes) >= 0)
    {
        last = 3;
    }
    plan->outer = 1;
    plan->reduction = 1;
    plan->inner = 1;
    for (int32_t d = 0; d < 4; ++d)
    {
        if (d < first)
        {
            plan->outer *= dims[d];
        }
        else if (d <= last)
        {
            plan->reduction *= dims[d];
            if (!axes[d] && dims[d] != 1)
            {
                plan->contiguous = false;
            }
        }
        else
        {
            plan->inner *= dims[d];
        }
    }
    return true;
}

static inline uint32_t arm_nn_extrema_load(const uint8_t *input, int32_t size)
{
    if (size == 2)
    {
        uint16_t bits;
        memcpy(&bits, input, sizeof(bits));
        return bits;
    }
    uint32_t bits;
    memcpy(&bits, input, sizeof(bits));
    return bits;
}

static inline void arm_nn_extrema_store(uint8_t *output, uint32_t bits, int32_t size)
{
    if (size == 2)
    {
        const uint16_t half = (uint16_t)bits;
        memcpy(output, &half, sizeof(half));
    }
    else
    {
        memcpy(output, &bits, sizeof(bits));
    }
}

// Integer ordering preserves subnormals and signed-zero ties. Refs #498.
static inline uint32_t arm_nn_extrema_key(uint32_t bits, uint32_t sign)
{
    if ((bits & (sign - 1)) == 0)
    {
        return sign;
    }
    return bits & sign ? (~bits & (sign | (sign - 1))) : (bits ^ sign);
}

static inline uint32_t
arm_nn_extrema_select(uint32_t acc, uint32_t value, uint32_t sign, uint32_t inf, uint32_t nan, bool maximum)
{
    if ((value & (sign - 1)) > inf || (acc & (sign - 1)) > inf)
    {
        return nan;
    }
    const uint32_t a = arm_nn_extrema_key(acc, sign);
    const uint32_t b = arm_nn_extrema_key(value, sign);
    return (maximum ? b > a : b < a) ? value : acc;
}

static inline void arm_nn_extrema_scalar(const uint8_t *input,
                                         uint8_t *output,
                                         const arm_nn_extrema_plan *plan,
                                         int32_t size,
                                         uint32_t sign,
                                         uint32_t inf,
                                         uint32_t nan,
                                         bool maximum)
{
    const uint32_t identity = maximum ? inf | sign : inf;
    if (plan->contiguous)
    {
        for (int32_t row = 0; row < plan->outer; ++row)
        {
            for (int32_t c = 0; c < plan->inner; ++c)
            {
                uint32_t acc = identity;
                for (int32_t r = 0; r < plan->reduction; ++r)
                {
                    const int32_t offset = (r * plan->inner + c) * size;
                    acc =
                        arm_nn_extrema_select(acc, arm_nn_extrema_load(input + offset, size), sign, inf, nan, maximum);
                }
                arm_nn_extrema_store(output + c * size, acc, size);
            }
            input += plan->reduction * plan->inner * size;
            output += plan->inner * size;
        }
        return;
    }
    for (int32_t i = 0; i < plan->output_count; ++i)
    {
        int32_t coordinate = i;
        int32_t base = 0;
        for (int32_t d = 3; d >= 0; --d)
        {
            base += (coordinate % plan->out[d]) * plan->stride[d];
            coordinate /= plan->out[d];
        }
        uint32_t acc = identity;
        for (int32_t n = 0; n < plan->limit[0]; ++n)
            for (int32_t h = 0; h < plan->limit[1]; ++h)
                for (int32_t w = 0; w < plan->limit[2]; ++w)
                    for (int32_t c = 0; c < plan->limit[3]; ++c)
                    {
                        const int32_t offset =
                            base + n * plan->step[0] + h * plan->step[1] + w * plan->step[2] + c * plan->step[3];
                        acc = arm_nn_extrema_select(
                            acc, arm_nn_extrema_load(input + offset * size, size), sign, inf, nan, maximum);
                    }
        arm_nn_extrema_store(output + i * size, acc, size);
    }
}

    #if ARM_NN_ENABLE_F32 && defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
static inline uint32x4_t arm_nn_extrema_key_f32(uint32x4_t value)
{
    const uint32x4_t sign = vdupq_n_u32(0x80000000u);
    uint32x4_t key = vpselq(vmvnq(value), veorq(value, sign), vcmpcsq_n_u32(value, 0x80000000u));
    return vpselq(sign, key, vcmpeqq_n_u32(vandq(value, vdupq_n_u32(0x7fffffffu)), 0));
}

static inline void
arm_nn_extrema_mve_f32(const float32_t *input, float32_t *output, const arm_nn_extrema_plan *plan, bool maximum)
{
    const uint32_t identity = maximum ? (0x7f800000u | 0x80000000u) : 0x7f800000u;
    if (plan->inner == 1)
    {
        const uint32_t seed = arm_nn_extrema_key(identity, 0x80000000u);
        for (int32_t row = 0; row < plan->outer; ++row)
        {
            uint32x4_t best = vdupq_n_u32(seed);
            mve_pred16_t any_nan = 0;
            for (int32_t r = 0; r < plan->reduction; r += 4)
            {
                const mve_pred16_t p = vctp32q((uint32_t)(plan->reduction - r));
                const uint32x4_t value = vreinterpretq_u32_f32(vld1q_z(input + r, p));
                any_nan |= vcmphiq_n_u32(vandq(value, vdupq_n_u32(0x7fffffffu)), 0x7f800000u) & p;
                const uint32x4_t key = vpselq(arm_nn_extrema_key_f32(value), vdupq_n_u32(seed), p);
                best = maximum ? vmaxq(best, key) : vminq(best, key);
            }
            const uint32_t key = maximum ? vmaxvq_u32(seed, best) : vminvq_u32(seed, best);
            uint32_t result = (key & 0x80000000u) ? (key ^ 0x80000000u) : (~key & (0x80000000u | 0x7fffffffu));
            if (any_nan)
            {
                result = 0x7fc00000u;
            }
            else if (key == 0x80000000u)
            {
                // Only zero has two encodings for the same key. Refs #498.
                for (int32_t r = 0; r < plan->reduction; ++r)
                {
                    const uint32_t value = arm_nn_extrema_load((const uint8_t *)(input + r), sizeof(*input));
                    if (!(value & 0x7fffffffu))
                    {
                        result = value;
                        break;
                    }
                }
            }
            arm_nn_extrema_store((uint8_t *)output++, result, sizeof(*output));
            input += plan->reduction;
        }
        return;
    }
    for (int32_t row = 0; row < plan->outer; ++row)
    {
        for (int32_t c = 0; c < plan->inner; c += 4)
        {
            const mve_pred16_t p = vctp32q((uint32_t)(plan->inner - c));
            uint32x4_t best = vdupq_n_u32(identity);
            uint32x4_t best_key = arm_nn_extrema_key_f32(best);
            mve_pred16_t any_nan = 0;
            for (int32_t r = 0; r < plan->reduction; ++r)
            {
                const uint32x4_t value = vreinterpretq_u32_f32(vld1q_z(input + r * plan->inner + c, p));
                const uint32x4_t key = arm_nn_extrema_key_f32(value);
                any_nan |= vcmphiq_n_u32(vandq(value, vdupq_n_u32(0x7fffffffu)), 0x7f800000u) & p;
                const mve_pred16_t better = maximum ? vcmphiq(key, best_key) : vcmphiq(best_key, key);
                best = vpselq(value, best, better);
                best_key = vpselq(key, best_key, better);
            }
            best = vpselq(vdupq_n_u32(0x7fc00000u), best, any_nan);
            vstrwq_p(output + c, vreinterpretq_f32_u32(best), p);
        }
        input += plan->reduction * plan->inner;
        output += plan->inner;
    }
}
    #endif

    #if ARM_NN_ENABLE_F16 && defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
static inline uint16x8_t arm_nn_extrema_key_f16(uint16x8_t value)
{
    const uint16x8_t sign = vdupq_n_u16(0x8000u);
    uint16x8_t key = vpselq(vmvnq(value), veorq(value, sign), vcmpcsq_n_u16(value, 0x8000u));
    return vpselq(sign, key, vcmpeqq_n_u16(vandq(value, vdupq_n_u16(0x7fffu)), 0));
}

static inline void
arm_nn_extrema_mve_f16(const float16_t *input, float16_t *output, const arm_nn_extrema_plan *plan, bool maximum)
{
    const uint32_t identity = maximum ? (0x7c00u | 0x8000u) : 0x7c00u;
    if (plan->inner == 1)
    {
        const uint32_t seed = arm_nn_extrema_key(identity, 0x8000u);
        for (int32_t row = 0; row < plan->outer; ++row)
        {
            uint16x8_t best = vdupq_n_u16(seed);
            mve_pred16_t any_nan = 0;
            for (int32_t r = 0; r < plan->reduction; r += 8)
            {
                const mve_pred16_t p = vctp16q((uint32_t)(plan->reduction - r));
                const uint16x8_t value = vreinterpretq_u16_f16(vld1q_z(input + r, p));
                any_nan |= vcmphiq_n_u16(vandq(value, vdupq_n_u16(0x7fffu)), 0x7c00u) & p;
                const uint16x8_t key = vpselq(arm_nn_extrema_key_f16(value), vdupq_n_u16(seed), p);
                best = maximum ? vmaxq(best, key) : vminq(best, key);
            }
            const uint32_t key = maximum ? vmaxvq_u16(seed, best) : vminvq_u16(seed, best);
            uint32_t result = (key & 0x8000u) ? (key ^ 0x8000u) : (~key & (0x8000u | 0x7fffu));
            if (any_nan)
            {
                result = 0x7e00u;
            }
            else if (key == 0x8000u)
            {
                // Only zero has two encodings for the same key. Refs #498.
                for (int32_t r = 0; r < plan->reduction; ++r)
                {
                    const uint32_t value = arm_nn_extrema_load((const uint8_t *)(input + r), sizeof(*input));
                    if (!(value & 0x7fffu))
                    {
                        result = value;
                        break;
                    }
                }
            }
            arm_nn_extrema_store((uint8_t *)output++, result, sizeof(*output));
            input += plan->reduction;
        }
        return;
    }
    for (int32_t row = 0; row < plan->outer; ++row)
    {
        for (int32_t c = 0; c < plan->inner; c += 8)
        {
            const mve_pred16_t p = vctp16q((uint32_t)(plan->inner - c));
            uint16x8_t best = vdupq_n_u16(identity);
            uint16x8_t best_key = arm_nn_extrema_key_f16(best);
            mve_pred16_t any_nan = 0;
            for (int32_t r = 0; r < plan->reduction; ++r)
            {
                const uint16x8_t value = vreinterpretq_u16_f16(vld1q_z(input + r * plan->inner + c, p));
                const uint16x8_t key = arm_nn_extrema_key_f16(value);
                any_nan |= vcmphiq_n_u16(vandq(value, vdupq_n_u16(0x7fffu)), 0x7c00u) & p;
                const mve_pred16_t better = maximum ? vcmphiq(key, best_key) : vcmphiq(best_key, key);
                best = vpselq(value, best, better);
                best_key = vpselq(key, best_key, better);
            }
            best = vpselq(vdupq_n_u16(0x7e00u), best, any_nan);
            vstrhq_p(output + c, vreinterpretq_f16_u16(best), p);
        }
        input += plan->reduction * plan->inner;
        output += plan->inner;
    }
}
    #endif

static inline arm_cmsis_nn_status arm_nn_extrema_run(const void *input,
                                                     const cmsis_nn_dims *input_dims,
                                                     const cmsis_nn_dims *axis_dims,
                                                     void *output,
                                                     const cmsis_nn_dims *output_dims,
                                                     int32_t size,
                                                     bool maximum)
{
    arm_nn_extrema_plan plan = {0};
    if (!arm_nn_extrema_prepare(input, input_dims, axis_dims, output, output_dims, size, &plan))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    if (!plan.output_count)
    {
        return ARM_CMSIS_NN_SUCCESS;
    }
    const uint32_t sign = size == 2 ? 0x8000u : 0x80000000u;
    const uint32_t inf = size == 2 ? 0x7c00u : 0x7f800000u;
    const uint32_t nan = size == 2 ? 0x7e00u : 0x7fc00000u;
    if (!plan.input_count)
    {
        for (int32_t i = 0; i < plan.output_count; ++i)
        {
            arm_nn_extrema_store((uint8_t *)output + i * size, maximum ? inf | sign : inf, size);
        }
    }
    else if (plan.copy)
    {
        memcpy(output, input, (size_t)plan.input_count * (size_t)size);
    }
    else
    {
    #if ARM_NN_ENABLE_F32 && defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
        if (size == 4 && plan.contiguous)
        {
            arm_nn_extrema_mve_f32(input, output, &plan, maximum);
            return ARM_CMSIS_NN_SUCCESS;
        }
    #endif
    #if ARM_NN_ENABLE_F16 && defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
        if (size == 2 && plan.contiguous)
        {
            arm_nn_extrema_mve_f16(input, output, &plan, maximum);
            return ARM_CMSIS_NN_SUCCESS;
        }
    #endif
        arm_nn_extrema_scalar(input, output, &plan, size, sign, inf, nan, maximum);
    }
    return ARM_CMSIS_NN_SUCCESS;
}

#endif
#endif

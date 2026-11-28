/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 */

#include "Internal/arm_nn_select_word.h"
#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"
#include <limits.h>

#if defined(ARM_MATH_MVEI) && !defined(ARM_MATH_AUTOVECTORIZE)
static inline uint32x4_t load_span(const unsigned char *data, uint32_t index, int32_t step, mve_pred16_t p)
{
    if (step == 1)
    {
        return arm_nn_mve_load_word_bits(data + index * sizeof(uint32_t), p);
    }
    uint32_t value;
    arm_nn_copy_word(&value, data);
    return vdupq_n_u32(value);
}
#endif

/* Each span is contiguous or scalar in every operand. */
__STATIC_FORCEINLINE void select_span(const bool *condition,
                                      const unsigned char *x,
                                      const unsigned char *y,
                                      unsigned char *output,
                                      int32_t count,
                                      int32_t cs,
                                      int32_t xs,
                                      int32_t ys)
{
#if defined(ARM_MATH_MVEI) && !defined(ARM_MATH_AUTOVECTORIZE)
    uint32_t i = 0;
    for (int32_t remaining = count; remaining > 0; remaining -= 4, i += 4)
    {
        const mve_pred16_t p = vctp32q((uint32_t)remaining);
        const uint32x4_t vc = cs == 1 ? vldrbq_z_u32((const uint8_t *)condition + i, p) : vdupq_n_u32(*condition);
        const uint32x4_t vx = load_span(x, i, xs, p);
        const uint32x4_t vy = load_span(y, i, ys, p);
        arm_nn_mve_store_word_bits(output + i * sizeof(uint32_t), vpselq(vx, vy, vcmpneq_n_u32(vc, 0)), p);
    }
#else
    for (int32_t i = 0; i < count; ++i)
    {
        const unsigned char *source = condition[cs == 1 ? i : 0] ? x + (xs == 1 ? i : 0) * sizeof(uint32_t)
                                                                 : y + (ys == 1 ? i : 0) * sizeof(uint32_t);
        arm_nn_copy_word(output + i * sizeof(uint32_t), source);
    }
#endif
}

arm_cmsis_nn_status arm_nn_select_word(const bool *condition,
                                       const unsigned char *x,
                                       const unsigned char *y,
                                       const cmsis_nn_select_v2_params *params,
                                       unsigned char *output)
{
    if (!condition || !x || !y || !params || !output || params->rank < 1 || params->rank > 8 || !params->output_shape ||
        !params->cond_strides || !params->x_strides || !params->y_strides)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    const int32_t rank = params->rank;
    const int32_t *shape = params->output_shape;
    const int32_t *strides[3] = {params->cond_strides, params->x_strides, params->y_strides};
    const int32_t limits[3] = {
        INT32_MAX - 1, INT32_MAX / (int32_t)sizeof(int32_t) - 1, INT32_MAX / (int32_t)sizeof(int32_t) - 1};
    int32_t max_offsets[3] = {0, 0, 0};
    int32_t count = 1;
    for (int32_t d = 0; d < rank; ++d)
    {
        if (shape[d] < 1 || count > (INT32_MAX / (int32_t)sizeof(int32_t)) / shape[d])
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
        count *= shape[d];
        for (int j = 0; j < 3; ++j)
        {
            const int32_t stride = strides[j][d];
            if (stride < 0 || (shape[d] > 1 && stride > (limits[j] - max_offsets[j]) / (shape[d] - 1)))
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            max_offsets[j] += (shape[d] - 1) * stride;
        }
    }

    int32_t outer_rank = rank;
    int32_t span = 1;
    int32_t steps[3] = {0, 0, 0};
    /* Fold the longest common contiguous/scalar suffix; no per-element division. */
    for (int32_t d = rank - 1; d >= 0; --d)
    {
        if (shape[d] == 1)
        {
            outer_rank = d;
            continue;
        }
        bool compatible = true;
        for (int j = 0; j < 3; ++j)
        {
            if (span == 1)
            {
                if (strides[j][d] > 1)
                {
                    compatible = false;
                }
            }
            else if (strides[j][d] != steps[j] * span)
            {
                compatible = false;
            }
        }
        if (!compatible)
        {
            break;
        }
        if (span == 1)
        {
            for (int j = 0; j < 3; ++j)
            {
                steps[j] = strides[j][d];
            }
        }
        span *= shape[d];
        outer_rank = d;
    }
    int32_t coordinates[8] = {0};
    int32_t offsets[3] = {0, 0, 0};
    for (int32_t written = 0; written < count; written += span)
    {
        select_span(condition + offsets[0],
                    x + offsets[1] * sizeof(uint32_t),
                    y + offsets[2] * sizeof(uint32_t),
                    output + written * sizeof(uint32_t),
                    span,
                    steps[0],
                    steps[1],
                    steps[2]);
        if (written == count - span)
        {
            break;
        }
        for (int32_t d = outer_rank - 1; d >= 0; --d)
        {
            if (coordinates[d] < shape[d] - 1)
            {
                ++coordinates[d];
                for (int j = 0; j < 3; ++j)
                {
                    offsets[j] += strides[j][d];
                }
                break;
            }
            for (int j = 0; j < 3; ++j)
            {
                offsets[j] -= coordinates[d] * strides[j][d];
            }
            coordinates[d] = 0;
        }
    }
    return ARM_CMSIS_NN_SUCCESS;
}

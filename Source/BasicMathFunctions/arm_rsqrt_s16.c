/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

static uint32_t arm_nn_isqrt_u32(uint32_t x)
{
    uint32_t res = 0;
    uint32_t bit = 1u << 30;

    while (bit > x)
    {
        bit >>= 2;
    }

    while (bit != 0)
    {
        if (x >= res + bit)
        {
            x -= res + bit;
            res = (res >> 1) + bit;
        }
        else
        {
            res >>= 1;
        }
        bit >>= 2;
    }

    return res;
}

/**
 * @addtogroup groupElementwise
 * @{
 */

/*
 * s16 elementwise reciprocal square root
 *
 * Refer header file for details.
 */
arm_cmsis_nn_status arm_rsqrt_s16(const int16_t *input,
                                 const int32_t input_offset,
                                 int16_t *output,
                                 const int32_t out_offset,
                                 const int32_t out_mult,
                                 const int32_t out_shift,
                                 const bool needs_rescale,
                                 const int32_t out_activation_min,
                                 const int32_t out_activation_max,
                                 const int32_t block_size)
{
    int32_t loop_count = block_size;

    while (loop_count > 0)
    {
        int32_t val = (int32_t)*input++ - input_offset;
        if (val < 0)
        {
            val = 0;
        }
        if (val > 0x7FFF)
        {
            val = 0x7FFF;
        }

        // Treat input as Q15, compute rsqrt in Q15.
        const uint32_t val_q15 = (uint32_t)val << 15;
        uint32_t sqrt_res = arm_nn_isqrt_u32(val_q15);
        int32_t rsqrt_res;
        if (sqrt_res == 0)
        {
            rsqrt_res = INT16_MAX;
        }
        else
        {
            // Q15 rsqrt: (2^15 / sqrt_res) in Q15 => (2^30 / sqrt_res)
            uint64_t numerator = (uint64_t)1 << 30;
            rsqrt_res = (int32_t)(numerator / sqrt_res);
        }

        int32_t acc;
        if (needs_rescale)
        {
            acc = arm_nn_requantize(rsqrt_res, out_mult, out_shift);
        }
        else
        {
            acc = rsqrt_res;
        }

        acc += out_offset;
        acc = MIN(acc, INT16_MAX);
        acc = MAX(acc, out_activation_min);
        acc = MIN(acc, out_activation_max);
        *output++ = (int16_t)acc;

        loop_count--;
    }

    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of Doxygen group
 */

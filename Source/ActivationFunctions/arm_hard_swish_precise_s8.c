/*
 * SPDX-FileCopyrightText: 2025 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_hard_swish_precise_s8.c
 * Description:  Hard Swish function for int8_t data type
 *
 * $Date:        09 September 2025
 * $Revision:    V.1.0.0
 *
 * Target Processor:  Cortex-M cores
 *
 * -------------------------------------------------------------------- */

#include "arm_nn_types.h"
#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

/**
 *  @ingroup groupNN
 */

/**
 * @addtogroup Acti
 * @{
 */

#if defined(ARM_MATH_MVEI)
    // Below this size the 4-lane path beats building the 256-entry table. Counts in #289.
    #define HARD_SWISH_PRECISE_S8_LUT_MIN_SIZE 1760

// Scalar formula for one element; must stay identical to the non-MVE loop below. See #289.
static inline int8_t hard_swish_precise_s8_elem(const int8_t in,
                                                const int32_t input_offset,
                                                const int32_t output_offset,
                                                const int32_t output_multiplier,
                                                const int32_t output_shift,
                                                const int32_t relu_q3,
                                                const int32_t relu_q6,
                                                const int32_t prescale)
{
    const int32_t x = (int32_t)in - input_offset;
    int32_t xr = ARM_NN_CLAMP(x + relu_q3, relu_q6, 0);
    xr = arm_nn_nonneg_divide_by_pot_s32(xr, prescale);
    int32_t y = x * xr;
    y = arm_nn_requantize(y, output_multiplier, output_shift);
    y += output_offset;
    y = ARM_NN_CLAMP(y, INT8_MAX, INT8_MIN);
    return (int8_t)y;
}
#endif

/*
 * Hard Swish activation function for int8_t data type.
 *
 * Refer header file for details.
 *
 */
arm_cmsis_nn_status arm_hard_swish_precise_s8(const int8_t *input,
                                              const int32_t input_offset,
                                              const int32_t output_offset,
                                              const int32_t output_multiplier,
                                              const int32_t output_shift,
                                              const int32_t relu_q3,
                                              const int32_t relu_q6,
                                              const int32_t prescale,
                                              int8_t *output,
                                              const int32_t output_size)
{

    if (!input || !output || output_size < 0)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    int32_t flat_size = output_size;

#if defined(ARM_MATH_MVEI)

    if (flat_size >= HARD_SWISH_PRECISE_S8_LUT_MIN_SIZE)
    {
        // Per-tensor function of one byte: tabulate the scalar formula once, then
        // gather 16 lanes per vector. Indexed by the input byte read as unsigned
        // (arm_sqrt_s8 layout). See #289.
        int8_t table[256];
        for (int32_t q = INT8_MIN; q <= INT8_MAX; q++)
        {
            table[(uint8_t)q] = hard_swish_precise_s8_elem(
                (int8_t)q, input_offset, output_offset, output_multiplier, output_shift, relu_q3, relu_q6, prescale);
        }

        while (flat_size >= 16)
        {
            const uint8x16_t idx = vreinterpretq_u8_s8(vld1q_s8(input));
            vst1q_s8(output, vldrbq_gather_offset_s8(table, idx));
            input += 16;
            output += 16;
            flat_size -= 16;
        }
        if (flat_size > 0)
        {
            const mve_pred16_t p = vctp8q((uint32_t)flat_size);
            const uint8x16_t idx = vreinterpretq_u8_s8(vld1q_z_s8(input, p));
            vst1q_p_s8(output, vldrbq_gather_offset_z_s8(table, idx, p), p);
        }
    }
    else
    {
        const int32_t qmin = INT8_MIN;
        const int32_t qmax = INT8_MAX;

        // Perform 4 operations in parallel
        uint32_t blkCnt = (flat_size + 3) / 4U;
        while (blkCnt > 0U)
        {
            // Load input and create predicate
            mve_pred16_t pred = vctp32q((uint32_t)flat_size);
            int32x4_t x = vldrbq_z_s32(input, pred);

            // Center input to remove zero-point
            x = vsubq_s32(x, vdupq_n_s32(input_offset));

            // Compute xr = clamp(x + relu_q3, 0, relu_q6)
            int32x4_t xr = vaddq_n_s32(x, relu_q3);
            xr = vmaxq_s32(xr, vdupq_n_s32(0));
            xr = vminq_s32(xr, vdupq_n_s32(relu_q6));

            // Prescale to prevent overflow in multiplication
            xr = vrshlq_n_s32(xr, -prescale);

            // Integer product x * xr (unit: s_in^2)
            int32x4_t y = vmulq_s32(x, xr);

            // Requantize the result (includes division by 6)
            y = arm_requantize_mve(y, output_multiplier, output_shift);
            y = vaddq_n_s32(y, output_offset);

            // Clamp and store
            y = vmaxq_s32(y, vdupq_n_s32(qmin));
            y = vminq_s32(y, vdupq_n_s32(qmax));
            vstrbq_p_s32(output, y, pred);

            // Increment pointers
            input += 4;
            output += 4;
            blkCnt -= 1;
            flat_size -= 4;
        }
    }

#else

    const int32_t qmin = INT8_MIN;
    const int32_t qmax = INT8_MAX;

    for (int32_t i = 0; i < flat_size; i++)
    {
        // Center input to remove zero-point
        int32_t x = (int32_t)input[i] - input_offset;

        // Compute xr = clamp(x + relu_q3, 0, relu_q6)
        int32_t xr = ARM_NN_CLAMP(x + relu_q3, relu_q6, 0);

        // Prescale to prevent overflow in multiplication
        xr = arm_nn_nonneg_divide_by_pot_s32(xr, prescale);

        // Integer product x * xr (unit: s_in^2)
        int32_t y = x * xr;

        // Requantize the result (includes division by 6)
        y = arm_nn_requantize(y, output_multiplier, output_shift);
        y += output_offset;

        // Clamp and store
        y = ARM_NN_CLAMP(y, qmax, qmin);
        output[i] = (int8_t)y;
    }

#endif

    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of Doxygen group
 */

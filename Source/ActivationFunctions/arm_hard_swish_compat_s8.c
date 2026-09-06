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
 * Title:        arm_hard_swish_compat_s8.c
 * Description:  Hard Swish function for int8_t compatible w/ TFLite
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
    // Below this size the 8-lane path beats building the 256-entry table. Counts in #289.
    #define HARD_SWISH_COMPAT_S8_LUT_MIN_SIZE 4160

// Scalar formula for one element; must stay identical to the non-MVE loop below. See #289.
static inline int8_t hard_swish_compat_s8_elem(const int8_t in,
                                               const int32_t input_offset,
                                               const int32_t output_offset,
                                               const int32_t output_multiplier_fp,
                                               const int32_t output_multiplier_exp,
                                               const int32_t relu_multiplier_fp,
                                               const int32_t relu_multiplier_exp)
{
    const int16_t x = (int16_t)((int32_t)in - input_offset);
    const int16_t hires = arm_nn_sat_lshift_s16(x, 7);
    const int16_t y_pre = arm_nn_sqrdmulh_s16(hires, (int16_t)output_multiplier_fp);
    int16_t rel = hires;

    if (relu_multiplier_exp > 0)
    {
        rel = arm_nn_sat_lshift_s16(rel, relu_multiplier_exp - 1);
        rel = arm_nn_sqrdmulh_s16(rel, (int16_t)relu_multiplier_fp);
        rel = arm_nn_sat_lshift_s16(rel, 1);
    }
    else if (relu_multiplier_exp < 0)
    {
        rel = arm_nn_sqrdmulh_s16(rel, relu_multiplier_fp);
        rel = arm_nn_divide_by_power_of_two_s16(rel, -relu_multiplier_exp);
    }
    else
    {
        rel = arm_nn_sqrdmulh_s16(rel, (int16_t)relu_multiplier_fp);
    }

    rel = (int16_t)(((int32_t)rel + 32768) >> 1);
    int16_t y = arm_nn_sqdmulh_s16(rel, y_pre);
    if (output_multiplier_exp < 0)
    {
        y = arm_nn_divide_by_power_of_two_s16(y, -output_multiplier_exp);
    }

    int32_t y8 = (int32_t)y + output_offset;
    y8 = ARM_NN_CLAMP(y8, INT8_MAX, INT8_MIN);
    return (int8_t)y8;
}
#endif

/*
 * Hard Swish activation function for int8_t data type.
 *
 * Refer header file for details.
 *
 */
arm_cmsis_nn_status arm_hard_swish_compat_s8(const int8_t *input,
                                             const int32_t input_offset,
                                             const int32_t output_offset,
                                             const int32_t output_multiplier_fp,
                                             const int32_t output_multiplier_exp,
                                             const int32_t relu_multiplier_fp,
                                             const int32_t relu_multiplier_exp,
                                             int8_t *output,
                                             const int32_t output_size)
{
    if (output_multiplier_exp > 0)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

#if defined(ARM_MATH_MVEI)

    if (output_size >= HARD_SWISH_COMPAT_S8_LUT_MIN_SIZE)
    {
        // Per-tensor function of one byte: tabulate the scalar formula once, then
        // gather 16 lanes per vector. Indexed by the input byte read as unsigned
        // (arm_sqrt_s8 layout). See #289.
        int8_t table[256];
        for (int32_t q = INT8_MIN; q <= INT8_MAX; q++)
        {
            table[(uint8_t)q] = hard_swish_compat_s8_elem((int8_t)q,
                                                          input_offset,
                                                          output_offset,
                                                          output_multiplier_fp,
                                                          output_multiplier_exp,
                                                          relu_multiplier_fp,
                                                          relu_multiplier_exp);
        }

        int32_t rem = output_size;
        while (rem >= 16)
        {
            const uint8x16_t idx = vreinterpretq_u8_s8(vld1q_s8(input));
            vst1q_s8(output, vldrbq_gather_offset_s8(table, idx));
            input += 16;
            output += 16;
            rem -= 16;
        }
        if (rem > 0)
        {
            const mve_pred16_t p = vctp8q((uint32_t)rem);
            const uint8x16_t idx = vreinterpretq_u8_s8(vld1q_z_s8(input, p));
            vst1q_p_s8(output, vldrbq_gather_offset_z_s8(table, idx, p), p);
        }
    }
    else
    {
        int32_t rem = output_size;

        while (rem > 0)
        {
            mve_pred16_t pred = vctp16q((uint32_t)rem);

            // Load input
            int16x8_t x = vldrbq_z_s16(input, pred);

            // x on integer domain (remove z_in)
            x = vsubq_s16(x, vdupq_n_s16(input_offset));

            // Hires = x << 7 (saturating)
            int16x8_t hires = vqshlq_n_s16(x, 7);

            // Preshift output scale: y_pre = SQRDMULH(hires, out_mul16)
            int16x8_t y_pre = vqrdmulhq_n_s16(hires, (int16_t)output_multiplier_fp);

            // Reluish path into [-1,1], then [0,1]
            int16x8_t rel = hires;

            if (relu_multiplier_exp > 0)
            {
                rel = vqshlq_s16(rel, vdupq_n_s16(relu_multiplier_exp - 1));
                rel = vqrdmulhq_n_s16(rel, (int16_t)relu_multiplier_fp);
                rel = vqshlq_n_s16(rel, 1);
            }
            else if (relu_multiplier_exp < 0)
            {
                rel = vqrdmulhq_n_s16(rel, relu_multiplier_fp);
                rel = arm_divide_by_power_of_two_mve_s16(rel, -relu_multiplier_exp);
            }
            else
            {
                rel = vqrdmulhq_n_s16(rel, (int16_t)relu_multiplier_fp);
            }

            // shift [-1,1] → [0,1]: (rel + 32768) >> 1  (rounded)
            rel = vrhaddq_s16(rel, vdupq_n_s16(32767));

            // y_pre is on preshift output scale. Multiply by relu using **non-rounded** SDHM.
            // vqdmulh floors; the scalar leg and TFLM truncate toward zero, so add one
            // where the product is negative and inexact (low 15 bits non-zero). See #289.
            int16x8_t y = vqdmulhq_s16(rel, y_pre);
            const uint16x8_t lo = vandq_u16(vreinterpretq_u16_s16(vmulq_s16(rel, y_pre)), vdupq_n_u16(0x7fff));
            const int16x8_t inexact = vreinterpretq_s16_u16(vminq_u16(lo, vdupq_n_u16(1)));
            y = vaddq_s16(y, vandq_s16(inexact, vshrq_n_s16(y_pre, 15)));

            // Finally apply output multiplier exponent
            if (output_multiplier_exp < 0)
            {
                y = arm_divide_by_power_of_two_mve_s16(y, -output_multiplier_exp);
            }

            // Add output offset and clamp the result
            y = vaddq_n_s16(y, output_offset);
            y = vmaxq_s16(y, vdupq_n_s16(INT8_MIN));
            y = vminq_s16(y, vdupq_n_s16(INT8_MAX));

            // Store output
            vstrbq_p_s16(output, y, pred);

            // Increment pointers
            input += 8;
            output += 8;
            rem -= 8;
        }
    }
#else

    for (int32_t i = 0; i < output_size; ++i)
    {
        // x on integer domain (remove z_in)
        int16_t x = (int16_t)((int32_t)input[i] - input_offset);

        // Hires = x << 7 (saturating)
        int16_t hires = arm_nn_sat_lshift_s16(x, 7);

        // Preshift output scale: y_pre = SQRDMULH(hires, out_mul16)
        int16_t y_pre = arm_nn_sqrdmulh_s16(hires, (int16_t)output_multiplier_fp);

        // Reluish path into [-1,1], then [0,1]
        int16_t rel = hires;

        if (relu_multiplier_exp > 0)
        {
            rel = arm_nn_sat_lshift_s16(rel, relu_multiplier_exp - 1);
            rel = arm_nn_sqrdmulh_s16(rel, (int16_t)relu_multiplier_fp);
            rel = arm_nn_sat_lshift_s16(rel, 1);
        }
        else if (relu_multiplier_exp < 0)
        {
            rel = arm_nn_sqrdmulh_s16(rel, relu_multiplier_fp);
            rel = arm_nn_divide_by_power_of_two_s16(rel, -relu_multiplier_exp);
        }
        else
        {
            rel = arm_nn_sqrdmulh_s16(rel, (int16_t)relu_multiplier_fp);
        }

        // shift [-1,1] → [0,1]: (rel + 32768) >> 1  (rounded)
        rel = (int16_t)(((int32_t)rel + 32768) >> 1);

        // y_pre is on preshift output scale. Multiply by relu using **non-rounded** SDHM
        int16_t y = arm_nn_sqdmulh_s16(rel, y_pre);

        // Finally apply output multiplier exponent
        if (output_multiplier_exp < 0)
        {
            y = arm_nn_divide_by_power_of_two_s16(y, -output_multiplier_exp);
        }

        // Add output offset and clamp the result
        int32_t y8 = (int32_t)y + output_offset;
        y8 = ARM_NN_CLAMP(y8, INT8_MAX, INT8_MIN);

        output[i] = (int8_t)y8;
    }

#endif /* ARM_MATH_MVEI */

    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of Doxygen group
 */

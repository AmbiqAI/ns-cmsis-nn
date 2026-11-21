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
 * Title:        arm_logistic_s16.c
 * Description:  Logistic activation function for s16 using partial look-up tables
 *
 * $Date:        19 January 2024
 * $Revision:    V.2.0.0
 *
 * Target Processor:  Cortex-M cores
 *
 * -------------------------------------------------------------------- */

#include "Internal/arm_nn_activation_lut_s16.h"
#include "arm_nn_tables.h"
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

/*
 * @brief Logistic activation function for s16
 *
 * @note  Refer header file for details.
 *
 */
arm_cmsis_nn_status arm_logistic_s16(const int16_t *input,
                                     int16_t *output,
                                     const int32_t input_size,
                                     int32_t input_multiplier,
                                     int32_t input_left_shift)
{
    // power-of-two case
    if (input_multiplier == 0)
    {
        input_multiplier = (int32_t)(3u << input_left_shift);
        input_left_shift = 0;
    }
    const int32_t round = (input_left_shift > 0) ? (1 << (input_left_shift - 1)) : 0;

#if defined(ARM_MATH_MVEI) && !defined(ARM_MATH_AUTOVECTORIZE)

    for (int32_t left = input_size; left > 0; left -= 4)
    {
        const mve_pred16_t active = vctp32q(left);
        const int32x4_t x = vldrhq_z_s32(input, active);
        const int32x4_t scaled =
            vshlq_s32(vaddq_n_s32(vmulq_n_s32(x, input_multiplier), round), vdupq_n_s32(-input_left_shift));
        const uint32x4_t magnitude = vreinterpretq_u32_s32(vabsq_s32(scaled));
        const uint32x4_t index = vshrq_n_u32(magnitude, 9);
        const mve_pred16_t lookup = arm_nn_activation_lut_predicate_s16(index, active);
        const uint32x4_t a = vldrhq_gather_shifted_offset_z_u32(sigmoid_table_uint16, index, lookup);
        const uint32x4_t b = vldrhq_gather_shifted_offset_z_u32(sigmoid_table_uint16 + 1, index, lookup);
        const uint32x4_t fraction = vandq_u32(magnitude, vdupq_n_u32(511));
        const uint32x4_t interp = vaddq_u32(vshlq_n_u32(a, 9), vmulq_u32(fraction, vsubq_u32(b, a)));
        const uint32x4_t rounded = vrshrq_n_u32(interp, 10);
        const int32x4_t pos = vpselq_s32(vreinterpretq_s32_u32(rounded), vdupq_n_s32(INT16_MAX), lookup);
        const int32x4_t neg = vsubq_s32(vdupq_n_s32(32768), pos);
        vstrhq_p_s32(output, vpselq_s32(pos, neg, vcmpgeq_n_s32(scaled, 0)), active);
        const int32_t advance = left < 4 ? left : 4;
        input += advance;
        output += advance;
    }

#elif defined(ARM_NN_ACTIVATION_S16_DSP_LUT)
    arm_nn_activation_dsp_s16(input, output, input_size, input_multiplier, input_left_shift, round, false);

#else

    for (int32_t i = 0; i < input_size; i++)
    {
        const int32_t input_data = arm_nn_mac_wrap_s32_bits(round, input[i], input_multiplier) >> input_left_shift;
        const uint32_t abs_input_data = input_data < 0 ? 0u - (uint32_t)input_data : (uint32_t)input_data;
        uint32_t uh = abs_input_data >> 9;
        uint32_t result;
        if (uh >= 255)
        {
            result = 0x7FFF << 10;
        }
        else
        {
            uint32_t ua = sigmoid_table_uint16[uh];
            uint32_t ub = sigmoid_table_uint16[uh + 1];
            uint32_t ut = abs_input_data & 0x1FF;
            result = (ua << 9) + ut * (ub - ua);
        }
        result = (input_data >= 0) ? (result + (1 << 9)) : ((1 << (16 + 9)) - result + (1 << 9) - 1);
        result >>= 10;
        output[i] = (int16_t)result;
    }

#endif

    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of Doxygen group
 */

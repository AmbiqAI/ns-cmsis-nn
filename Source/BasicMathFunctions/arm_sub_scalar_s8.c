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
 * Title:        arm_sub_scalar_s8
 * Description:  Elementwise subtraction with scalar input
 *
 * $Date:        4 November 2025
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

#include <stdbool.h>

/**
 *  @ingroup Public
 */

/**
 * @addtogroup groupElementwise
 * @{
 */

static arm_cmsis_nn_status arm_sub_scalar_s8_core(const int8_t *scalar_vect,
                                                  const int8_t *vector_vect,
                                                  const int32_t scalar_offset,
                                                  const int32_t scalar_mult,
                                                  const int32_t scalar_shift,
                                                  const int32_t vector_offset,
                                                  const int32_t vector_mult,
                                                  const int32_t vector_shift,
                                                  const int32_t left_shift,
                                                  int8_t *output,
                                                  const int32_t out_offset,
                                                  const int32_t out_mult,
                                                  const int32_t out_shift,
                                                  const int32_t out_activation_min,
                                                  const int32_t out_activation_max,
                                                  const int32_t block_size,
                                                  const bool scalar_minus_vector)
{
    // Compute scalar contribution once
    int32_t scalar_val = ((int32_t)scalar_vect[0] + scalar_offset) << left_shift;
    scalar_val = arm_nn_requantize(scalar_val, scalar_mult, scalar_shift);

#if defined(ARM_MATH_MVEI)
    int32_t count = block_size;

    const int32x4_t scalar_vec = vdupq_n_s32(scalar_val);

    while (count > 0)
    {
        int32x4_t vect;
        int32x4_t rst;

        mve_pred16_t p = vctp32q((uint32_t)count);

        vect = vldrbq_z_s32(vector_vect, p);
        vect = vaddq_s32(vect, vdupq_n_s32(vector_offset));
        vect = vshlq_r_s32(vect, left_shift);
        vect = arm_requantize_mve(vect, vector_mult, vector_shift);

        rst = scalar_minus_vector ? vsubq_s32(scalar_vec, vect) : vsubq_s32(vect, scalar_vec);
        rst = arm_requantize_mve(rst, out_mult, out_shift);

        rst = vaddq_n_s32(rst, out_offset);

        rst = vmaxq_s32(rst, vdupq_n_s32(out_activation_min));
        rst = vminq_s32(rst, vdupq_n_s32(out_activation_max));

        vector_vect += 4;
        vstrbq_p_s32(output, rst, p);

        output += 4;
        count -= 4;
    }

#else
    int32_t loop_count;
    int32_t input_2;
    int32_t diff;

    #if defined(ARM_MATH_DSP)

    int32_t scalar_1, scalar_2, scalar_3, scalar_4;
    int32_t a_vec, b_vec, a_tmp, b_tmp;

    int32_t scalar_offset_packed, vector_offset_packed;

    int8_t r1, r2, r3, r4;

    scalar_offset_packed = (scalar_offset << 16U) | (scalar_offset & 0x0FFFFL);
    vector_offset_packed = (vector_offset << 16U) | (vector_offset & 0x0FFFFL);

    loop_count = block_size >> 2;

    // Expand scalar value four times and apply offset
    read_and_pad_reordered_scalar(scalar_vect[0], &b_tmp, &a_tmp);
    a_tmp = SADD16(a_tmp, scalar_offset_packed);
    b_tmp = SADD16(b_tmp, scalar_offset_packed);

    scalar_1 = (b_tmp & 0x0FFFF) << left_shift;
    scalar_1 = arm_nn_requantize(scalar_1, scalar_mult, scalar_shift);
    scalar_3 = ((b_tmp >> 16) & 0x0FFFF) << left_shift;
    scalar_3 = arm_nn_requantize(scalar_3, scalar_mult, scalar_shift);
    scalar_2 = (a_tmp & 0x0FFFF) << left_shift;
    scalar_2 = arm_nn_requantize(scalar_2, scalar_mult, scalar_shift);
    scalar_4 = ((a_tmp >> 16) & 0x0FFFF) << left_shift;
    scalar_4 = arm_nn_requantize(scalar_4, scalar_mult, scalar_shift);

    while (loop_count > 0)
    {
        vector_vect = read_and_pad_reordered(vector_vect, &b_vec, &a_vec);

        a_vec = SADD16(a_vec, vector_offset_packed);
        b_vec = SADD16(b_vec, vector_offset_packed);

        /* Diff 1 */
        input_2 = (b_vec & 0x0FFFF) << left_shift;
        input_2 = arm_nn_requantize(input_2, vector_mult, vector_shift);

        diff = scalar_minus_vector ? (scalar_1 - input_2) : (input_2 - scalar_1);
        diff = arm_nn_requantize(diff, out_mult, out_shift);
        diff += out_offset;
        diff = MAX(diff, out_activation_min);
        diff = MIN(diff, out_activation_max);
        r1 = (int8_t)diff;

        /* Diff 3 */
        input_2 = ((b_vec >> 16) & 0x0FFFF) << left_shift;
        input_2 = arm_nn_requantize(input_2, vector_mult, vector_shift);

        diff = scalar_minus_vector ? (scalar_3 - input_2) : (input_2 - scalar_3);
        diff = arm_nn_requantize(diff, out_mult, out_shift);
        diff += out_offset;
        diff = MAX(diff, out_activation_min);
        diff = MIN(diff, out_activation_max);
        r3 = (int8_t)diff;

        /* Diff 2 */
        input_2 = (a_vec & 0x0FFFF) << left_shift;
        input_2 = arm_nn_requantize(input_2, vector_mult, vector_shift);

        diff = scalar_minus_vector ? (scalar_2 - input_2) : (input_2 - scalar_2);
        diff = arm_nn_requantize(diff, out_mult, out_shift);
        diff += out_offset;
        diff = MAX(diff, out_activation_min);
        diff = MIN(diff, out_activation_max);
        r2 = (int8_t)diff;

        /* Diff 4 */
        input_2 = ((a_vec >> 16) & 0x0FFFF) << left_shift;
        input_2 = arm_nn_requantize(input_2, vector_mult, vector_shift);

        diff = scalar_minus_vector ? (scalar_4 - input_2) : (input_2 - scalar_4);
        diff = arm_nn_requantize(diff, out_mult, out_shift);
        diff += out_offset;
        diff = MAX(diff, out_activation_min);
        diff = MIN(diff, out_activation_max);
        r4 = (int8_t)diff;

        arm_nn_write_s8x4_ia(&output, PACK_S8x4_32x1(r1, r2, r3, r4));

        loop_count--;
    }

    loop_count = block_size & 0x3;
    #else
    loop_count = block_size;
    #endif

    while (loop_count > 0)
    {
        /* C = scalar +/- vector */

        input_2 = (*vector_vect++ + vector_offset) << left_shift;
        input_2 = arm_nn_requantize(input_2, vector_mult, vector_shift);

        diff = scalar_minus_vector ? (scalar_val - input_2) : (input_2 - scalar_val);
        diff = arm_nn_requantize(diff, out_mult, out_shift);
        diff += out_offset;

        diff = MAX(diff, out_activation_min);
        diff = MIN(diff, out_activation_max);

        *output++ = (int8_t)diff;

        /* Decrement loop counter */
        loop_count--;
    }

#endif /* ARM_MATH_MVEI */

    return ARM_CMSIS_NN_SUCCESS;
}


/*
 * s8 elementwise subtract with scalar (scalar - vector)
 *
 * Refer header file for details.
 *
 */
arm_cmsis_nn_status arm_sub_scalar_s8(const int8_t *input_1_vect,
                                      const int8_t *input_2_vect,
                                      const int32_t input_1_offset,
                                      const int32_t input_1_mult,
                                      const int32_t input_1_shift,
                                      const int32_t input_2_offset,
                                      const int32_t input_2_mult,
                                      const int32_t input_2_shift,
                                      const int32_t left_shift,
                                      int8_t *output,
                                      const int32_t out_offset,
                                      const int32_t out_mult,
                                      const int32_t out_shift,
                                      const int32_t out_activation_min,
                                      const int32_t out_activation_max,
                                      const int32_t block_size)
{
    return arm_sub_scalar_s8_core(input_1_vect,
                                  input_2_vect,
                                  input_1_offset,
                                  input_1_mult,
                                  input_1_shift,
                                  input_2_offset,
                                  input_2_mult,
                                  input_2_shift,
                                  left_shift,
                                  output,
                                  out_offset,
                                  out_mult,
                                  out_shift,
                                  out_activation_min,
                                  out_activation_max,
                                  block_size,
                                  true);
}

/*
 * Internal helper for vector - scalar subtraction
 */
arm_cmsis_nn_status arm_rsub_scalar_s8(const int8_t *input_1_vect,
                                       const int8_t *input_2_vect,
                                       const int32_t input_1_offset,
                                       const int32_t input_1_mult,
                                       const int32_t input_1_shift,
                                       const int32_t input_2_offset,
                                       const int32_t input_2_mult,
                                       const int32_t input_2_shift,
                                       const int32_t left_shift,
                                       int8_t *output,
                                       const int32_t out_offset,
                                       const int32_t out_mult,
                                       const int32_t out_shift,
                                       const int32_t out_activation_min,
                                       const int32_t out_activation_max,
                                       const int32_t block_size)
{
    return arm_sub_scalar_s8_core(input_2_vect,
                                  input_1_vect,
                                  input_2_offset,
                                  input_2_mult,
                                  input_2_shift,
                                  input_1_offset,
                                  input_1_mult,
                                  input_1_shift,
                                  left_shift,
                                  output,
                                  out_offset,
                                  out_mult,
                                  out_shift,
                                  out_activation_min,
                                  out_activation_max,
                                  block_size,
                                  false);
}

/**
 * @} end of Doxygen group
 */

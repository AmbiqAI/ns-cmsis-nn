/*
 * SPDX-FileCopyrightText: Copyright 2010-2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_elementwise_mul_s8
 * Description:  Element wise multiplication
 *
 * $Date:        20 January 2023
 * $Revision:    V.2.2.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

/**
 *  @ingroup Public
 */

/**
 * @addtogroup groupElementwise
 * @{
 */

/*
 * s8 element wise multiplication of two vectors
 *
 * Refer header file for details.
 *
 */
arm_cmsis_nn_status arm_elementwise_mul_s8(const int8_t *input_1_vect,
                                           const int8_t *input_2_vect,
                                           const int32_t input_1_offset,
                                           const int32_t input_2_offset,
                                           int8_t *output,
                                           const int32_t out_offset,
                                           const int32_t out_mult,
                                           const int32_t out_shift,
                                           const int32_t out_activation_min,
                                           const int32_t out_activation_max,
                                           const int32_t block_size)
{

#if defined(ARM_MATH_MVEI)
    // Process 16 elements per iteration (full 128-bit utilization for int8_t)
    uint32_t nonpredicate_loops = block_size / 16;
    uint32_t predicate_elements = block_size % 16;

    // Pre-compute constants for vectorized operations
    const int32x4_t out_mult_vec = vdupq_n_s32(out_mult);
    const int32x4_t out_shift_vec = vdupq_n_s32(out_shift);
    const int32x4_t out_min_vec = vdupq_n_s32(out_activation_min);
    const int32x4_t out_max_vec = vdupq_n_s32(out_activation_max);
    const int16_t casted_input_1_offset = (int16_t)input_1_offset;
    const int16_t casted_input_2_offset = (int16_t)input_2_offset;

    // Main loop - process 16 elements at a time (4 vectors of 4 elements each)
    for (size_t i = 0; i < nonpredicate_loops; i++)
    {
        //swap to instead load 8 elements at a time
        const uint16x8_t offset_16x8 = {0,4,1,5,2,6,3,7};
        int16x8_t in16_1_a = vldrbq_gather_offset_s16(input_1_vect, offset_16x8);
        in16_1_a = vaddq_n_s16(in16_1_a, casted_input_1_offset);
        int16x8_t in16_2_a = vldrbq_gather_offset_s16(input_2_vect,offset_16x8);
        in16_2_a = vaddq_n_s16(in16_2_a, casted_input_2_offset);



        int32x4_t res_a = vmullbq_int_s16(in16_1_a, in16_2_a);
        res_a = arm_requantize_mve_32x4(res_a, out_mult_vec, out_shift_vec);

        int16x8_t in16_1_b = vldrbq_gather_offset_s16(input_1_vect + 8, offset_16x8);
        in16_1_b = vaddq_n_s16(in16_1_b, casted_input_1_offset);

        int32x4_t res_b = vmulltq_int_s16(in16_1_a, in16_2_a);
        res_b = arm_requantize_mve_32x4(res_b, out_mult_vec, out_shift_vec);
        int16x8_t in16_2_b = vldrbq_gather_offset_s16(input_2_vect + 8,offset_16x8);
        in16_2_b = vaddq_n_s16(in16_2_b, casted_input_2_offset);

        int32x4_t res_c = vmullbq_int_s16(in16_1_b, in16_2_b);
        res_c = arm_requantize_mve_32x4(res_c, out_mult_vec, out_shift_vec);

        int32x4_t res_d = vmulltq_int_s16(in16_1_b, in16_2_b);
        res_d = arm_requantize_mve_32x4(res_d, out_mult_vec, out_shift_vec);

        // Requantize all batches

        // Add offset and apply activation functions
        res_a = vaddq_n_s32(res_a, out_offset);
        res_b = vaddq_n_s32(res_b, out_offset);
        res_c = vaddq_n_s32(res_c, out_offset);
        res_d = vaddq_n_s32(res_d, out_offset);

        int16x8_t half0 = vdupq_n_s16(0);
        //narrow from 32 bit to 16 bit
        half0 = vqmovnbq_s32(half0, res_a); 
        half0 = vqmovntq_s32(half0, res_b); 

        int16x8_t half1 = vdupq_n_s16(0);

        half1 = vqmovnbq_s32(half1, res_c);
        half1 = vqmovntq_s32(half1, res_d);
        
        //narrow from 16 bit to 8 bit
        int8x16_t out_vec = vdupq_n_s8(0);
        out_vec = vqmovnbq_s16(out_vec, half0); // saturate bottom half
        out_vec = vqmovntq_s16(out_vec, half1); // saturate top half

        uint8x16_t big_offset = {0, 8, 4, 12,1,9,5,13,2,10,6,14,3,11,7,15};

        vstrbq_scatter_offset_s8(output, big_offset, out_vec);

        input_1_vect += 16;
        input_2_vect += 16;
        output += 16;
    }

    // Handle remaining elements with predication
    if (predicate_elements > 0) {
        // Process remaining elements in chunks of 4
        while (predicate_elements >= 4) {
            mve_pred16_t p = vctp32q(4);
            int32x4_t input_1 = vldrbq_z_s32(input_1_vect, p);
            int32x4_t input_2 = vldrbq_z_s32(input_2_vect, p);
            
            input_1 = vaddq_n_s32(input_1, input_1_offset);
            input_2 = vaddq_n_s32(input_2, input_2_offset);
            
            int32x4_t res_0 = vmulq_s32(input_1, input_2);
            res_0 = arm_requantize_mve_32x4(res_0, out_mult_vec, out_shift_vec);
            res_0 = vaddq_n_s32(res_0, out_offset);
            res_0 = vmaxq_s32(res_0, out_min_vec);
            res_0 = vminq_s32(res_0, out_max_vec);
            
            vstrbq_p_s32(output, res_0, p);
            
            input_1_vect += 4;
            input_2_vect += 4;
            output += 4;
            predicate_elements -= 4;
        }
        
        // Final iteration for 1-3 remaining elements
        if (predicate_elements > 0) {
            mve_pred16_t p = vctp32q(predicate_elements);
            int32x4_t input_1 = vldrbq_z_s32(input_1_vect, p);
            int32x4_t input_2 = vldrbq_z_s32(input_2_vect, p);
            
            input_1 = vaddq_n_s32(input_1, input_1_offset);
            input_2 = vaddq_n_s32(input_2, input_2_offset);
            
            int32x4_t res_0 = vmulq_s32(input_1, input_2);
            res_0 = arm_requantize_mve_32x4(res_0, out_mult_vec, out_shift_vec);
            res_0 = vaddq_n_s32(res_0, out_offset);
            res_0 = vmaxq_s32(res_0, out_min_vec);
            res_0 = vminq_s32(res_0, out_max_vec);
            
            vstrbq_p_s32(output, res_0, p);
        }
    }
#else
    int32_t loop_count;
    int32_t input_1;
    int32_t input_2;
    int32_t mul_res;

    #if defined(ARM_MATH_DSP)
    int32_t a_1, b_1, a_2, b_2;

    int32_t offset_1_packed, offset_2_packed;

    int8_t r1, r2, r3, r4;

    offset_1_packed = (input_1_offset << 16U) | (input_1_offset & 0x0FFFFL);
    offset_2_packed = (input_2_offset << 16U) | (input_2_offset & 0x0FFFFL);

    loop_count = block_size >> 2;

    while (loop_count > 0)
    {
        /* 4 outputs are calculated in one loop. The order of calculation is follows the order of output sign extension
           intrinsic */
        input_1_vect = read_and_pad_reordered(input_1_vect, &b_1, &a_1);
        input_2_vect = read_and_pad_reordered(input_2_vect, &b_2, &a_2);

        a_1 = SADD16(a_1, offset_1_packed);
        b_1 = SADD16(b_1, offset_1_packed);

        a_2 = SADD16(a_2, offset_2_packed);
        b_2 = SADD16(b_2, offset_2_packed);

        /* Mul 1 */
        mul_res = SMULBB(b_1, b_2);
        mul_res = arm_nn_requantize(mul_res, out_mult, out_shift) + out_offset;

        mul_res = MAX(mul_res, out_activation_min);
        mul_res = MIN(mul_res, out_activation_max);
        r1 = (int8_t)mul_res;

        /* Mul 3 */
        mul_res = SMULTT(b_1, b_2);
        mul_res = arm_nn_requantize(mul_res, out_mult, out_shift) + out_offset;
        mul_res = MAX(mul_res, out_activation_min);
        mul_res = MIN(mul_res, out_activation_max);
        r3 = (int8_t)mul_res;

        /* Mul 2 */
        mul_res = SMULBB(a_1, a_2);
        mul_res = arm_nn_requantize(mul_res, out_mult, out_shift) + out_offset;
        mul_res = MAX(mul_res, out_activation_min);
        mul_res = MIN(mul_res, out_activation_max);
        r2 = (int8_t)mul_res;

        /* Mul 4 */
        mul_res = SMULTT(a_1, a_2);
        mul_res = arm_nn_requantize(mul_res, out_mult, out_shift) + out_offset;
        mul_res = MAX(mul_res, out_activation_min);
        mul_res = MIN(mul_res, out_activation_max);
        r4 = (int8_t)mul_res;

        arm_nn_write_s8x4_ia(&output, PACK_S8x4_32x1(r1, r2, r3, r4));

        loop_count--;
    }

    loop_count = block_size & 0x3;
    #else
    loop_count = block_size;
    #endif

    while (loop_count > 0)
    {
        /* C = A * B */

        input_1 = *input_1_vect++ + input_1_offset;
        input_2 = *input_2_vect++ + input_2_offset;

        mul_res = input_1 * input_2;
        mul_res = arm_nn_requantize(mul_res, out_mult, out_shift) + out_offset;

        mul_res = MAX(mul_res, out_activation_min);
        mul_res = MIN(mul_res, out_activation_max);

        *output++ = (int8_t)mul_res;

        /* Decrement loop counter */
        loop_count--;
    }
#endif
    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of Doxygen group
 */

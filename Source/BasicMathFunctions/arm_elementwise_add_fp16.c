/*
 * SPDX-FileCopyrightText: Copyright 2022 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
 * Title:        arm_elementwise_add_fp16
 * Description:  Elementwise add
 *
 * $Date:        11 Feb 2025
 * $Revision:    V.1.0.0
 *
 * Target Processor:  Cortex-M CPUs
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
  * fp16 elementwise add
  *
  * Refer header file for details.
  *
  */

 arm_cmsis_nn_status arm_elementwise_add_fp16(
    const float16_t *input_1_vect,
    const float16_t *input_2_vect,
    float16_t *output,
    const float16_t out_activation_min,
    const float16_t out_activation_max,
    const int32_t block_size
) {

#if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)

    int32_t count = block_size;

    while (count > 0)
    {

        mve_pred16_t pred = vctp32q(count);

        float16x8_t vect_1 = vld1q_z_f16(input_1_vect, pred);
        float16x8_t vect_2 = vld1q_z_f16(input_2_vect, pred);
        float16x8_t rst = vaddq_x_f16(input_1_vect, input_2_vect, pred);

        input_1_vect += 8;
        input_2_vect += 8;

        rst = vminnmq_x_f16(rst, vdupq_n_s32(out_activation_max));
        rst = vmaxnmq_x_f16(rst, vdupq_n_s32(out_activation_min));

        vstrhq_p_f16(output, rst, pred);

        output += 8;
        count -= 8;
    }

 #else  // #if defined(ARM_MATH_MVEI)
    float16_t sum;

    for(int32_t i = 0; i < block_size; i++)
    {
        sum = input_1_vect[i] + input_2_vect[i];
        sum = MAX(sum, out_activation_min);
        sum = MIN(sum, out_activation_max);
        output[i] = sum;
    }

#endif // #if defined(ARM_MATH_MVEI)
    return (ARM_CMSIS_NN_SUCCESS);
 }

 /**
  * @} end of Doxygen group
  */

/*
 * SPDX-FileCopyrightText: Copyright 2010-2019, 2022 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
 * Title:        arm_relu6_s8.c
 * Description:  Basic s8 version of ReLU6
 *
 * $Date:        26 October 2022
 * $Revision:    V.1.0.2
 *
 * Target Processor:  Cortex-M cores
 *
 * -------------------------------------------------------------------- */

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

/**
 *  @ingroup Public
 */

/**
 * @addtogroup Acti
 * @{
 */


/*
* ReLU6 activation function for int8_t data type.
*
* Refer header file for details.
*
*/
arm_cmsis_nn_status arm_relu6_s8(
    const int8_t *input,
    const int8_t lower,
    const int8_t upper,
    int8_t *output,
    const int32_t output_size)
{

int32_t flat_size = output_size;

#if defined(ARM_MATH_MVEI)

    __ASM volatile(
        " .p2align 2                             \n"
        "   wlstp.8         lr, %[cnt], 1f       \n"
        "   vdup.8          q0, %[lower]         \n"
        "   vdup.8          q1, %[upper]         \n"
        "2:                                      \n"
        "   vldrb.8         q2, [%[in]], #16     \n"
        "   vmax.s8         q2, q2, q0           \n"
        "   vmin.s8         q2, q2, q1           \n"
        "   vstrb.8         q2, [%[out]], #16    \n"
        "   letp            lr, 2b               \n"
        "1:                                      \n"
        : [in] "+r"(input), [out] "+r"(output)
        : [lower] "r"(lower), [upper] "r"(upper), [cnt] "r"(flat_size)
        : "q0", "q1", "q2", "memory", "r14"
    );

#else

    for (int i = 0; i < flat_size; ++i)
    {
        output[i] = CLAMP(input[i], upper, lower);
    }

#endif

return ARM_CMSIS_NN_SUCCESS;
}


/*
 *  Basic ReLU6 function
 *
 * Refer to header file for details.
 *
 */

void arm_relu6_default_s8(int8_t *data, uint16_t size)
{
    arm_relu6_s8(data, 0, 6, data, size);
}


/**
 * @} end of Acti group
 */

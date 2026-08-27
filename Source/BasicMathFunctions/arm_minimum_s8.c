/*
 * SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-FileCopyrightText: Copyright 2024-2026 Ambiq <opensource@ambiq.com>
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
 * Title:        arm_minimum_s8
 * Description:  Minimum and Maximum
 *
 * $Date:        08 October 2024
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "Internal/arm_nn_broadcast_walk.h"
#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

/**
 *  @ingroup Public
 */

/**
 * @addtogroup minimumMaximum
 * @{
 */

static arm_cmsis_nn_status
arm_min_no_broadcast_s8(const int8_t *input_1, const int8_t *input_2, int8_t *output, int32_t flat_size)
{
#if defined(ARM_MATH_MVEI)

    __ASM volatile(" .p2align 2                             \n"
                   "   wlstp.8         lr, %[cnt], 1f       \n"
                   "2:                                      \n"
                   "   vldrb.8         q0, [%[in1]], #16    \n"
                   "   vldrb.8         q1, [%[in2]], #16    \n"
                   "   vmin.s8         q2, q0, q1           \n"
                   "   vstrb.8         q2, [%[out]], #16    \n"
                   "   letp            lr, 2b               \n"
                   "1:                                      \n"
                   : [in1] "+r"(input_1), [in2] "+r"(input_2), [out] "+r"(output)
                   : [cnt] "r"(flat_size)
                   : "q0", "q1", "q2", "memory", "r14");

#else
    while (flat_size > 0)
    {
        int8_t in1 = *input_1++;
        int8_t in2 = *input_2++;
        *output++ = in1 >= in2 ? in2 : in1;
        --flat_size;
    }
#endif

    return ARM_CMSIS_NN_SUCCESS;
}

static arm_cmsis_nn_status
arm_min_scalar_s8(const int8_t *input_1, const int8_t *input_2, int8_t *output, int32_t flat_size)
{
#if defined(ARM_MATH_MVEI)

    __ASM volatile(" .p2align 2                             \n"
                   "   wlstp.8         lr, %[cnt], 1f       \n"
                   "   vdup.8          q0, %[in1]           \n"
                   "2:                                      \n"
                   "   vldrb.8         q1, [%[in2]], #16    \n"
                   "   vmin.s8         q2, q0, q1           \n"
                   "   vstrb.8         q2, [%[out]], #16    \n"
                   "   letp            lr, 2b               \n"
                   "1:                                      \n"
                   : [in2] "+r"(input_2), [out] "+r"(output)
                   : [in1] "r"(*input_1), [cnt] "r"(flat_size)
                   : "q0", "q1", "q2", "memory", "r14");

#else
    int8_t in1 = *input_1;
    while (flat_size > 0)
    {
        int8_t in2 = *input_2++;
        *output++ = in1 >= in2 ? in2 : in1;
        --flat_size;
    }
#endif
    return ARM_CMSIS_NN_SUCCESS;
}

/*
 * s8 minimum
 *
 * Refer header file for details.
 *
 */
arm_cmsis_nn_status arm_minimum_s8(const cmsis_nn_context *ctx,
                                   const int8_t *input_1_data,
                                   const cmsis_nn_dims *input_1_dims,
                                   const int8_t *input_2_data,
                                   const cmsis_nn_dims *input_2_dims,
                                   int8_t *output_data,
                                   const cmsis_nn_dims *output_dims)
{
    (void)ctx;
    if (!input_1_data || !input_2_data || !output_data || !input_1_dims || !input_2_dims || !output_dims ||
        !arm_nn_broadcast_dims_valid(input_1_dims, input_2_dims, output_dims))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    ARM_NN_BROADCAST_WALK_NHWC(int8_t,
                               int8_t,
                               input_1_data,
                               input_1_dims,
                               input_2_data,
                               input_2_dims,
                               output_data,
                               output_dims,
                               arm_min_no_broadcast_s8,
                               arm_min_scalar_s8,
                               arm_min_scalar_s8);

    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of Doxygen group
 */

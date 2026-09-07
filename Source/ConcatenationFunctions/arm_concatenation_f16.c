/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
 * Title:        arm_concatenation_f16.c
 * Description:  Concatenation operators for float16 tensors
 *
 * $Date:        19 March 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#include "Internal/arm_concatenation_common.h"
#include "Internal/arm_nn_axis_copy_common.h"
#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

#if ARM_NN_ENABLE_F16

/**
 * @ingroup Public
 */

/**
 * @addtogroup NNSupport
 * @{
 */

ARM_CONCATENATION_DEFINE(f16, float16_t)

arm_cmsis_nn_status arm_concatenation_f16(const float16_t *const *input_data,
                                          const int32_t num_inputs,
                                          const int32_t *axis_sizes,
                                          const int32_t output_dims,
                                          const int32_t *output_shape,
                                          const int32_t axis,
                                          float16_t *output_data)
{
    int32_t outer;
    int32_t inner;
    if (output_dims < 1 || axis < 0 || axis >= output_dims || output_shape == NULL || axis_sizes == NULL ||
        arm_nn_axis_copy_plan(
            output_shape, output_dims, axis, axis + 1, output_shape[axis], num_inputs, axis_sizes, &outer, &inner) !=
            ARM_CMSIS_NN_SUCCESS)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    const int32_t total = outer * output_shape[axis] * inner;
    if ((output_data == NULL && total != 0) ||
        !arm_nn_axis_copy_ptrs_ok((const void *const *)input_data, num_inputs, total))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    arm_nn_axis_gather_f16(input_data, outer, num_inputs, axis_sizes, inner, output_data);
    return ARM_CMSIS_NN_SUCCESS;
}

/** @} */

#endif /* ARM_NN_ENABLE_F16 */

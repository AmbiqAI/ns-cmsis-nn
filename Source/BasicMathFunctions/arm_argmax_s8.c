/*
 * SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_argmax_s8
 * Description:  ArgMax reduction along a chosen axis for s8 tensors
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#include "arm_nnfunctions.h"

/**
 *  @ingroup Public
 */

/**
 * @addtogroup groupBasicMath
 * @{
 */

arm_cmsis_nn_status
arm_argmax_s8(const int8_t *input_data, const cmsis_nn_dims *input_dims, const int32_t axis, int32_t *output_data)
{
    if ((axis < 0) || (axis > 3))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    const int32_t dims[4] = {input_dims->n, input_dims->h, input_dims->w, input_dims->c};
    const int32_t axis_dim = dims[axis];

    if (axis_dim <= 0)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    int32_t inner_count = 1;
    for (int32_t i = axis + 1; i < 4; ++i)
    {
        inner_count *= dims[i];
    }

    int32_t outer_count = 1;
    for (int32_t i = 0; i < axis; ++i)
    {
        outer_count *= dims[i];
    }

    const int32_t block = axis_dim * inner_count;

    for (int32_t outer = 0; outer < outer_count; ++outer)
    {
        const int32_t outer_offset = outer * block;
        for (int32_t inner = 0; inner < inner_count; ++inner)
        {
            const int32_t base_index = outer_offset + inner;
            const int8_t *ptr = input_data + base_index;

            int32_t best_index = 0;
            int8_t best_value = ptr[0];

            for (int32_t k = 1; k < axis_dim; ++k)
            {
                const int8_t value = ptr[k * inner_count];
                if (value > best_value)
                {
                    best_value = value;
                    best_index = k;
                }
            }

            output_data[outer * inner_count + inner] = best_index;
        }
    }

    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of groupBasicMath
 */

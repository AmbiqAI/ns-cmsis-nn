/*
 * SPDX-FileCopyrightText: Copyright 2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
 * Title:        arm_svdf_get_buffer_sizes_s8.c
 * Description:  Collection of get buffer size functions for svdf s8 layer function.
 *
 * $Date:        5 September 2023
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "arm_nnfunctions.h"

/**
 *  @ingroup SVDF
 */

/**
 * @addtogroup GetBufferSizeSVDF
 * @{
 */

int32_t arm_svdf_s8_get_buffer_size_dsp(const cmsis_nn_dims *weights_feature_dims)
{
    (void)weights_feature_dims;
    return 0;
}

int32_t arm_svdf_s8_get_buffer_size_mve(const cmsis_nn_dims *weights_feature_dims)
{
    // Computed in 64 bits so that a row count large enough to overflow the int32_t byte count cannot wrap past
    // the range check below.
    const int64_t required_bytes = (int64_t)weights_feature_dims->n * (int64_t)sizeof(int32_t);

    if ((weights_feature_dims->n < 0) || (required_bytes > INT32_MAX))
    {
        return -1;
    }

    return (int32_t)required_bytes;
}

int32_t arm_svdf_s8_get_buffer_size(const cmsis_nn_dims *weights_feature_dims)
{
    // Validated once here, ahead of the dispatch below, so an invalid dim returns -1 on every build target -
    // not just the MVE leg, which is also called directly by binding glue and re-checks this on its own.
    const int64_t required_bytes = (int64_t)weights_feature_dims->n * (int64_t)sizeof(int32_t);

    if ((weights_feature_dims->n < 0) || (required_bytes > INT32_MAX))
    {
        return -1;
    }

#if defined(ARM_MATH_MVEI)
    return arm_svdf_s8_get_buffer_size_mve(weights_feature_dims);
#else
    return arm_svdf_s8_get_buffer_size_dsp(weights_feature_dims);
#endif
}

/**
 * @} end of GetBufferSizeSVDF group
 */

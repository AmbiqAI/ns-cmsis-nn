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
 * Title:        arm_fully_connected_get_buffer_sizes_s8.c
 * Description:  Collection of get buffer size functions for fully connected s8 layer function.
 *
 * $Date:        15 August 2023
 * $Revision:    V.1.1.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "arm_nnfunctions.h"

/**
 *  @ingroup FC
 */

/**
 * @addtogroup GetBufferSizeFC
 * @{
 */

int32_t arm_fully_connected_s8_get_buffer_size_dsp(const cmsis_nn_dims *filter_dims)
{
    // This leg needs no buffer, but it is a public entry point that the Python bindings call directly, so it has
    // to reject an out-of-range filter_dims->c with the same -1 the dispatcher and the MVE leg return rather than
    // a 0 the caller would read as "no buffer needed" (issue #349). The bound is the dispatcher's, not this leg's:
    // no buffer is sized here, so the check exists only to keep the three entry points answering alike.
    const int64_t required_bytes = (int64_t)filter_dims->c * (int64_t)sizeof(int32_t);

    if ((filter_dims->c < 0) || (required_bytes > INT32_MAX))
    {
        return -1;
    }

    return 0;
}

int32_t arm_fully_connected_s8_get_buffer_size_mve(const cmsis_nn_dims *filter_dims)
{
    // Computed in 64 bits so that a channel count large enough to overflow the int32_t byte count cannot wrap
    // past the range check below.
    const int64_t required_bytes = (int64_t)filter_dims->c * (int64_t)sizeof(int32_t);

    if ((filter_dims->c < 0) || (required_bytes > INT32_MAX))
    {
        return -1;
    }

    return (int32_t)required_bytes;
}

int32_t arm_fully_connected_s8_get_buffer_size(const cmsis_nn_dims *filter_dims)
{
    // Validated once here, ahead of the dispatch below, so an invalid dim returns -1 on every build target.
    // Both leg variants are also called directly by binding glue and re-check this on their own.
    const int64_t required_bytes = (int64_t)filter_dims->c * (int64_t)sizeof(int32_t);

    if ((filter_dims->c < 0) || (required_bytes > INT32_MAX))
    {
        return -1;
    }

#if defined(ARM_MATH_MVEI)
    return arm_fully_connected_s8_get_buffer_size_mve(filter_dims);
#else
    return arm_fully_connected_s8_get_buffer_size_dsp(filter_dims);
#endif
}

/**
 * @} end of GetBufferSizeFC group
 */

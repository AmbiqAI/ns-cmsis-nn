/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include "Internal/arm_nn_reduce_extrema_flt.h"

#if ARM_NN_ENABLE_F16

/** @ingroup Reduction */
arm_cmsis_nn_status arm_reduce_min_f16(const float16_t *input_data,
                                       const cmsis_nn_dims *input_dims,
                                       const cmsis_nn_dims *axis_dims,
                                       float16_t *output_data,
                                       const cmsis_nn_dims *output_dims)
{
    return arm_nn_extrema_run(input_data, input_dims, axis_dims, output_data, output_dims, 2, false);
}

#endif

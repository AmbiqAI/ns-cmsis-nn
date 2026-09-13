/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include "Internal/arm_nn_arg_extrema_flt.h"

#if ARM_NN_ENABLE_F16

/** @ingroup Reduction */
arm_cmsis_nn_status
arm_argmin_f16(const float16_t *input_data, const cmsis_nn_dims *input_dims, int32_t axis, int32_t *output_data)
{
    return arm_nn_arg_extrema(input_data, input_dims, axis, output_data, 2, false);
}

#endif

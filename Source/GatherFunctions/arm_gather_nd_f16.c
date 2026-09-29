/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include "Internal/arm_nn_gather_common.h"
#include "arm_nnfunctions.h"

#if ARM_NN_ENABLE_F16

/**
 * @ingroup Public
 * @addtogroup Gather
 * @{
 */

arm_cmsis_nn_status arm_gather_nd_f16(const float16_t *params_data,
                                      const cmsis_nn_dims *params_dims,
                                      const int32_t *indices_data,
                                      const cmsis_nn_dims *indices_dims,
                                      const cmsis_nn_gather_nd_params *params,
                                      float16_t *output_data,
                                      const cmsis_nn_dims *output_dims)
{
    arm_nn_gather_nd_plan plan;
    const arm_cmsis_nn_status status = arm_nn_gather_nd_prepare(params_data,
                                                                params_dims,
                                                                indices_data,
                                                                indices_dims,
                                                                params,
                                                                output_data,
                                                                output_dims,
                                                                sizeof(float16_t),
                                                                &plan);
    if (status != ARM_CMSIS_NN_SUCCESS || plan.output_count == 0)
    {
        return status;
    }
    arm_nn_gather_nd_copy_f16(params_data, indices_data, output_data, &plan);
    return ARM_CMSIS_NN_SUCCESS;
}

/** @} */
#endif /* ARM_NN_ENABLE_F16 */

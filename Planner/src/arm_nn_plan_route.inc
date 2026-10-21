/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

/*
 * The library's route selectors for one target ISA. The including file sets ARM_NN_ROUTE_MVEI and
 * ARM_NN_ROUTE_AUTOVECTORIZE and names the functions through ARM_NN_PLAN_ROUTE_FN, once per ISA, so a host build
 * routes for every target without compiling any target's intrinsics.
 */

#include "Internal/arm_nn_dispatch.h"
#include "arm_nn_plan_internal.h"

int32_t ARM_NN_PLAN_ROUTE_FN(dw_conv_s8)(const cmsis_nn_dw_conv_params *dw_conv_params,
                                         const cmsis_nn_dims *input_dims,
                                         const cmsis_nn_dims *filter_dims,
                                         const cmsis_nn_dims *output_dims)
{
    return (int32_t)arm_nn_depthwise_conv_wrapper_s8_select(dw_conv_params, input_dims, filter_dims, output_dims);
}

int32_t ARM_NN_PLAN_ROUTE_FN(fc_s8)(const cmsis_nn_quant_params *quant_params)
{
    return (int32_t)arm_nn_fully_connected_wrapper_s8_select(quant_params);
}

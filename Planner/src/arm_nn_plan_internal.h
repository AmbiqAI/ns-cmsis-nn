/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#ifndef ARM_NN_PLAN_INTERNAL_H
#define ARM_NN_PLAN_INTERNAL_H

#include "arm_nn_types.h"

#define ARM_NN_PLAN_ROUTE_DECLS(suffix)                                                                                \
    int32_t arm_nn_plan_route_dw_conv_s8_##suffix(const cmsis_nn_dw_conv_params *dw_conv_params,                       \
                                                  const cmsis_nn_dims *input_dims,                                     \
                                                  const cmsis_nn_dims *filter_dims,                                    \
                                                  const cmsis_nn_dims *output_dims);                                   \
    int32_t arm_nn_plan_route_fc_s8_##suffix(const cmsis_nn_quant_params *quant_params);

ARM_NN_PLAN_ROUTE_DECLS(scalar)
ARM_NN_PLAN_ROUTE_DECLS(mve)
ARM_NN_PLAN_ROUTE_DECLS(mve_autovec)

#endif

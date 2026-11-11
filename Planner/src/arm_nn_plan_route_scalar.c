/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

/* Routes of a library without MVE (scalar and DSP builds route alike). */
#define ARM_NN_ROUTE_MVEI 0
#define ARM_NN_ROUTE_AUTOVECTORIZE 0
#define ARM_NN_PLAN_ROUTE_FN(name) arm_nn_plan_route_##name##_scalar
#include "arm_nn_plan_route.inc"

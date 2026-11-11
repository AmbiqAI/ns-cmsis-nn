/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

/* Routes of an MVE library built with ARM_MATH_AUTOVECTORIZE. */
#define ARM_NN_ROUTE_MVEI 1
#define ARM_NN_ROUTE_AUTOVECTORIZE 1
#define ARM_NN_PLAN_ROUTE_FN(name) arm_nn_plan_route_##name##_mve_autovec
#include "arm_nn_plan_route.inc"

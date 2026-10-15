/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

/* arm_avg_pool_f32 (#630): average pooling cases in the shared pooling-window template. */

#define PW_PREFIX avg_pool_f32
#define PW_KERNEL arm_avg_pool_f32
#define PW_T float32_t
#define PW_PARAMS_T cmsis_nn_pool_params_f32
#define PW_ACT_MIN (-1000.0f)
#define PW_ACT_MAX 1000.0f
#define PW_AVG 1
#define PW_TOL 1.0e-5f
#define PW_CH 5

#include "../Utils/pool_window_cases.h"

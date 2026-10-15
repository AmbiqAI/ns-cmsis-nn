/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

/* arm_avg_pool_f16 (#630): average pooling cases in the shared pooling-window template. */

#define PW_PREFIX avg_pool_f16
#define PW_KERNEL arm_avg_pool_f16
#define PW_T float16_t
#define PW_PARAMS_T cmsis_nn_pool_params_f16
#define PW_ACT_MIN ((float16_t)(-1000.0f))
#define PW_ACT_MAX ((float16_t)1000.0f)
#define PW_AVG 1
#define PW_TOL 2.0e-2f
#define PW_CH 9

#include "../Utils/pool_window_cases.h"

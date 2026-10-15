/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

/* arm_max_pool_f16 (#630): max pooling cases in the shared pooling-window template. */

#define PW_PREFIX max_pool_f16
#define PW_KERNEL arm_max_pool_f16
#define PW_T float16_t
#define PW_PARAMS_T cmsis_nn_pool_params_f16
#define PW_ACT_MIN ((float16_t)(-1000.0f))
#define PW_ACT_MAX ((float16_t)1000.0f)
#define PW_AVG 0
#define PW_TOL 0.0f
#define PW_CH 9

#include "../Utils/pool_window_cases.h"

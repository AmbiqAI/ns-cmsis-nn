/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include "arm_nnfunctions.h"
#include "unity.h"
static const uint16_t resize_special_bits[] = {0x7e53, 0x8000, 1};
#define RESIZE_TYPE float16_t
#define RESIZE_ENTRY arm_resize_bilinear_f16
#define RESIZE_TOLERANCE 0.016f
#include "../Utils/resize_bilinear_flt_tests.h"

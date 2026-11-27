/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include "arm_nnfunctions.h"
#include "unity.h"
static const uint32_t resize_special_bits[] = {0x7fc12345, 0x80000000, 1};
#define RESIZE_TYPE float32_t
#define RESIZE_ENTRY arm_resize_bilinear_f32
#define RESIZE_TOLERANCE 1e-05f
#include "../Utils/resize_bilinear_flt_tests.h"

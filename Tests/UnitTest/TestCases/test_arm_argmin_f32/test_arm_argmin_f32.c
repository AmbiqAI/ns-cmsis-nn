/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */
#define AE_HALF 0
#define AE_MAX 0
#define AE_KERNEL arm_argmin_f32
#include "../Utils/arg_extrema_flt_test_common.h"

/* Extents whose 64-bit products wrap. Three int32 extents of 2^30, 2^30 and 16
 * multiply to 2^64, which a 64-bit product reads as 0, so a bound taken on the
 * wrapped product would admit the array as empty and return success without
 * writing. The array has 2^64 elements, beyond the INT32_MAX-byte bound, and is
 * refused like any other oversized array, with nothing written, as the count
 * checks of the base refused it. One shape per axis, each wrapping another
 * product: the extents after the axis (axis 0), the output count (axis 1) and
 * the extents before it (axes 2 and 3). */
static void ae_extent_wrap(void)
{
    const int32_t big = INT32_C(1) << 30;
    const cmsis_nn_dims shapes[4] = {{1, big, big, 16}, {big, 1, big, 16}, {big, big, 1, 16}, {big, big, 16, 1}};
    for (int32_t axis = 0; axis < 4; ++axis)
    {
        for (int i = 0; i < AE_CAP + 2 * AE_GUARD; ++i)
            ae_output[i] = AE_SENTINEL;
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, AE_KERNEL(ae_storage, &shapes[axis], axis, ae_output + AE_GUARD));
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, AE_KERNEL(NULL, &shapes[axis], axis, NULL));
        for (int i = 0; i < AE_CAP + 2 * AE_GUARD; ++i)
            TEST_ASSERT_EQUAL_INT32(AE_SENTINEL, ae_output[i]);
    }
}

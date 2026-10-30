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
    /* Here the output count, 2^62, does not wrap, but the input's element count
     * (2^64) and the output's byte count (2^64) both do, so a wrapped bound
     * would admit 2^62 writes. Refused as well. (A case that can only fail by
     * writing out of bounds, so it follows the fix.) */
    const cmsis_nn_dims writes = {big, big, 4, 4};
    for (int i = 0; i < AE_CAP + 2 * AE_GUARD; ++i)
        ae_output[i] = AE_SENTINEL;
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, AE_KERNEL(ae_storage, &writes, 3, ae_output + AE_GUARD));
    for (int i = 0; i < AE_CAP + 2 * AE_GUARD; ++i)
        TEST_ASSERT_EQUAL_INT32(AE_SENTINEL, ae_output[i]);
    /* A zero extent still makes an empty array whatever the others are, with
     * the zero last as well as first: success, nothing read or written. */
    const cmsis_nn_dims zero_last = {1, INT32_MAX, INT32_MAX, 0};
    const cmsis_nn_dims zero_first = {0, INT32_MAX, INT32_MAX, 1};
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, AE_KERNEL(NULL, &zero_last, 0, NULL));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, AE_KERNEL(NULL, &zero_first, 3, NULL));
}

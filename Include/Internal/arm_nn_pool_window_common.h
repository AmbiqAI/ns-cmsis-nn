/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_nn_pool_window_common.h
 * Description:  Shared pooling-window geometry check
 *
 * $Date:        1 October 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#ifndef ARM_NN_POOL_WINDOW_COMMON_H
#define ARM_NN_POOL_WINDOW_COMMON_H

#include "arm_nnsupportfunctions.h"

#include <stdbool.h>
#include <stdint.h>

/*
 * Whether every pooling window along one axis overlaps the input, and every bound the pooling loops form along it
 * fits in an int32_t. n is the output extent, s the stride, p the padding, k the filter extent and w the input
 * extent. Window i covers [b, b + k) with b = i * s - p, clipped to [0, w); it is empty exactly when k <= 0,
 * w <= 0, b >= w or b + k <= 0. b is linear in i, so each condition holds for some i exactly when it holds at
 * i = 0 or i = n - 1, and the same two ends bound b, -b, w - b and, more strictly than the loops need, b + k.
 * The loops that step b by s also step one stride past the last window, to b = n * s - p, which is bounded too.
 * Window positions are negated, so their lower bound is -INT32_MAX; the step past the last window is only ever an
 * increment result and is never negated, so INT32_MIN is in range for it. Expects n >= 1.
 */
__STATIC_FORCEINLINE bool
arm_nn_pool_axis_valid(const int32_t n, const int32_t s, const int32_t p, const int32_t k, const int32_t w)
{
    if ((k <= 0) || (w <= 0))
    {
        return false;
    }
    const int64_t b_first = -(int64_t)p;
    const int64_t b_last = (int64_t)(n - 1) * s - p;
    const int64_t lo = ARM_NN_MIN(b_first, b_last);
    const int64_t hi = ARM_NN_MAX(b_first, b_last);
    if ((hi >= w) || (lo + k <= 0))
    {
        return false;
    }
    const int64_t b_past = b_last + s;
    return (lo >= -(int64_t)INT32_MAX) && (hi + k <= INT32_MAX) && ((int64_t)w - lo <= INT32_MAX) &&
        (b_past >= INT32_MIN) && (b_past <= INT32_MAX);
}

#endif /* ARM_NN_POOL_WINDOW_COMMON_H */

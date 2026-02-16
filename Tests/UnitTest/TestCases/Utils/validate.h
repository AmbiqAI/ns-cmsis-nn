/*
 * SPDX-FileCopyrightText: Copyright 2010-2022, 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

static inline int validate(int8_t *act, const int8_t *ref, int size)
{
    int test_passed = true;
    int count = 0;
    int total = 0;

    for (int i = 0; i < size; ++i)
    {
        total++;
        if (act[i] != ref[i])
        {
            count++;
            printf("ERROR at pos %d: Act: %d Ref: %d\r\n", i, act[i], ref[i]);
            test_passed = false;
        }
    }

    if (!test_passed)
    {
        printf("%d of %d failed\r\n", count, total);
    }

    return test_passed;
}

static inline int validate_s16(int16_t *act, const int16_t *ref, int size)
{
    int test_passed = true;
    int count = 0;
    int total = 0;

    for (int i = 0; i < size; ++i)
    {
        total++;
        if (act[i] != ref[i])
        {
            count++;
            printf("ERROR at pos %d: Act: %d Ref: %d\r\n", i, act[i], ref[i]);
            test_passed = false;
        }
    }

    if (!test_passed)
    {
        printf("%d of %d failed\r\n", count, total);
    }

    return test_passed;
}

static inline int validate_s32(int32_t *act, const int32_t *ref, int size)
{
    int test_passed = true;
    int count = 0;
    int total = 0;

    for (int i = 0; i < size; ++i)
    {
        total++;
        if (act[i] != ref[i])
        {
            count++;
            printf("ERROR at pos %d: Act: %ld Ref: %ld\r\n", i, act[i], ref[i]);
            test_passed = false;
        }
    }

    if (!test_passed)
    {
        printf("%d of %d failed\r\n", count, total);
    }

    return test_passed;
}

static inline int validate_bool(const bool *act, const bool *ref, int size)
{
    int test_passed = true;
    int count = 0;

    for (int i = 0; i < size; ++i)
    {
        int actual = act[i] ? 1 : 0;
        int expected = ref[i] ? 1 : 0;

        if (actual != expected)
        {
            count++;
            printf("ERROR at pos %d: Act(bool): %d Ref(bool): %d\r\n", i, actual, expected);
            test_passed = false;
        }
    }

    if (!test_passed)
    {
        printf("%d of %d failed\r\n", count, size);
    }

    return test_passed;
}

/**
 * Validate two int8_t buffers with an absolute tolerance.
 *
 * A mismatch is reported when |act[i] - ref[i]| > tol.
 *
 * @param act   Actual/output buffer
 * @param ref   Reference/expected buffer
 * @param size  Number of elements
 * @param tol   Allowed absolute tolerance (e.g., 0 for exact match, 1 for ±1)
 * @return      1 (true) if all elements within tolerance, 0 (false) otherwise
 */
static inline int validate_tol_s8(const int8_t *act,
                                  const int8_t *ref,
                                  int size,
                                  int tol)
{
    if (tol < 0) tol = 0;  // guard
    int test_passed = 1;
    int count = 0;

    for (int i = 0; i < size; ++i)
    {
        int diff = (int)act[i] - (int)ref[i];
        if (diff < 0) diff = -diff;

        if (diff > tol)
        {
            ++count;
            printf("ERROR at pos %d: Act: %d Ref: %d (|diff|=%d > tol=%d)\r\n",
                   i, (int)act[i], (int)ref[i], diff, tol);
            test_passed = 0;
        }
    }

    if (!test_passed)
    {
        printf("%d of %d failed\r\n", count, size);
    }

    return test_passed;
}


/**
 * Validate two int16_t buffers with an absolute tolerance.
 *
 * A mismatch is reported when |act[i] - ref[i]| > tol.
 *
 * @param act   Actual/output buffer
 * @param ref   Reference/expected buffer
 * @param size  Number of elements
 * @param tol   Allowed absolute tolerance (e.g., 0 for exact match, 1 for ±1)
 * @return      1 (true) if all elements within tolerance, 0 (false) otherwise
 */
static inline int validate_tol_s16(const int16_t *act,
                                  const int16_t *ref,
                                  int size,
                                  int tol)
{
    if (tol < 0) tol = 0;  // guard
    int test_passed = 1;
    int count = 0;

    for (int i = 0; i < size; ++i)
    {
        int diff = (int)act[i] - (int)ref[i];
        if (diff < 0) diff = -diff;

        if (diff > tol)
        {
            ++count;
            printf("ERROR at pos %d: Act: %d Ref: %d (|diff|=%d > tol=%d)\r\n",
                   i, (int)act[i], (int)ref[i], diff, tol);
            test_passed = 0;
        }
    }

    if (!test_passed)
    {
        printf("%d of %d failed\r\n", count, size);
    }

    return test_passed;
}

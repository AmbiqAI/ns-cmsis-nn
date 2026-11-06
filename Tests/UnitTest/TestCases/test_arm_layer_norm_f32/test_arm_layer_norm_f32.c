/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include <math.h>
#include <stdint.h>
#include <string.h>

#include <arm_nnfunctions.h>
#include <unity.h>

#include "layer_norm_f32_data.h"

#define LAYER_NORM_F32_BUFFER (1024)
/* The scalar and MVE paths sum in different orders; both stay within 1e-6 of float64 layer norm on these cases. */
#define LAYER_NORM_F32_ATOL (1e-5f)
#define LAYER_NORM_F32_RTOL (1e-5f)

static float layer_norm_f32_out[LAYER_NORM_F32_BUFFER];

static void expect_close(const float *actual, const float *expected, int32_t size)
{
    for (int32_t i = 0; i < size; i++)
    {
        TEST_ASSERT_FLOAT_WITHIN(
            LAYER_NORM_F32_ATOL + LAYER_NORM_F32_RTOL * fabsf(expected[i]), expected[i], actual[i]);
    }
}

void layer_norm_f32_reference_cases(void)
{
    for (int32_t i = 0; i < LAYER_NORM_F32_NUM_CASES; i++)
    {
        const layer_norm_f32_case *c = &layer_norm_f32_cases[i];
        const int32_t size = c->rows * c->row_size;
        TEST_ASSERT_TRUE(size <= LAYER_NORM_F32_BUFFER);
        TEST_ASSERT_EQUAL(
            ARM_CMSIS_NN_SUCCESS,
            arm_layer_norm_f32(c->rows, c->row_size, c->epsilon, c->input, c->gamma, c->beta, layer_norm_f32_out));
        expect_close(layer_norm_f32_out, c->expected, size);
    }
}

void layer_norm_f32_in_place(void)
{
    for (int32_t i = 0; i < LAYER_NORM_F32_NUM_CASES; i++)
    {
        const layer_norm_f32_case *c = &layer_norm_f32_cases[i];
        const int32_t size = c->rows * c->row_size;
        memcpy(layer_norm_f32_out, c->input, (size_t)size * sizeof(float));
        TEST_ASSERT_EQUAL(
            ARM_CMSIS_NN_SUCCESS,
            arm_layer_norm_f32(
                c->rows, c->row_size, c->epsilon, layer_norm_f32_out, c->gamma, c->beta, layer_norm_f32_out));
        expect_close(layer_norm_f32_out, c->expected, size);
    }
}

/* Without gamma and beta every row comes out with mean 0 and variance 1 (eps is small next to the variance). */
void layer_norm_f32_identity_moments(void)
{
    const layer_norm_f32_case *c = &layer_norm_f32_cases[3];
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_layer_norm_f32(1, c->row_size, 0.0f, c->input, NULL, NULL, layer_norm_f32_out));
    float sum = 0.0f;
    float sq = 0.0f;
    for (int32_t k = 0; k < c->row_size; k++)
    {
        sum += layer_norm_f32_out[k];
        sq += layer_norm_f32_out[k] * layer_norm_f32_out[k];
    }
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, 0.0f, sum / (float)c->row_size);
    TEST_ASSERT_FLOAT_WITHIN(1e-4f, 1.0f, sq / (float)c->row_size);
}

void layer_norm_f32_invalid(void)
{
    const layer_norm_f32_case *c = &layer_norm_f32_cases[1];
    float *out = layer_norm_f32_out;
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_layer_norm_f32(1, c->row_size, c->epsilon, NULL, NULL, NULL, out));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_layer_norm_f32(1, c->row_size, c->epsilon, c->input, NULL, NULL, NULL));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_layer_norm_f32(-1, c->row_size, c->epsilon, c->input, NULL, NULL, out));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_layer_norm_f32(1, 0, c->epsilon, c->input, NULL, NULL, out));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_layer_norm_f32(1, c->row_size, -1.0f, c->input, NULL, NULL, out));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_layer_norm_f32(1, c->row_size, NAN, c->input, NULL, NULL, out));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_layer_norm_f32(1, c->row_size, INFINITY, c->input, NULL, NULL, out));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_layer_norm_f32(1, c->row_size, -0.0f, c->input, NULL, NULL, out));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_layer_norm_f32(0, c->row_size, c->epsilon, c->input, NULL, NULL, out));
}

/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include <stdint.h>
#include <string.h>

#include <Internal/arm_nn_layer_norm.h>
#include <arm_nnfunctions.h>
#include <unity.h>

#include "../Utils/validate.h"
#include "layer_norm_s8_fast_data.h"

#define LAYER_NORM_S8_FAST_BUFFER (1024)

static int8_t layer_norm_s8_fast_out[LAYER_NORM_S8_FAST_BUFFER];

static cmsis_nn_layer_norm_fast_params case_params(const layer_norm_s8_fast_case *c)
{
    const cmsis_nn_layer_norm_fast_params params = {.epsilon = c->epsilon,
                                                    .norm_frac_bits = c->norm_frac_bits,
                                                    .output_shift = c->output_shift,
                                                    .activation = {c->act_min, c->act_max}};
    return params;
}

static arm_cmsis_nn_status run_case(const layer_norm_s8_fast_case *c, const int8_t *input, int8_t *output)
{
    const cmsis_nn_layer_norm_fast_params params = case_params(c);
    return arm_layer_norm_s8_fast(&params, c->rows, c->row_size, input, c->gamma, c->bias, output);
}

/* Every generated case, against the reference's exact output (layer_norm_int_fast). */
void layer_norm_s8_fast_reference_cases(void)
{
    for (int32_t i = 0; i < LAYER_NORM_S8_FAST_NUM_CASES; i++)
    {
        const layer_norm_s8_fast_case *c = &layer_norm_s8_fast_cases[i];
        const int32_t size = c->rows * c->row_size;
        TEST_ASSERT_TRUE(size <= LAYER_NORM_S8_FAST_BUFFER);
        memset(layer_norm_s8_fast_out, 0x5A, sizeof(layer_norm_s8_fast_out));
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, run_case(c, c->input, layer_norm_s8_fast_out));
        TEST_ASSERT_TRUE(validate(layer_norm_s8_fast_out, c->expected, size));
        /* Nothing past the output is written. */
        if (size < LAYER_NORM_S8_FAST_BUFFER)
        {
            TEST_ASSERT_EQUAL((int8_t)0x5A5A, layer_norm_s8_fast_out[size]);
        }
    }
}

/* The output may alias the input. */
void layer_norm_s8_fast_in_place(void)
{
    for (int32_t i = 0; i < LAYER_NORM_S8_FAST_NUM_CASES; i++)
    {
        const layer_norm_s8_fast_case *c = &layer_norm_s8_fast_cases[i];
        const int32_t size = c->rows * c->row_size;
        memcpy(layer_norm_s8_fast_out, c->input, (size_t)size * sizeof(int8_t));
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, run_case(c, layer_norm_s8_fast_out, layer_norm_s8_fast_out));
        TEST_ASSERT_TRUE(validate(layer_norm_s8_fast_out, c->expected, size));
    }
}

/* A narrower activation range clamps the full-range result (the clamp is the last step). */
void layer_norm_s8_fast_activation(void)
{
    static int8_t expected[LAYER_NORM_S8_FAST_BUFFER];
    for (int32_t i = 0; i < LAYER_NORM_S8_FAST_NUM_CASES; i++)
    {
        const layer_norm_s8_fast_case *c = &layer_norm_s8_fast_cases[i];
        const int32_t size = c->rows * c->row_size;
        cmsis_nn_layer_norm_fast_params params = case_params(c);
        params.activation.min = -100;
        params.activation.max = 90;
        for (int32_t j = 0; j < size; j++)
        {
            expected[j] = (int8_t)ARM_NN_CLAMP(c->expected[j], 90, -100);
        }
        TEST_ASSERT_EQUAL(
            ARM_CMSIS_NN_SUCCESS,
            arm_layer_norm_s8_fast(&params, c->rows, c->row_size, c->input, c->gamma, c->bias, layer_norm_s8_fast_out));
        TEST_ASSERT_TRUE(validate(layer_norm_s8_fast_out, expected, size));
    }
}

void layer_norm_s8_fast_invalid(void)
{
    const layer_norm_s8_fast_case *c = &layer_norm_s8_fast_cases[2];
    cmsis_nn_layer_norm_fast_params params = case_params(c);
    const int32_t n = c->row_size;
    const int16_t *g = c->gamma;
    const int16_t *b = c->bias;
    const int8_t *x = c->input;
    int8_t *out = layer_norm_s8_fast_out;

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_layer_norm_s8_fast(NULL, 1, n, x, g, b, out));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_layer_norm_s8_fast(&params, 1, n, NULL, g, b, out));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_layer_norm_s8_fast(&params, 1, n, x, NULL, b, out));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_layer_norm_s8_fast(&params, 1, n, x, g, NULL, out));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_layer_norm_s8_fast(&params, 1, n, x, g, b, NULL));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_layer_norm_s8_fast(&params, -1, n, x, g, b, out));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_layer_norm_s8_fast(&params, 1, 0, x, g, b, out));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_layer_norm_s8_fast(&params, 1, ARM_NN_LAYER_NORM_MAX_ROW_SIZE_S8 + 1, x, g, b, out));
    params.epsilon = -1;
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_layer_norm_s8_fast(&params, 1, n, x, g, b, out));
    params.epsilon = ARM_NN_LAYER_NORM_MAX_EPSILON + 1;
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_layer_norm_s8_fast(&params, 1, n, x, g, b, out));
    params = case_params(c);
    params.norm_frac_bits = ARM_NN_LAYER_NORM_FAST_MIN_FRAC_BITS - 1;
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_layer_norm_s8_fast(&params, 1, n, x, g, b, out));
    params.norm_frac_bits = ARM_NN_LAYER_NORM_FAST_MAX_FRAC_BITS + 1;
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_layer_norm_s8_fast(&params, 1, n, x, g, b, out));
    params = case_params(c);
    params.output_shift = 0;
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_layer_norm_s8_fast(&params, 1, n, x, g, b, out));
    params.output_shift = 8;
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_layer_norm_s8_fast(&params, 1, n, x, g, b, out));
    params = case_params(c);
    params.activation.min = 1;
    params.activation.max = 0;
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_layer_norm_s8_fast(&params, 1, n, x, g, b, out));
    params.activation.min = -129;
    params.activation.max = 127;
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_layer_norm_s8_fast(&params, 1, n, x, g, b, out));
    params.activation.min = -128;
    params.activation.max = 128;
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_layer_norm_s8_fast(&params, 1, n, x, g, b, out));
    /* No rows is valid and writes nothing. */
    params = case_params(c);
    memset(out, 0x5A, sizeof(layer_norm_s8_fast_out));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_layer_norm_s8_fast(&params, 0, n, x, g, b, out));
    TEST_ASSERT_EQUAL((int8_t)0x5A5A, out[0]);
}

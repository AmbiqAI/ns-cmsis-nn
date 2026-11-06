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
#include "layer_norm_s8_data.h"

#define LAYER_NORM_S8_BUFFER (1024)

static int8_t layer_norm_s8_out[LAYER_NORM_S8_BUFFER];

static arm_cmsis_nn_status run_case(const layer_norm_s8_case *c, const int8_t *input, int8_t *output)
{
    const cmsis_nn_layer_norm_params params = {
        .epsilon = c->epsilon, .output_shift = c->output_shift, .activation = {c->act_min, c->act_max}};
    return arm_layer_norm_s8(&params, c->rows, c->row_size, input, c->gamma, c->bias, output);
}

/* Every generated case, against the reference's exact output for this build's rounding mode. */
void layer_norm_s8_reference_cases(void)
{
    for (int32_t i = 0; i < LAYER_NORM_S8_NUM_CASES; i++)
    {
        const layer_norm_s8_case *c = &layer_norm_s8_cases[i];
        const int32_t size = c->rows * c->row_size;
        TEST_ASSERT_TRUE(size <= LAYER_NORM_S8_BUFFER);
        memset(layer_norm_s8_out, 0x5A, sizeof(layer_norm_s8_out));
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, run_case(c, c->input, layer_norm_s8_out));
        TEST_ASSERT_TRUE(validate(layer_norm_s8_out, c->expected, size));
        /* Nothing past the output is written. */
        if (size < LAYER_NORM_S8_BUFFER)
        {
            TEST_ASSERT_EQUAL((int8_t)0x5A5A, layer_norm_s8_out[size]);
        }
    }
}

/* The output may alias the input. */
void layer_norm_s8_in_place(void)
{
    for (int32_t i = 0; i < LAYER_NORM_S8_NUM_CASES; i++)
    {
        const layer_norm_s8_case *c = &layer_norm_s8_cases[i];
        const int32_t size = c->rows * c->row_size;
        memcpy(layer_norm_s8_out, c->input, (size_t)size * sizeof(int8_t));
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, run_case(c, layer_norm_s8_out, layer_norm_s8_out));
        TEST_ASSERT_TRUE(validate(layer_norm_s8_out, c->expected, size));
    }
}

void layer_norm_s8_invalid(void)
{
    const layer_norm_s8_case *c = &layer_norm_s8_cases[1];
    cmsis_nn_layer_norm_params params = {
        .epsilon = c->epsilon, .output_shift = c->output_shift, .activation = {c->act_min, c->act_max}};
    const int32_t n = c->row_size;
    const int32_t *g = c->gamma;
    const int32_t *b = c->bias;
    int8_t *out = layer_norm_s8_out;

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_layer_norm_s8(NULL, 1, n, c->input, g, b, out));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_layer_norm_s8(&params, 1, n, NULL, g, b, out));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_layer_norm_s8(&params, 1, n, c->input, NULL, b, out));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_layer_norm_s8(&params, 1, n, c->input, g, NULL, out));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_layer_norm_s8(&params, 1, n, c->input, g, b, NULL));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_layer_norm_s8(&params, -1, n, c->input, g, b, out));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_layer_norm_s8(&params, 1, 0, c->input, g, b, out));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_layer_norm_s8(&params, 1, ARM_NN_LAYER_NORM_MAX_ROW_SIZE_S8 + 1, c->input, g, b, out));
    params.epsilon = -1;
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_layer_norm_s8(&params, 1, n, c->input, g, b, out));
    params.epsilon = ARM_NN_LAYER_NORM_MAX_EPSILON + 1;
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_layer_norm_s8(&params, 1, n, c->input, g, b, out));
    params.epsilon = c->epsilon;
    params.output_shift = 0;
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_layer_norm_s8(&params, 1, n, c->input, g, b, out));
    params.output_shift = 32;
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_layer_norm_s8(&params, 1, n, c->input, g, b, out));
    params.output_shift = c->output_shift;
    params.activation.min = 1;
    params.activation.max = 0;
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_layer_norm_s8(&params, 1, n, c->input, g, b, out));
    params.activation.min = c->act_min;
    params.activation.max = c->act_max;
    /* No rows is valid and writes nothing. */
    memset(out, 0x5A, sizeof(layer_norm_s8_out));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_layer_norm_s8(&params, 0, n, c->input, g, b, out));
    TEST_ASSERT_EQUAL((int8_t)0x5A5A, out[0]);
}

/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <arm_nnfunctions.h>
#include <unity.h>

static void assert_f16_array_close(const float16_t *expected, const float16_t *actual, int32_t length)
{
    for (int32_t index = 0; index < length; ++index)
    {
        TEST_ASSERT_FLOAT_WITHIN(2.0e-3f, (float)expected[index], (float)actual[index]);
    }
}

void prelu_f16_per_channel_broadcast(void)
{
    const cmsis_nn_dims input_dims = {1, 1, 3, 5};
    const cmsis_nn_dims alpha_dims = {1, 1, 1, 5};
    const cmsis_nn_dims output_dims = {1, 1, 3, 5};
    const float16_t input[] = {(float16_t)-2.0f,
                               (float16_t)-1.0f,
                               (float16_t)0.0f,
                               (float16_t)1.0f,
                               (float16_t)2.0f,
                               (float16_t)-4.0f,
                               (float16_t)-2.0f,
                               (float16_t)-0.5f,
                               (float16_t)0.5f,
                               (float16_t)-3.0f,
                               (float16_t)4.0f,
                               (float16_t)-8.0f,
                               (float16_t)1.0f,
                               (float16_t)-1.0f,
                               (float16_t)-2.0f};
    const float16_t alpha[] = {(float16_t)0.5f, (float16_t)0.25f, (float16_t)2.0f, (float16_t)3.0f, (float16_t)-1.0f};
    const float16_t expected[] = {(float16_t)-1.0f,
                                  (float16_t)-0.25f,
                                  (float16_t)0.0f,
                                  (float16_t)1.0f,
                                  (float16_t)2.0f,
                                  (float16_t)-2.0f,
                                  (float16_t)-0.5f,
                                  (float16_t)-1.0f,
                                  (float16_t)0.5f,
                                  (float16_t)3.0f,
                                  (float16_t)4.0f,
                                  (float16_t)-2.0f,
                                  (float16_t)1.0f,
                                  (float16_t)-3.0f,
                                  (float16_t)2.0f};
    float16_t output[15] = {0};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_prelu_f16(&input_dims, input, &alpha_dims, alpha, &output_dims, output));
    assert_f16_array_close(expected, output, 15);
}

void prelu_f16_scalar_alpha(void)
{
    const cmsis_nn_dims input_dims = {1, 1, 1, 4};
    const cmsis_nn_dims alpha_dims = {1, 1, 1, 1};
    const cmsis_nn_dims output_dims = {1, 1, 1, 4};
    const float16_t input[] = {(float16_t)-10.0f, (float16_t)-1.0f, (float16_t)0.0f, (float16_t)4.0f};
    const float16_t alpha[] = {(float16_t)0.1f};
    const float16_t expected[] = {(float16_t)-1.0f, (float16_t)-0.1f, (float16_t)0.0f, (float16_t)4.0f};
    float16_t output[4] = {0};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_prelu_f16(&input_dims, input, &alpha_dims, alpha, &output_dims, output));
    assert_f16_array_close(expected, output, 4);
}

void prelu_f16_rejects_invalid_broadcast(void)
{
    const cmsis_nn_dims input_dims = {1, 1, 3, 2};
    const cmsis_nn_dims alpha_dims = {1, 1, 2, 2};
    const cmsis_nn_dims output_dims = {1, 1, 3, 2};
    const float16_t input[6] = {0};
    const float16_t alpha[4] = {0};
    float16_t output[6] = {0};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_prelu_f16(&input_dims, input, &alpha_dims, alpha, &output_dims, output));
}

/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include "../../../../Include/Internal/arm_nn_fp_control.h"
#include <arm_nnfunctions.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unity.h>

#include "prelu_f32_data.h"

static const cmsis_nn_dims prelu_f32_input_dims = {1, 3, 4, 5};

static void prelu_f32_check(const cmsis_nn_dims *alpha_dims, const float32_t *alpha, const float32_t *expected)
{
    float32_t output[PRELU_F32_DST_SIZE] = {0};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_prelu_f32(&prelu_f32_input_dims,
                                    prelu_f32_input,
                                    alpha_dims,
                                    alpha,
                                    &prelu_f32_input_dims,
                                    output));

    for (int i = 0; i < PRELU_F32_DST_SIZE; ++i)
    {
        TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, expected[i], output[i]);
    }
}

void prelu_f32_alpha_per_channel_arm_prelu_f32(void)
{
    const cmsis_nn_dims alpha_dims = {1, 1, 1, 5};
    prelu_f32_check(&alpha_dims, prelu_f32_alpha_chan, prelu_f32_ref_chan);
}

void prelu_f32_alpha_full_arm_prelu_f32(void)
{
    const cmsis_nn_dims alpha_dims = {1, 3, 4, 5};
    prelu_f32_check(&alpha_dims, prelu_f32_alpha_full, prelu_f32_ref_full);
}

void prelu_f32_alpha_scalar_arm_prelu_f32(void)
{
    const cmsis_nn_dims alpha_dims = {1, 1, 1, 1};
    prelu_f32_check(&alpha_dims, prelu_f32_alpha_scal, prelu_f32_ref_scal);
}

void prelu_f32_alpha_broadcast_w_arm_prelu_f32(void)
{
    const cmsis_nn_dims alpha_dims = {1, 3, 1, 5};
    prelu_f32_check(&alpha_dims, prelu_f32_alpha_gen, prelu_f32_ref_gen);
}

void prelu_f32_alpha_broadcast_c_arm_prelu_f32(void)
{
    const cmsis_nn_dims alpha_dims = {1, 3, 4, 1};
    prelu_f32_check(&alpha_dims, prelu_f32_alpha_gen2, prelu_f32_ref_gen2);
}

void prelu_f32_arg_error_arm_prelu_f32(void)
{
    float32_t output[PRELU_F32_DST_SIZE] = {0};
    const cmsis_nn_dims alpha_dims = {1, 1, 1, 5};
    // Alpha dim that is neither 1 nor the input dim
    const cmsis_nn_dims bad_alpha_dims = {1, 1, 1, 3};
    // Output dims that differ from input dims
    const cmsis_nn_dims bad_output_dims = {1, 3, 4, 4};

    TEST_ASSERT_EQUAL(
        ARM_CMSIS_NN_ARG_ERROR,
        arm_prelu_f32(
            &prelu_f32_input_dims, NULL, &alpha_dims, prelu_f32_alpha_chan, &prelu_f32_input_dims, output));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_prelu_f32(&prelu_f32_input_dims,
                                    prelu_f32_input,
                                    &bad_alpha_dims,
                                    prelu_f32_alpha_chan,
                                    &prelu_f32_input_dims,
                                    output));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_prelu_f32(&prelu_f32_input_dims,
                                    prelu_f32_input,
                                    &alpha_dims,
                                    prelu_f32_alpha_chan,
                                    &bad_output_dims,
                                    output));
}

/*
 * Bit-exact edges (#873): every pairing of special inputs and alphas through each dispatch path, against
 * x >= 0 ? x : x * alpha with the product formed exactly in double and rounded once to float, so subnormal
 * inputs, alphas and products are exact. A NaN result is checked for NaN-ness only.
 */
static const uint32_t prelu_f32_edge_x[] = {0x00000000u, 0x80000000u, 0x00000001u, 0x80000001u, 0x80000008u,
                                            0x807FFFFFu, 0x00400000u, 0x00800000u, 0x80800000u, 0x3F800000u,
                                            0xBF800000u, 0x9E000000u, 0x7F7FFFFFu, 0xFF7FFFFFu, 0x7F800000u,
                                            0xFF800000u, 0x7FC00000u, 0xFFC00001u};
static const uint32_t prelu_f32_edge_alpha[] = {0x3E800000u, 0x00000000u, 0x80000000u, 0x00000001u,
                                                0x80000001u, 0x00400000u, 0x3F800000u, 0xBF000000u,
                                                0x0D800000u, 0x7F7FFFFFu, 0x7F800000u, 0x7FC00000u};
#define PRELU_F32_EDGE_NX (sizeof(prelu_f32_edge_x) / sizeof(prelu_f32_edge_x[0]))
#define PRELU_F32_EDGE_NA (sizeof(prelu_f32_edge_alpha) / sizeof(prelu_f32_edge_alpha[0]))
#define PRELU_F32_EDGE_N (PRELU_F32_EDGE_NX * PRELU_F32_EDGE_NA)

static float32_t prelu_f32_from_bits(uint32_t bits)
{
    float32_t f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

static uint32_t prelu_f32_to_bits(float32_t f)
{
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    return bits;
}

static void prelu_f32_edge_expect(uint32_t x, uint32_t a, uint32_t got)
{
    char message[48];
    snprintf(message, sizeof(message), "x 0x%08lx alpha 0x%08lx", (unsigned long)x, (unsigned long)a);
    const int x_nan = (x & 0x7FFFFFFFu) > 0x7F800000u;
    const int a_nan = (a & 0x7FFFFFFFu) > 0x7F800000u;
    const int non_negative = !(x & 0x80000000u) || (x & 0x7FFFFFFFu) == 0u;
    if (!x_nan && non_negative)
    {
        TEST_ASSERT_EQUAL_HEX32_MESSAGE(x, got, message);
        return;
    }
    const double product = (double)prelu_f32_from_bits(x) * (double)prelu_f32_from_bits(a);
    if (x_nan || a_nan || isnan(product))
    {
        TEST_ASSERT_TRUE_MESSAGE((got & 0x7FFFFFFFu) > 0x7F800000u, message);
        return;
    }
    TEST_ASSERT_EQUAL_HEX32_MESSAGE(prelu_f32_to_bits((float32_t)product), got, message);
}

void prelu_f32_edges_arm_prelu_f32(void)
{
    static float32_t input[PRELU_F32_EDGE_N];
    static float32_t alpha[PRELU_F32_EDGE_N];
    static float32_t output[PRELU_F32_EDGE_N];

    // Identical shapes: x major, alpha minor, 216 elements
    for (size_t i = 0; i < PRELU_F32_EDGE_NX; ++i)
    {
        for (size_t j = 0; j < PRELU_F32_EDGE_NA; ++j)
        {
            input[i * PRELU_F32_EDGE_NA + j] = prelu_f32_from_bits(prelu_f32_edge_x[i]);
            alpha[i * PRELU_F32_EDGE_NA + j] = prelu_f32_from_bits(prelu_f32_edge_alpha[j]);
        }
    }
    const cmsis_nn_dims full = {1, 1, 1, (int32_t)PRELU_F32_EDGE_N};
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_prelu_f32(&full, input, &full, alpha, &full, output));
    for (size_t k = 0; k < PRELU_F32_EDGE_N; ++k)
    {
        prelu_f32_edge_expect(prelu_f32_edge_x[k / PRELU_F32_EDGE_NA],
                              prelu_f32_edge_alpha[k % PRELU_F32_EDGE_NA],
                              prelu_f32_to_bits(output[k]));
    }

    // Per-channel alpha (1,1,1,C) and the general walk with a per-pixel alpha row (1,2,1,C) or value (1,2,1,1)
    for (size_t i = 0; i < PRELU_F32_EDGE_NX; ++i)
    {
        for (size_t j = 0; j < PRELU_F32_EDGE_NA; ++j)
        {
            input[j * PRELU_F32_EDGE_NX + i] = prelu_f32_from_bits(prelu_f32_edge_x[i]);
        }
    }
    for (size_t j = 0; j < PRELU_F32_EDGE_NA; ++j)
    {
        const cmsis_nn_dims in_dims = {1, 1, 1, (int32_t)PRELU_F32_EDGE_NX};
        const cmsis_nn_dims one = {1, 1, 1, 1};
        alpha[0] = prelu_f32_from_bits(prelu_f32_edge_alpha[j]);
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_prelu_f32(&in_dims, input, &one, alpha, &in_dims, output));
        for (size_t i = 0; i < PRELU_F32_EDGE_NX; ++i)
        {
            prelu_f32_edge_expect(prelu_f32_edge_x[i], prelu_f32_edge_alpha[j], prelu_f32_to_bits(output[i]));
        }
        for (size_t i = 0; i < PRELU_F32_EDGE_NX; ++i)
        {
            alpha[i] = alpha[0];
            alpha[PRELU_F32_EDGE_NX + i] = prelu_f32_from_bits(prelu_f32_edge_alpha[(j + 1) % PRELU_F32_EDGE_NA]);
        }
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_prelu_f32(&in_dims, input, &in_dims, alpha, &in_dims, output));
        for (size_t i = 0; i < PRELU_F32_EDGE_NX; ++i)
        {
            prelu_f32_edge_expect(prelu_f32_edge_x[i], prelu_f32_edge_alpha[j], prelu_f32_to_bits(output[i]));
        }
        const cmsis_nn_dims walk_in = {1, 2, 1, (int32_t)PRELU_F32_EDGE_NX};
        const cmsis_nn_dims walk_row = {1, 2, 1, (int32_t)PRELU_F32_EDGE_NX};
        const cmsis_nn_dims walk_value = {1, 2, 1, 1};
        const cmsis_nn_dims walk_chan = {1, 1, 1, (int32_t)PRELU_F32_EDGE_NX};
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_prelu_f32(&walk_in, input, &walk_row, alpha, &walk_in, output));
        for (size_t i = 0; i < 2 * PRELU_F32_EDGE_NX; ++i)
        {
            prelu_f32_edge_expect(prelu_f32_edge_x[i % PRELU_F32_EDGE_NX],
                                  prelu_f32_edge_alpha[(j + i / PRELU_F32_EDGE_NX) % PRELU_F32_EDGE_NA],
                                  prelu_f32_to_bits(output[i]));
        }
        float32_t pair[2] = {alpha[0], alpha[PRELU_F32_EDGE_NX]};
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_prelu_f32(&walk_in, input, &walk_value, pair, &walk_in, output));
        for (size_t i = 0; i < 2 * PRELU_F32_EDGE_NX; ++i)
        {
            prelu_f32_edge_expect(prelu_f32_edge_x[i % PRELU_F32_EDGE_NX],
                                  prelu_f32_edge_alpha[(j + i / PRELU_F32_EDGE_NX) % PRELU_F32_EDGE_NA],
                                  prelu_f32_to_bits(output[i]));
        }
        // (1,1,1,C) alpha against a 2-pixel input is the per-channel path
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_prelu_f32(&walk_in, input, &walk_chan, alpha, &walk_in, output));
        for (size_t i = 0; i < 2 * PRELU_F32_EDGE_NX; ++i)
        {
            prelu_f32_edge_expect(
                prelu_f32_edge_x[i % PRELU_F32_EDGE_NX], prelu_f32_edge_alpha[j], prelu_f32_to_bits(output[i]));
        }
    }
}

/* In place (output == input): a redone lane uses the x it loaded, not the value the vector store left (#873) */
void prelu_f32_in_place_arm_prelu_f32(void)
{
    static float32_t data[PRELU_F32_EDGE_N];
    static float32_t alpha[PRELU_F32_EDGE_N];
    for (size_t i = 0; i < PRELU_F32_EDGE_NX; ++i)
    {
        for (size_t j = 0; j < PRELU_F32_EDGE_NA; ++j)
        {
            data[i * PRELU_F32_EDGE_NA + j] = prelu_f32_from_bits(prelu_f32_edge_x[i]);
            alpha[i * PRELU_F32_EDGE_NA + j] = prelu_f32_from_bits(prelu_f32_edge_alpha[j]);
        }
    }
    const cmsis_nn_dims full = {1, 1, 1, (int32_t)PRELU_F32_EDGE_N};
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_prelu_f32(&full, data, &full, alpha, &full, data));
    for (size_t k = 0; k < PRELU_F32_EDGE_N; ++k)
    {
        prelu_f32_edge_expect(prelu_f32_edge_x[k / PRELU_F32_EDGE_NA],
                              prelu_f32_edge_alpha[k % PRELU_F32_EDGE_NA],
                              prelu_f32_to_bits(data[k]));
    }
    for (size_t j = 0; j < PRELU_F32_EDGE_NA; ++j)
    {
        const cmsis_nn_dims dims = {1, 1, 1, (int32_t)PRELU_F32_EDGE_NX};
        const cmsis_nn_dims one = {1, 1, 1, 1};
        const float32_t a = prelu_f32_from_bits(prelu_f32_edge_alpha[j]);
        for (size_t i = 0; i < PRELU_F32_EDGE_NX; ++i)
        {
            data[i] = prelu_f32_from_bits(prelu_f32_edge_x[i]);
        }
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_prelu_f32(&dims, data, &one, &a, &dims, data));
        for (size_t i = 0; i < PRELU_F32_EDGE_NX; ++i)
        {
            prelu_f32_edge_expect(prelu_f32_edge_x[i], prelu_f32_edge_alpha[j], prelu_f32_to_bits(data[i]));
        }
    }
}

/* With FPSCR.FZ set the scalar path's compare sees a negative subnormal as -0 and keeps x; the MVE path's redo makes
   the same choice (#873) */
void prelu_f32_flush_to_zero_arm_prelu_f32(void)
{
#if defined(ARM_NN_FP_CONTROL_AVAILABLE)
    static const uint32_t xs[] = {0x80000008u, 0x80000001u, 0x807FFFFFu, 0xBF800000u, 0x00000008u, 0x80000000u, 0x9E000000u};
    float32_t in[7], out[7];
    for (size_t i = 0; i < 7; ++i)
    {
        in[i] = prelu_f32_from_bits(xs[i]);
    }
    const float32_t alpha = 0.25f;
    const cmsis_nn_dims dims = {1, 1, 1, 7}, one = {1, 1, 1, 1};
    const arm_nn_fp_control_t saved = arm_nn_get_fp_control();
    arm_nn_set_fp_control(saved | (1u << 24));
    const arm_cmsis_nn_status status = arm_prelu_f32(&dims, in, &one, &alpha, &dims, out);
    arm_nn_set_fp_control(saved);
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, status);
    static const uint32_t expected[] = {0x80000008u, 0x80000001u, 0x807FFFFFu, 0xBE800000u, 0x00000008u, 0x80000000u,
                                        0x9D000000u};
    for (size_t i = 0; i < 7; ++i)
    {
        TEST_ASSERT_EQUAL_HEX32(expected[i], prelu_f32_to_bits(out[i]));
    }
#else
    TEST_IGNORE_MESSAGE("FP controls require Arm floating-point hardware");
#endif
}

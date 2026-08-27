/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include <arm_nnfunctions.h>
#include <unity.h>

// Every broadcast combination of the batch (n) and h dimensions, packed against a modulo-indexed
// reference, on M = 2, K = 3, N = 5 so the last packed block is partial. The RHS-h-broadcast case
// (lhs h = 2 against rhs h = 1) is the one that exercises the outer RHS stride.
static void bmm_f16_broadcast_case(int32_t lhs_n, int32_t lhs_h, int32_t rhs_n, int32_t rhs_h, int32_t packed)
{
    enum
    {
        M = 2,
        K = 3,
        N = 5,
        PACKED_MAT = K * (((N + 8 - 1) / 8) * 8)
    };
    const int32_t out_n = lhs_n > rhs_n ? lhs_n : rhs_n;
    const int32_t out_h = lhs_h > rhs_h ? lhs_h : rhs_h;
    const cmsis_nn_context ctx = {NULL, 0};
    const cmsis_nn_bmm_params_f16 bmm_params = {.adj_x = false,
                                                .adj_y = false,
                                                .activation = {(float16_t)-100.0f, (float16_t)100.0f},
                                                .rhs_format = packed ? ARM_NN_WEIGHT_FORMAT_NT_N_PACKED
                                                                     : ARM_NN_WEIGHT_FORMAT_STANDARD};
    const cmsis_nn_dims lhs_dims = {lhs_n, lhs_h, K, M};
    const cmsis_nn_dims rhs_dims = {rhs_n, rhs_h, K, N};
    const cmsis_nn_dims output_dims = {out_n, out_h, N, M};
    float16_t lhs[2 * 2 * M * K];
    float16_t rhs[2 * 2 * N * K];
    float16_t rhs_packed[2 * 2 * PACKED_MAT];
    float16_t output[2 * 2 * M * N];

    for (int32_t i = 0; i < lhs_n * lhs_h * M * K; ++i)
    {
        lhs[i] = (float16_t)((float32_t)((i * 7) % 11 - 5) / 4.0f);
    }
    for (int32_t i = 0; i < rhs_n * rhs_h * N * K; ++i)
    {
        rhs[i] = (float16_t)((float32_t)((i * 5) % 13 - 6) / 4.0f);
    }
    for (int32_t mat = 0; mat < rhs_n * rhs_h; ++mat)
    {
        float16_t *dst = rhs_packed + mat * PACKED_MAT;
        for (int32_t i = 0; i < PACKED_MAT; ++i)
        {
            dst[i] = (float16_t)0.0f;
        }
        for (int32_t n = 0; n < N; ++n)
        {
            for (int32_t k = 0; k < K; ++k)
            {
                dst[((n / 8) * K + k) * 8 + (n % 8)] = rhs[(mat * N + n) * K + k];
            }
        }
    }

    TEST_ASSERT_EQUAL(
        ARM_CMSIS_NN_SUCCESS,
        arm_batch_matmul_f16(
            &ctx, &bmm_params, &lhs_dims, lhs, &rhs_dims, packed ? rhs_packed : rhs, &output_dims, output));

    for (int32_t n = 0; n < out_n; ++n)
    {
        for (int32_t h = 0; h < out_h; ++h)
        {
            const int32_t lmat = (n % lhs_n) * lhs_h + (h % lhs_h);
            const int32_t rmat = (n % rhs_n) * rhs_h + (h % rhs_h);
            for (int32_t m = 0; m < M; ++m)
            {
                for (int32_t col = 0; col < N; ++col)
                {
                    float32_t acc = 0.0f;
                    for (int32_t k = 0; k < K; ++k)
                    {
                        acc += (float32_t)lhs[(lmat * M + m) * K + k] * (float32_t)rhs[(rmat * N + col) * K + k];
                    }
                    TEST_ASSERT_FLOAT_WITHIN(2.0e-2f, acc, (float32_t)output[((n * out_h + h) * M + m) * N + col]);
                }
            }
        }
    }
}

void batch_matmul_f16_packed_broadcast_shapes_arm_batch_matmul_f16(void)
{
    static const int32_t shapes[6][4] = {
        {2, 1, 2, 1}, {2, 2, 2, 1}, {2, 1, 1, 1}, {2, 2, 1, 1}, {2, 1, 2, 2}, {1, 1, 2, 1}};
    for (int32_t i = 0; i < 6; ++i)
    {
        bmm_f16_broadcast_case(shapes[i][0], shapes[i][1], shapes[i][2], shapes[i][3], 1);
        bmm_f16_broadcast_case(shapes[i][0], shapes[i][1], shapes[i][2], shapes[i][3], 0);
    }
}

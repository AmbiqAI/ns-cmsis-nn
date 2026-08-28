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

// One batch-matmul shape (M = 2, K = 3, rhs rows N = n) in either RHS layout, checked against a
// modulo-indexed reference. The broadcast sweep runs every batch (n) / h combination of the two
// operands; the RHS-h-broadcast case (lhs h = 2 against rhs h = 1) is the one that exercises the
// outer RHS stride. The packed buffer is zeroed in full so a kernel that strides by the wrong
// per-matrix footprint reads zeros rather than stack garbage.
static void
bmm_f16_broadcast_case(int32_t lhs_n, int32_t lhs_h, int32_t rhs_n, int32_t rhs_h, int32_t n, int32_t packed)
{
    enum
    {
        M = 2,
        K = 3,
        N_MAX = 9,
        PACKED_MAT_MAX = K * (((N_MAX + 8 - 1) / 8) * 8)
    };
    const int32_t packed_mat = K * (((n + 8 - 1) / 8) * 8);
    const int32_t out_n = lhs_n > rhs_n ? lhs_n : rhs_n;
    const int32_t out_h = lhs_h > rhs_h ? lhs_h : rhs_h;
    const cmsis_nn_context ctx = {NULL, 0};
    const cmsis_nn_bmm_params_f16 bmm_params = {.adj_x = false,
                                                .adj_y = false,
                                                .activation = {(float16_t)-100.0f, (float16_t)100.0f},
                                                .rhs_format = packed ? ARM_NN_WEIGHT_FORMAT_NT_N_PACKED
                                                                     : ARM_NN_WEIGHT_FORMAT_STANDARD};
    const cmsis_nn_dims lhs_dims = {lhs_n, lhs_h, K, M};
    const cmsis_nn_dims rhs_dims = {rhs_n, rhs_h, K, n};
    const cmsis_nn_dims output_dims = {out_n, out_h, n, M};
    float16_t lhs[2 * 2 * M * K];
    float16_t rhs[2 * 2 * N_MAX * K];
    float16_t rhs_packed[2 * 2 * PACKED_MAT_MAX];
    float16_t output[2 * 2 * M * N_MAX];

    TEST_ASSERT_TRUE(n <= N_MAX);
    for (int32_t i = 0; i < lhs_n * lhs_h * M * K; ++i)
    {
        lhs[i] = (float16_t)((float32_t)((i * 7) % 11 - 5) / 4.0f);
    }
    for (int32_t i = 0; i < rhs_n * rhs_h * n * K; ++i)
    {
        rhs[i] = (float16_t)((float32_t)((i * 5) % 13 - 6) / 4.0f);
    }
    for (int32_t i = 0; i < 2 * 2 * PACKED_MAT_MAX; ++i)
    {
        rhs_packed[i] = (float16_t)0.0f;
    }
    for (int32_t mat = 0; mat < rhs_n * rhs_h; ++mat)
    {
        float16_t *dst = rhs_packed + mat * packed_mat;
        for (int32_t row = 0; row < n; ++row)
        {
            for (int32_t k = 0; k < K; ++k)
            {
                dst[((row / 8) * K + k) * 8 + (row % 8)] = rhs[(mat * n + row) * K + k];
            }
        }
    }

    TEST_ASSERT_EQUAL(
        ARM_CMSIS_NN_SUCCESS,
        arm_batch_matmul_f16(
            &ctx, &bmm_params, &lhs_dims, lhs, &rhs_dims, packed ? rhs_packed : rhs, &output_dims, output));

    for (int32_t b = 0; b < out_n; ++b)
    {
        for (int32_t h = 0; h < out_h; ++h)
        {
            const int32_t lmat = (b % lhs_n) * lhs_h + (h % lhs_h);
            const int32_t rmat = (b % rhs_n) * rhs_h + (h % rhs_h);
            for (int32_t m = 0; m < M; ++m)
            {
                for (int32_t col = 0; col < n; ++col)
                {
                    float32_t acc = 0.0f;
                    for (int32_t k = 0; k < K; ++k)
                    {
                        acc += (float32_t)lhs[(lmat * M + m) * K + k] * (float32_t)rhs[(rmat * n + col) * K + k];
                    }
                    TEST_ASSERT_FLOAT_WITHIN(2.0e-2f, acc, (float32_t)output[((b * out_h + h) * M + m) * n + col]);
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
        bmm_f16_broadcast_case(shapes[i][0], shapes[i][1], shapes[i][2], shapes[i][3], 5, 1);
        bmm_f16_broadcast_case(shapes[i][0], shapes[i][1], shapes[i][2], shapes[i][3], 5, 0);
    }
}

// Pins the 8-wide packed block of this precision: at N = 3 and N = 9 the padded footprint
// (K * 8 and K * 16 for 8-wide blocks) differs from the 4-wide one (K * 4 and K * 12), so a
// kernel striding by the other precision's block width reads batch 1 from the wrong offset.
// N = 5 pads to K * 8 under both widths and cannot tell them apart.
void batch_matmul_f16_packed_block_width_batch2_arm_batch_matmul_f16(void)
{
    bmm_f16_broadcast_case(2, 1, 2, 1, 3, 1);
    bmm_f16_broadcast_case(2, 1, 2, 1, 9, 1);
}

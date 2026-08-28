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

// Regression shape for the packed-RHS batch stride: B=2, M=2, K=3, N=5.
// N=5 is not a multiple of the 4-wide packing block, so each packed RHS matrix
// occupies K * ceil(N / 4) * 4 = 3 * 8 = 24 elements instead of N * K = 15.
#define BMM_F32_BATCH (2)
#define BMM_F32_M (2)
#define BMM_F32_K (3)
#define BMM_F32_N (5)
#define BMM_F32_BLOCK (4)
#define BMM_F32_N_PADDED (((BMM_F32_N + BMM_F32_BLOCK - 1) / BMM_F32_BLOCK) * BMM_F32_BLOCK)
#define BMM_F32_RHS_PACKED_MAT_SIZE (BMM_F32_K * BMM_F32_N_PADDED)
#define BMM_F32_OUT_MAT_SIZE (BMM_F32_M * BMM_F32_N)

// LHS: [B][M][K], row-major.
static const float32_t bmm_f32_lhs[BMM_F32_BATCH][BMM_F32_M][BMM_F32_K] = {
    {{1.0f, 2.0f, 3.0f}, {-1.0f, 0.5f, 2.0f}},
    {{0.25f, -2.0f, 1.0f}, {3.0f, 1.0f, -0.5f}},
};

// RHS in the standard (adj_y == false) layout: [B][N][K], i.e. row n holds the K weights of output column n.
static const float32_t bmm_f32_rhs[BMM_F32_BATCH][BMM_F32_N][BMM_F32_K] = {
    {{1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f, 1.0f}, {2.0f, -1.0f, 0.5f}},
    {{-1.0f, 2.0f, 0.0f}, {0.5f, 0.5f, 0.5f}, {3.0f, 0.0f, -2.0f}, {0.0f, -1.0f, 1.0f}, {1.5f, 2.5f, -0.25f}},
};

// Packs one [N][K] matrix into the [ceil(N/4)][K][4] layout consumed by arm_nn_mat_mult_nt_n_packed_f32:
// packed[(block * K + k) * 4 + lane] = rhs[block * 4 + lane][k], tail lanes past N zero-filled.
static void bmm_f32_pack_rhs_matrix(const float32_t rhs[BMM_F32_N][BMM_F32_K], float32_t *packed)
{
    for (int32_t i = 0; i < BMM_F32_RHS_PACKED_MAT_SIZE; ++i)
    {
        packed[i] = 0.0f;
    }
    for (int32_t n = 0; n < BMM_F32_N; ++n)
    {
        const int32_t block = n / BMM_F32_BLOCK;
        const int32_t lane = n % BMM_F32_BLOCK;
        for (int32_t k = 0; k < BMM_F32_K; ++k)
        {
            packed[(block * BMM_F32_K + k) * BMM_F32_BLOCK + lane] = rhs[n][k];
        }
    }
}

// Plain-loop reference: out[b][m][n] = sum_k lhs[b][m][k] * rhs[b][n][k].
static void bmm_f32_reference(float32_t *out)
{
    for (int32_t b = 0; b < BMM_F32_BATCH; ++b)
    {
        for (int32_t m = 0; m < BMM_F32_M; ++m)
        {
            for (int32_t n = 0; n < BMM_F32_N; ++n)
            {
                float32_t acc = 0.0f;
                for (int32_t k = 0; k < BMM_F32_K; ++k)
                {
                    acc += bmm_f32_lhs[b][m][k] * bmm_f32_rhs[b][n][k];
                }
                out[b * BMM_F32_OUT_MAT_SIZE + m * BMM_F32_N + n] = acc;
            }
        }
    }
}

// Every batch after the first is read from the wrong offset if the kernel advances the packed RHS by
// N * K instead of its padded footprint, so batch 1 of this case is the regression check.
void batch_matmul_f32_packed_rhs_batch2_n5_arm_batch_matmul_f32(void)
{
    const cmsis_nn_context ctx = {NULL, 0};
    const cmsis_nn_bmm_params_f32 bmm_params = {.adj_x = false,
                                                .adj_y = false,
                                                .activation = {-100.0f, 100.0f},
                                                .rhs_format = ARM_NN_WEIGHT_FORMAT_NT_N_PACKED};
    // cmsis_nn_dims is {n, h, w, c}: the kernel reads M from lhs_dims.c, K from rhs_dims.w and N from rhs_dims.c.
    const cmsis_nn_dims lhs_dims = {BMM_F32_BATCH, 1, BMM_F32_K, BMM_F32_M};
    const cmsis_nn_dims rhs_dims = {BMM_F32_BATCH, 1, BMM_F32_K, BMM_F32_N};
    const cmsis_nn_dims output_dims = {BMM_F32_BATCH, 1, BMM_F32_N, BMM_F32_M};
    float32_t rhs_packed[BMM_F32_BATCH * BMM_F32_RHS_PACKED_MAT_SIZE];
    float32_t output[BMM_F32_BATCH * BMM_F32_OUT_MAT_SIZE] = {0};
    float32_t expected[BMM_F32_BATCH * BMM_F32_OUT_MAT_SIZE];

    for (int32_t b = 0; b < BMM_F32_BATCH; ++b)
    {
        bmm_f32_pack_rhs_matrix(bmm_f32_rhs[b], rhs_packed + b * BMM_F32_RHS_PACKED_MAT_SIZE);
    }
    bmm_f32_reference(expected);

    TEST_ASSERT_EQUAL(0, arm_batch_matmul_f32_get_buffer_size(&bmm_params, &lhs_dims, &rhs_dims, &output_dims));
    TEST_ASSERT_EQUAL(
        ARM_CMSIS_NN_SUCCESS,
        arm_batch_matmul_f32(
            &ctx, &bmm_params, &lhs_dims, &bmm_f32_lhs[0][0][0], &rhs_dims, rhs_packed, &output_dims, output));

    for (int32_t i = 0; i < BMM_F32_BATCH * BMM_F32_OUT_MAT_SIZE; ++i)
    {
        TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, expected[i], output[i]);
    }
}

// Control: same shapes and data through the standard (unpacked) RHS layout.
void batch_matmul_f32_standard_rhs_batch2_n5_arm_batch_matmul_f32(void)
{
    const cmsis_nn_context ctx = {NULL, 0};
    const cmsis_nn_bmm_params_f32 bmm_params = {
        .adj_x = false, .adj_y = false, .activation = {-100.0f, 100.0f}, .rhs_format = ARM_NN_WEIGHT_FORMAT_STANDARD};
    const cmsis_nn_dims lhs_dims = {BMM_F32_BATCH, 1, BMM_F32_K, BMM_F32_M};
    const cmsis_nn_dims rhs_dims = {BMM_F32_BATCH, 1, BMM_F32_K, BMM_F32_N};
    const cmsis_nn_dims output_dims = {BMM_F32_BATCH, 1, BMM_F32_N, BMM_F32_M};
    float32_t output[BMM_F32_BATCH * BMM_F32_OUT_MAT_SIZE] = {0};
    float32_t expected[BMM_F32_BATCH * BMM_F32_OUT_MAT_SIZE];

    bmm_f32_reference(expected);

    TEST_ASSERT_EQUAL(0, arm_batch_matmul_f32_get_buffer_size(&bmm_params, &lhs_dims, &rhs_dims, &output_dims));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_batch_matmul_f32(&ctx,
                                           &bmm_params,
                                           &lhs_dims,
                                           &bmm_f32_lhs[0][0][0],
                                           &rhs_dims,
                                           &bmm_f32_rhs[0][0][0],
                                           &output_dims,
                                           output));

    for (int32_t i = 0; i < BMM_F32_BATCH * BMM_F32_OUT_MAT_SIZE; ++i)
    {
        TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, expected[i], output[i]);
    }
}

// One batch-matmul shape (M = 2, K = 3, rhs rows N = n) in either RHS layout, checked against a
// modulo-indexed reference. The broadcast sweep runs every batch (n) / h combination of the two
// operands; the RHS-h-broadcast case (lhs h = 2 against rhs h = 1) is the one that exercises the
// outer RHS stride. The packed buffer is zeroed in full so a kernel that strides by the wrong
// per-matrix footprint reads zeros rather than stack garbage.
static void
bmm_f32_broadcast_case(int32_t lhs_n, int32_t lhs_h, int32_t rhs_n, int32_t rhs_h, int32_t n, int32_t packed)
{
    enum
    {
        M = 2,
        K = 3,
        N_MAX = 9,
        PACKED_MAT_MAX = K * (((N_MAX + 4 - 1) / 4) * 4)
    };
    const int32_t packed_mat = K * (((n + 4 - 1) / 4) * 4);
    const int32_t out_n = lhs_n > rhs_n ? lhs_n : rhs_n;
    const int32_t out_h = lhs_h > rhs_h ? lhs_h : rhs_h;
    const cmsis_nn_context ctx = {NULL, 0};
    const cmsis_nn_bmm_params_f32 bmm_params = {.adj_x = false,
                                                .adj_y = false,
                                                .activation = {(float32_t)-100.0f, (float32_t)100.0f},
                                                .rhs_format = packed ? ARM_NN_WEIGHT_FORMAT_NT_N_PACKED
                                                                     : ARM_NN_WEIGHT_FORMAT_STANDARD};
    const cmsis_nn_dims lhs_dims = {lhs_n, lhs_h, K, M};
    const cmsis_nn_dims rhs_dims = {rhs_n, rhs_h, K, n};
    const cmsis_nn_dims output_dims = {out_n, out_h, n, M};
    float32_t lhs[2 * 2 * M * K];
    float32_t rhs[2 * 2 * N_MAX * K];
    float32_t rhs_packed[2 * 2 * PACKED_MAT_MAX];
    float32_t output[2 * 2 * M * N_MAX];

    TEST_ASSERT_TRUE(n <= N_MAX);
    for (int32_t i = 0; i < lhs_n * lhs_h * M * K; ++i)
    {
        lhs[i] = (float32_t)((float32_t)((i * 7) % 11 - 5) / 4.0f);
    }
    for (int32_t i = 0; i < rhs_n * rhs_h * n * K; ++i)
    {
        rhs[i] = (float32_t)((float32_t)((i * 5) % 13 - 6) / 4.0f);
    }
    for (int32_t i = 0; i < 2 * 2 * PACKED_MAT_MAX; ++i)
    {
        rhs_packed[i] = (float32_t)0.0f;
    }
    for (int32_t mat = 0; mat < rhs_n * rhs_h; ++mat)
    {
        float32_t *dst = rhs_packed + mat * packed_mat;
        for (int32_t row = 0; row < n; ++row)
        {
            for (int32_t k = 0; k < K; ++k)
            {
                dst[((row / 4) * K + k) * 4 + (row % 4)] = rhs[(mat * n + row) * K + k];
            }
        }
    }

    TEST_ASSERT_EQUAL(
        ARM_CMSIS_NN_SUCCESS,
        arm_batch_matmul_f32(
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
                    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, acc, (float32_t)output[((b * out_h + h) * M + m) * n + col]);
                }
            }
        }
    }
}

void batch_matmul_f32_packed_broadcast_shapes_arm_batch_matmul_f32(void)
{
    static const int32_t shapes[6][4] = {
        {2, 1, 2, 1}, {2, 2, 2, 1}, {2, 1, 1, 1}, {2, 2, 1, 1}, {2, 1, 2, 2}, {1, 1, 2, 1}};
    for (int32_t i = 0; i < 6; ++i)
    {
        bmm_f32_broadcast_case(shapes[i][0], shapes[i][1], shapes[i][2], shapes[i][3], 5, 1);
        bmm_f32_broadcast_case(shapes[i][0], shapes[i][1], shapes[i][2], shapes[i][3], 5, 0);
    }
}

// Pins the 4-wide packed block of this precision: at N = 3 and N = 9 the padded footprint
// (K * 4 and K * 12 for 4-wide blocks) differs from the 8-wide one (K * 8 and K * 16), so a
// kernel striding by the other precision's block width reads batch 1 from the wrong offset.
// N = 5 pads to K * 8 under both widths and cannot tell them apart.
void batch_matmul_f32_packed_block_width_batch2_arm_batch_matmul_f32(void)
{
    bmm_f32_broadcast_case(2, 1, 2, 1, 3, 1);
    bmm_f32_broadcast_case(2, 1, 2, 1, 9, 1);
}

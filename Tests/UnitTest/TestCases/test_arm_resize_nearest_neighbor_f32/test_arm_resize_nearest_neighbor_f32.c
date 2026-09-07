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

#include "../Utils/resize_nearest_neighbor_flt_test_common.h"

#define RESIZE_NN_GUARD 8

static float32_t resize_nn_f32_input[RESIZE_NN_FLT_MAX_INPUT];
static float32_t resize_nn_f32_output[RESIZE_NN_FLT_MAX_OUTPUT + RESIZE_NN_GUARD];
static float32_t resize_nn_f32_expected[RESIZE_NN_FLT_MAX_OUTPUT];
static int32_t resize_nn_f32_scratch[RESIZE_NN_FLT_MAX_SCRATCH_ELEMENTS + RESIZE_NN_GUARD];

static void resize_nn_f32_run_case(const resize_nn_flt_case *c, uint32_t seed)
{
    const cmsis_nn_resize_params params = {.align_corners = c->align_corners,
                                           .half_pixel_centers = c->half_pixel_centers};
    const cmsis_nn_dims input_dims = {c->n, c->h, c->w, c->c};
    const cmsis_nn_dims output_size_dims = {1, 1, 1, 2};
    const int32_t output_size[2] = {c->out_h, c->out_w};
    const cmsis_nn_dims output_dims = {c->n, c->out_h, c->out_w, c->c};
    const size_t in_elements = (size_t)c->n * c->h * c->w * c->c;
    const size_t out_elements = (size_t)c->n * c->out_h * c->out_w * c->c;

    /* The in-test formula must reproduce the TFLite-observed maps. */
    for (int32_t y = 0; y < c->out_h; ++y)
    {
        TEST_ASSERT_EQUAL_INT32_MESSAGE(
            c->y_map[y], resize_nn_ref_index(y, c->h, c->out_h, c->align_corners, c->half_pixel_centers), c->name);
    }
    for (int32_t x = 0; x < c->out_w; ++x)
    {
        TEST_ASSERT_EQUAL_INT32_MESSAGE(
            c->x_map[x], resize_nn_ref_index(x, c->w, c->out_w, c->align_corners, c->half_pixel_centers), c->name);
    }

    const int32_t scratch_bytes = arm_resize_nearest_neighbor_f32_get_buffer_size(&output_dims);
    TEST_ASSERT_EQUAL_INT32_MESSAGE((c->out_h + c->out_w) * (int32_t)sizeof(int32_t), scratch_bytes, c->name);
    const cmsis_nn_context ctx = {.buf = resize_nn_f32_scratch, .size = scratch_bytes};

    resize_nn_fill_bits(resize_nn_f32_input, in_elements, sizeof(float32_t), seed);
    resize_nn_gather(resize_nn_f32_expected, resize_nn_f32_input, sizeof(float32_t), c);
    for (size_t i = 0; i < RESIZE_NN_FLT_MAX_OUTPUT + RESIZE_NN_GUARD; ++i)
    {
        const uint32_t canary = RESIZE_NN_CANARY_WORD;
        memcpy(&resize_nn_f32_output[i], &canary, sizeof(canary));
    }
    for (size_t i = 0; i < RESIZE_NN_FLT_MAX_SCRATCH_ELEMENTS + RESIZE_NN_GUARD; ++i)
    {
        resize_nn_f32_scratch[i] = (int32_t)RESIZE_NN_CANARY_WORD;
    }

    TEST_ASSERT_EQUAL_MESSAGE(ARM_CMSIS_NN_SUCCESS,
                              arm_resize_nearest_neighbor_f32(&ctx,
                                                              &params,
                                                              &input_dims,
                                                              resize_nn_f32_input,
                                                              &output_size_dims,
                                                              output_size,
                                                              &output_dims,
                                                              resize_nn_f32_output),
                              c->name);

    TEST_ASSERT_EQUAL_MEMORY_MESSAGE(
        resize_nn_f32_expected, resize_nn_f32_output, out_elements * sizeof(float32_t), c->name);
    for (size_t i = out_elements; i < RESIZE_NN_FLT_MAX_OUTPUT + RESIZE_NN_GUARD; ++i)
    {
        uint32_t bits;
        memcpy(&bits, &resize_nn_f32_output[i], sizeof(bits));
        TEST_ASSERT_EQUAL_HEX32_MESSAGE(RESIZE_NN_CANARY_WORD, bits, c->name);
    }
    for (size_t i = (size_t)(c->out_h + c->out_w); i < RESIZE_NN_FLT_MAX_SCRATCH_ELEMENTS + RESIZE_NN_GUARD; ++i)
    {
        TEST_ASSERT_EQUAL_HEX32_MESSAGE(RESIZE_NN_CANARY_WORD, (uint32_t)resize_nn_f32_scratch[i], c->name);
    }
}

void resize_nearest_neighbor_f32_tflite_cases(void)
{
    for (int32_t i = 0; i < RESIZE_NN_FLT_CASE_COUNT; ++i)
    {
        resize_nn_f32_run_case(&resize_nn_flt_cases[i], 0x9E3779B9U + (uint32_t)i);
    }
}

/* Every special pattern placed in every channel of the TFLite-verified half_pixel_upscale_c9 case (1x4x4x9 -> 7x7),
 * so each pattern is copied to several outputs through both the whole-vector and the tail lanes. */
void resize_nearest_neighbor_f32_special_values(void)
{
    const resize_nn_flt_case *c = NULL;
    for (int32_t i = 0; i < RESIZE_NN_FLT_CASE_COUNT; ++i)
    {
        if (strcmp(resize_nn_flt_cases[i].name, "half_pixel_upscale_c9") == 0)
        {
            c = &resize_nn_flt_cases[i];
        }
    }
    TEST_ASSERT_NOT_NULL(c);
    const cmsis_nn_resize_params params = {.align_corners = c->align_corners,
                                           .half_pixel_centers = c->half_pixel_centers};
    const cmsis_nn_dims input_dims = {c->n, c->h, c->w, c->c};
    const cmsis_nn_dims output_size_dims = {1, 1, 1, 2};
    const int32_t output_size[2] = {c->out_h, c->out_w};
    const cmsis_nn_dims output_dims = {c->n, c->out_h, c->out_w, c->c};
    const cmsis_nn_context ctx = {.buf = resize_nn_f32_scratch,
                                  .size = arm_resize_nearest_neighbor_f32_get_buffer_size(&output_dims)};
    const size_t out_elements = (size_t)c->n * c->out_h * c->out_w * c->c;

    memset(resize_nn_f32_input, 0, sizeof(resize_nn_f32_input));
    for (int32_t pixel = 0; pixel < c->n * c->h * c->w; ++pixel)
    {
        for (int32_t ch = 0; ch < c->c; ++ch)
        {
            memcpy(&resize_nn_f32_input[pixel * c->c + ch],
                   &resize_nn_special_bits_f32[(ch + pixel) % 8],
                   sizeof(float32_t));
        }
    }
    resize_nn_gather(resize_nn_f32_expected, resize_nn_f32_input, sizeof(float32_t), c);
    memset(resize_nn_f32_output, 0, sizeof(resize_nn_f32_output));

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_resize_nearest_neighbor_f32(&ctx,
                                                      &params,
                                                      &input_dims,
                                                      resize_nn_f32_input,
                                                      &output_size_dims,
                                                      output_size,
                                                      &output_dims,
                                                      resize_nn_f32_output));
    TEST_ASSERT_EQUAL_MEMORY(resize_nn_f32_expected, resize_nn_f32_output, out_elements * sizeof(float32_t));
}

void resize_nearest_neighbor_f32_rejects_bad_args(void)
{
    const cmsis_nn_resize_params params = {.align_corners = false, .half_pixel_centers = false};
    const cmsis_nn_dims input_dims = {1, 2, 3, 2};
    const cmsis_nn_dims output_size_dims = {1, 1, 1, 2};
    const int32_t output_size[2] = {2, 6};
    const cmsis_nn_dims output_dims = {1, 2, 6, 2};
    cmsis_nn_context ctx = {.buf = resize_nn_f32_scratch, .size = 8 * (int32_t)sizeof(int32_t)};
    const uint32_t canary = RESIZE_NN_CANARY_WORD;

    memset(resize_nn_f32_input, 0, sizeof(resize_nn_f32_input));
    for (size_t i = 0; i < RESIZE_NN_FLT_MAX_OUTPUT + RESIZE_NN_GUARD; ++i)
    {
        memcpy(&resize_nn_f32_output[i], &canary, sizeof(canary));
    }
    for (size_t i = 0; i < RESIZE_NN_FLT_MAX_SCRATCH_ELEMENTS + RESIZE_NN_GUARD; ++i)
    {
        resize_nn_f32_scratch[i] = (int32_t)RESIZE_NN_CANARY_WORD;
    }

#define RESIZE_NN_CALL(CTX, PARAMS, IN_DIMS, IN, SIZE_DIMS, SIZE, OUT_DIMS, OUT)                                       \
    arm_resize_nearest_neighbor_f32((CTX), (PARAMS), (IN_DIMS), (IN), (SIZE_DIMS), (SIZE), (OUT_DIMS), (OUT))
#define RESIZE_NN_EXPECT_ARG_ERROR(...) TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, RESIZE_NN_CALL(__VA_ARGS__))

    /* Baseline call succeeds, so each rejection below is attributable to its one change. */
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      RESIZE_NN_CALL(&ctx,
                                     &params,
                                     &input_dims,
                                     resize_nn_f32_input,
                                     &output_size_dims,
                                     output_size,
                                     &output_dims,
                                     resize_nn_f32_output));

    /* Re-canary the scratch: a rejected call must not write even where the baseline legitimately did. */
    for (size_t i = 0; i < RESIZE_NN_FLT_MAX_SCRATCH_ELEMENTS + RESIZE_NN_GUARD; ++i)
    {
        resize_nn_f32_scratch[i] = (int32_t)RESIZE_NN_CANARY_WORD;
    }

    RESIZE_NN_EXPECT_ARG_ERROR(NULL,
                               &params,
                               &input_dims,
                               resize_nn_f32_input,
                               &output_size_dims,
                               output_size,
                               &output_dims,
                               resize_nn_f32_output);
    RESIZE_NN_EXPECT_ARG_ERROR(&ctx,
                               NULL,
                               &input_dims,
                               resize_nn_f32_input,
                               &output_size_dims,
                               output_size,
                               &output_dims,
                               resize_nn_f32_output);
    RESIZE_NN_EXPECT_ARG_ERROR(
        &ctx, &params, NULL, resize_nn_f32_input, &output_size_dims, output_size, &output_dims, resize_nn_f32_output);
    RESIZE_NN_EXPECT_ARG_ERROR(
        &ctx, &params, &input_dims, NULL, &output_size_dims, output_size, &output_dims, resize_nn_f32_output);
    RESIZE_NN_EXPECT_ARG_ERROR(
        &ctx, &params, &input_dims, resize_nn_f32_input, NULL, output_size, &output_dims, resize_nn_f32_output);
    RESIZE_NN_EXPECT_ARG_ERROR(
        &ctx, &params, &input_dims, resize_nn_f32_input, &output_size_dims, NULL, &output_dims, resize_nn_f32_output);
    RESIZE_NN_EXPECT_ARG_ERROR(
        &ctx, &params, &input_dims, resize_nn_f32_input, &output_size_dims, output_size, NULL, resize_nn_f32_output);
    RESIZE_NN_EXPECT_ARG_ERROR(
        &ctx, &params, &input_dims, resize_nn_f32_input, &output_size_dims, output_size, &output_dims, NULL);

    /* Scratch: NULL, one byte short, and misaligned. */
    cmsis_nn_context bad_ctx = {.buf = NULL, .size = ctx.size};
    RESIZE_NN_EXPECT_ARG_ERROR(&bad_ctx,
                               &params,
                               &input_dims,
                               resize_nn_f32_input,
                               &output_size_dims,
                               output_size,
                               &output_dims,
                               resize_nn_f32_output);
    bad_ctx.buf = resize_nn_f32_scratch;
    bad_ctx.size = ctx.size - 1;
    RESIZE_NN_EXPECT_ARG_ERROR(&bad_ctx,
                               &params,
                               &input_dims,
                               resize_nn_f32_input,
                               &output_size_dims,
                               output_size,
                               &output_dims,
                               resize_nn_f32_output);
    bad_ctx.buf = (uint8_t *)resize_nn_f32_scratch + 2;
    bad_ctx.size = ctx.size;
    RESIZE_NN_EXPECT_ARG_ERROR(&bad_ctx,
                               &params,
                               &input_dims,
                               resize_nn_f32_input,
                               &output_size_dims,
                               output_size,
                               &output_dims,
                               resize_nn_f32_output);

    /* Size tensor not exactly two elements; zero and negative output size. */
    const cmsis_nn_dims three_dims = {1, 1, 1, 3};
    RESIZE_NN_EXPECT_ARG_ERROR(
        &ctx, &params, &input_dims, resize_nn_f32_input, &three_dims, output_size, &output_dims, resize_nn_f32_output);
    const int32_t zero_h[2] = {0, 6};
    const cmsis_nn_dims zero_h_dims = {1, 0, 6, 2};
    RESIZE_NN_EXPECT_ARG_ERROR(
        &ctx, &params, &input_dims, resize_nn_f32_input, &output_size_dims, zero_h, &zero_h_dims, resize_nn_f32_output);
    const int32_t neg_w[2] = {2, -6};
    const cmsis_nn_dims neg_w_dims = {1, 2, -6, 2};
    RESIZE_NN_EXPECT_ARG_ERROR(
        &ctx, &params, &input_dims, resize_nn_f32_input, &output_size_dims, neg_w, &neg_w_dims, resize_nn_f32_output);

    /* output_shape disagreeing with the size tensor or the input's n / c; a zero input dimension. */
    const cmsis_nn_dims wrong_h = {1, 3, 6, 2};
    RESIZE_NN_EXPECT_ARG_ERROR(&ctx,
                               &params,
                               &input_dims,
                               resize_nn_f32_input,
                               &output_size_dims,
                               output_size,
                               &wrong_h,
                               resize_nn_f32_output);
    const cmsis_nn_dims wrong_w = {1, 2, 5, 2};
    RESIZE_NN_EXPECT_ARG_ERROR(&ctx,
                               &params,
                               &input_dims,
                               resize_nn_f32_input,
                               &output_size_dims,
                               output_size,
                               &wrong_w,
                               resize_nn_f32_output);
    const cmsis_nn_dims wrong_n = {2, 2, 6, 2};
    RESIZE_NN_EXPECT_ARG_ERROR(&ctx,
                               &params,
                               &input_dims,
                               resize_nn_f32_input,
                               &output_size_dims,
                               output_size,
                               &wrong_n,
                               resize_nn_f32_output);
    const cmsis_nn_dims wrong_c = {1, 2, 6, 3};
    RESIZE_NN_EXPECT_ARG_ERROR(&ctx,
                               &params,
                               &input_dims,
                               resize_nn_f32_input,
                               &output_size_dims,
                               output_size,
                               &wrong_c,
                               resize_nn_f32_output);
    const cmsis_nn_dims zero_in = {1, 2, 0, 2};
    const cmsis_nn_dims zero_in_out = {1, 2, 6, 2};
    RESIZE_NN_EXPECT_ARG_ERROR(&ctx,
                               &params,
                               &zero_in,
                               resize_nn_f32_input,
                               &output_size_dims,
                               output_size,
                               &zero_in_out,
                               resize_nn_f32_output);

    /* Dimensions whose product overflows int32_t, and a shape tensor whose negative dims multiply to 2. */
    const cmsis_nn_dims overflow_in = {1, 65536, 65536, 1};
    const cmsis_nn_dims overflow_out = {1, 2, 6, 1};
    RESIZE_NN_EXPECT_ARG_ERROR(&ctx,
                               &params,
                               &overflow_in,
                               resize_nn_f32_input,
                               &output_size_dims,
                               output_size,
                               &overflow_out,
                               resize_nn_f32_output);
    const cmsis_nn_dims negative_size_dims = {1, 1, -1, -2};
    RESIZE_NN_EXPECT_ARG_ERROR(&ctx,
                               &params,
                               &input_dims,
                               resize_nn_f32_input,
                               &negative_size_dims,
                               output_size,
                               &output_dims,
                               resize_nn_f32_output);

#undef RESIZE_NN_EXPECT_ARG_ERROR
#undef RESIZE_NN_CALL

    /* No rejected call may have written past the baseline's output or its scratch maps. */
    for (size_t i = 2 * 6 * 2; i < RESIZE_NN_FLT_MAX_OUTPUT + RESIZE_NN_GUARD; ++i)
    {
        uint32_t bits;
        memcpy(&bits, &resize_nn_f32_output[i], sizeof(bits));
        TEST_ASSERT_EQUAL_HEX32(RESIZE_NN_CANARY_WORD, bits);
    }
    for (size_t i = 0; i < RESIZE_NN_FLT_MAX_SCRATCH_ELEMENTS + RESIZE_NN_GUARD; ++i)
    {
        TEST_ASSERT_EQUAL_HEX32(RESIZE_NN_CANARY_WORD, (uint32_t)resize_nn_f32_scratch[i]);
    }
}

void resize_nearest_neighbor_f32_buffer_size(void)
{
    const cmsis_nn_dims dims = {1, 7, 9, 3};
    TEST_ASSERT_EQUAL_INT32(16 * (int32_t)sizeof(int32_t), arm_resize_nearest_neighbor_f32_get_buffer_size(&dims));
    const cmsis_nn_dims one = {4, 1, 1, 64};
    TEST_ASSERT_EQUAL_INT32(2 * (int32_t)sizeof(int32_t), arm_resize_nearest_neighbor_f32_get_buffer_size(&one));
    TEST_ASSERT_EQUAL_INT32(-1, arm_resize_nearest_neighbor_f32_get_buffer_size(NULL));
    const cmsis_nn_dims zero_h = {1, 0, 9, 3};
    TEST_ASSERT_EQUAL_INT32(-1, arm_resize_nearest_neighbor_f32_get_buffer_size(&zero_h));
    const cmsis_nn_dims neg_w = {1, 7, -1, 3};
    TEST_ASSERT_EQUAL_INT32(-1, arm_resize_nearest_neighbor_f32_get_buffer_size(&neg_w));
    const cmsis_nn_dims huge = {1, INT32_MAX, 1, 1};
    TEST_ASSERT_EQUAL_INT32(-1, arm_resize_nearest_neighbor_f32_get_buffer_size(&huge));
    const cmsis_nn_dims huge_product = {1, INT32_MAX / 2, INT32_MAX / 2, 1};
    TEST_ASSERT_EQUAL_INT32(-1, arm_resize_nearest_neighbor_f32_get_buffer_size(&huge_product));
}

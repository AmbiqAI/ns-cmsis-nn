/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

/*
 * Shared pooling-window geometry cases (#630) for arm_max_pool_{s8,s16,f16,f32} and arm_avg_pool_{f16,f32}. The
 * including suite defines:
 *   PW_PREFIX    name prefix of the generated case functions
 *   PW_KERNEL    kernel under test
 *   PW_T         element type
 *   PW_PARAMS_T  pooling parameter type
 *   PW_ACT_MIN   activation minimum, PW_ACT_MAX activation maximum (wide enough not to clamp the test data)
 *   PW_AVG       1 for average pooling, 0 for max pooling
 *   PW_TOL       absolute tolerance against the float reference
 *   PW_CH        channel count of the multi-channel cases (one vector block plus a tail on MVE builds)
 *
 * Every case fills the whole output buffer with a byte pattern first. A rejected call, and a call with an empty
 * output, must leave every byte of it in place; a computed call must leave the bytes past its output in place.
 */

#pragma once

#include <arm_nnfunctions.h>
#include <stdint.h>
#include <string.h>
#include <unity.h>

#define PW_CAT_(a, b) a##_##b
#define PW_CAT(a, b) PW_CAT_(a, b)
#define PW_FN(name) PW_CAT(PW_PREFIX, name)

#define PW_FILL 0x55
/* Room for three batches of the 3x4 batch layer, the largest input any case uses. */
#define PW_BUF_MAX (36 * PW_CH)

typedef struct
{
    int32_t in_h;
    int32_t in_w;
    int32_t ch;
    int32_t k_h;
    int32_t k_w;
    int32_t stride_h;
    int32_t stride_w;
    int32_t pad_h;
    int32_t pad_w;
    int32_t out_h;
    int32_t out_w;
} pw_geom;

/* One axis of a single-channel layer: input extent w, filter extent k, stride s, padding p, output extent n, and
   whether every window overlaps the input. The other axis is 1 throughout. */
typedef struct
{
    int32_t w;
    int32_t k;
    int32_t s;
    int32_t p;
    int32_t n;
    int valid;
} pw_axis_case;

static PW_T pw_input[PW_BUF_MAX];
static PW_T pw_output[PW_BUF_MAX];

/* Fills the input with small integers, which every element type holds exactly. */
static void pw_fill_input(void)
{
    for (int32_t i = 0; i < PW_BUF_MAX; i++)
    {
        pw_input[i] = (PW_T)(float)((i * 7) % 17 - 8);
    }
}

static void pw_fill_output(void) { memset(pw_output, PW_FILL, sizeof(pw_output)); }

/* Checks that the output bytes from element first onwards still hold the fill pattern. */
static void pw_assert_untouched_from(const int32_t first)
{
    const uint8_t *bytes = (const uint8_t *)&pw_output[first];
    TEST_ASSERT_EACH_EQUAL_HEX8(PW_FILL, bytes, (PW_BUF_MAX - first) * (int32_t)sizeof(PW_T));
}

static arm_cmsis_nn_status pw_run(const pw_geom *g, const int32_t batch)
{
    const cmsis_nn_dims input_dims = {batch, g->in_h, g->in_w, g->ch};
    const cmsis_nn_dims filter_dims = {1, g->k_h, g->k_w, 1};
    const cmsis_nn_dims output_dims = {batch, g->out_h, g->out_w, g->ch};
    PW_PARAMS_T pool_params;
    pool_params.stride.w = g->stride_w;
    pool_params.stride.h = g->stride_h;
    pool_params.padding.w = g->pad_w;
    pool_params.padding.h = g->pad_h;
    pool_params.activation.min = PW_ACT_MIN;
    pool_params.activation.max = PW_ACT_MAX;
    const cmsis_nn_context ctx = {NULL, 0};
    return PW_KERNEL(&ctx, &pool_params, &input_dims, pw_input, &filter_dims, &output_dims, pw_output);
}

/* The in-input part [*lo, *hi) of the window [b, b + k) on an axis of extent w, formed in int64_t. */
static void pw_clip(const int64_t b, const int32_t k, const int32_t w, int64_t *lo, int64_t *hi)
{
    *lo = b > 0 ? b : 0;
    *hi = b + k < w ? b + k : w;
}

/* Runs a layer that must succeed and checks every output element against the definition: the reduction of the input
   elements whose position lies in its window [i * stride - padding, i * stride - padding + filter) on both axes, with
   the window formed in int64_t. Also checks the bytes past the output. */
static void pw_check_valid(const pw_geom *g, const int32_t batch)
{
    const int32_t in_size = g->in_h * g->in_w * g->ch;
    const int32_t out_size = g->out_h * g->out_w * g->ch;
    TEST_ASSERT_TRUE(in_size * batch <= PW_BUF_MAX);
    TEST_ASSERT_TRUE(out_size * batch <= PW_BUF_MAX);
    pw_fill_output();
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, pw_run(g, batch));
    for (int32_t nb = 0; nb < batch; nb++)
    {
        const PW_T *in = &pw_input[nb * in_size];
        const PW_T *out = &pw_output[nb * out_size];
        for (int32_t oy = 0; oy < g->out_h; oy++)
        {
            int64_t y_lo;
            int64_t y_hi;
            pw_clip((int64_t)oy * g->stride_h - g->pad_h, g->k_h, g->in_h, &y_lo, &y_hi);
            for (int32_t ox = 0; ox < g->out_w; ox++)
            {
                int64_t x_lo;
                int64_t x_hi;
                pw_clip((int64_t)ox * g->stride_w - g->pad_w, g->k_w, g->in_w, &x_lo, &x_hi);
                TEST_ASSERT_TRUE((y_lo < y_hi) && (x_lo < x_hi));
                for (int32_t c = 0; c < g->ch; c++)
                {
                    float acc = 0.0f;
                    int32_t count = 0;
                    for (int64_t y = y_lo; y < y_hi; y++)
                    {
                        for (int64_t x = x_lo; x < x_hi; x++)
                        {
                            const float v = (float)in[(y * g->in_w + x) * g->ch + c];
#if PW_AVG
                            acc += v;
#else
                            if ((count == 0) || (v > acc))
                            {
                                acc = v;
                            }
#endif
                            count++;
                        }
                    }
#if PW_AVG
                    acc /= (float)count;
#endif
                    TEST_ASSERT_FLOAT_WITHIN(PW_TOL, acc, (float)out[(oy * g->out_w + ox) * g->ch + c]);
                }
            }
        }
    }
    pw_assert_untouched_from(out_size * batch);
}

/* Runs a layer that must return status and leave the whole output buffer untouched. */
static void pw_check_untouched(const pw_geom *g, const int32_t batch, const arm_cmsis_nn_status status)
{
    pw_fill_output();
    TEST_ASSERT_EQUAL(status, pw_run(g, batch));
    pw_assert_untouched_from(0);
}

/* Runs each single-channel axis case along x and along y on the current input. */
static void pw_check_axis_cases(const pw_axis_case *cases, const size_t count)
{
    for (int axis = 0; axis < 2; axis++)
    {
        for (size_t i = 0; i < count; i++)
        {
            const pw_axis_case *a = &cases[i];
            const pw_geom g = axis == 0 ? (pw_geom){1, a->w, 1, 1, a->k, 1, a->s, 0, a->p, 1, a->n}
                                        : (pw_geom){a->w, 1, 1, a->k, 1, a->s, 1, a->p, 0, a->n, 1};
            if (a->valid)
            {
                pw_check_valid(&g, 1);
            }
            else
            {
                pw_check_untouched(&g, 1, ARM_CMSIS_NN_ARG_ERROR);
            }
        }
    }
}

/* A layer in which some output window lies entirely outside the input, past its end or in the padding (padding at
   least the filter extent), along x, along y or both, is rejected with ARM_CMSIS_NN_ARG_ERROR before any output is
   written. */
void PW_FN(empty_window)(void)
{
    pw_fill_input();
    /* in_h, in_w, ch, k_h, k_w, stride_h, stride_w, pad_h, pad_w, out_h, out_w */
    const pw_geom cases[] = {
        {2, 2, PW_CH, 1, 1, 1, 1, 0, 0, 3, 3},
        {2, 2, PW_CH, 1, 1, 1, 1, 0, 0, 2, 3},
        {2, 2, PW_CH, 1, 1, 1, 1, 0, 0, 3, 2},
        {2, 2, PW_CH, 1, 1, 1, 1, 0, 1, 2, 3},
        {2, 2, PW_CH, 1, 1, 1, 1, 1, 0, 3, 2},
        {2, 2, PW_CH, 1, 1, 1, 1, 0, 2, 2, 4},
        {2, 2, PW_CH, 2, 2, 1, 1, 2, 2, 1, 1},
        {2, 2, PW_CH, 2, 2, 1, 1, 2, 2, 3, 3},
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++)
    {
        pw_check_untouched(&cases[i], 1, ARM_CMSIS_NN_ARG_ERROR);
    }
}

/* Window positions near the int32_t limits along one axis, single channel. Valid layers are computed and the rest
   are rejected before any output is written: a filter extent that takes the last window end past INT32_MAX, a last
   window past the input, a stride that puts the second window past the input, a step one stride past the last
   window that leaves the int32_t range (positive and negative stride), and an input extent minus window position
   that leaves it. A step past the last window that lands exactly on INT32_MIN is in range. */
void PW_FN(window_bound_limits)(void)
{
    const pw_axis_case cases[] = {
        {2, INT32_MAX - 1, 1, 1, 3, 1},
        {3, INT32_MAX - 2, INT32_MAX - 2, INT32_MAX - 3, 2, 1},
        {2, INT32_MAX, 1, 1, 3, 0},
        {2, INT32_MAX - 1, 1, 1, 4, 0},
        {1, 1, 2000000000, 0, 2, 0},
        {3, INT32_MAX - 2, INT32_MAX - 1, INT32_MAX - 3, 2, 0},
        {1, 2100000000, -1000000000, 2000000000, 1, 0},
        {2, INT32_MAX, 1, INT32_MAX - 1, 1, 0},
        /* Two windows at 0 and -2^30: the step past the last one is exactly INT32_MIN, which fits; one more
           negative and it does not. */
        {1, 1073741825, -1073741824, 0, 2, 1},
        {1, 1073741826, -1073741825, 0, 2, 0},
    };
    pw_input[0] = (PW_T)20.0f;
    pw_input[1] = (PW_T)10.0f;
    pw_input[2] = (PW_T)30.0f;
    pw_check_axis_cases(cases, sizeof(cases) / sizeof(cases[0]));
}

/* An output with no rows or no columns, or a negative extent, has no window: the call succeeds and writes nothing,
   whatever the other extent, the stride, or an input whose element count does not fit in an int32_t. */
void PW_FN(empty_output)(void)
{
    pw_fill_input();
    const pw_geom cases[] = {
        {1, 1, PW_CH, 1, 1, 1, 1, 0, 0, 2, 0},
        {1, 1, PW_CH, 1, 1, 1, 1, 0, 0, 0, 3},
        {1, 1, PW_CH, 1, 1, 1, 1, 0, 0, 2, -1},
        {1, 1, PW_CH, 1, 1, 1, 1, 0, 0, -1, 2},
        {1, 1, PW_CH, 1, 1, 2000000000, 1, 0, 0, 3, 0},
        {1, 1, PW_CH, 1, 1, 1, 2000000000, 0, 0, 0, 3},
        {2, INT32_MAX, PW_CH, 1, 1, 1, 1, 0, 0, 0, 1},
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++)
    {
        pw_check_untouched(&cases[i], 1, ARM_CMSIS_NN_SUCCESS);
    }
}

/* An ordinary layer with different stride, padding and filter extent along y and x (input 4x5, filter 3x2, stride
   2x1, padding 1x0, output 2x4), against the reference. Catches the two axes being swapped in the window bounds. */
void PW_FN(asymmetric_axes)(void)
{
    pw_fill_input();
    const pw_geom g = {4, 5, PW_CH, 3, 2, 2, 1, 1, 0, 2, 4};
    pw_check_valid(&g, 1);
}

/* Negative strides, whose windows run from the end of the input back to its start, along x and along y: computed
   against the reference when every window overlaps the input, and rejected when the last window has moved wholly
   before the input or the first window starts past it. The first window is then the highest one, so these catch
   the check taking the first window for the lowest. */
void PW_FN(negative_stride)(void)
{
    pw_fill_input();
    /* in_h, in_w, ch, k_h, k_w, stride_h, stride_w, pad_h, pad_w, out_h, out_w */
    const pw_geom valid[] = {
        {3, 4, PW_CH, 2, 2, -1, -1, -1, -2, 2, 3},
        {4, 3, PW_CH, 2, 2, -1, -1, -2, -1, 3, 2},
    };
    const pw_geom empty[] = {
        {3, 4, PW_CH, 2, 2, -1, -1, -1, -2, 2, 5},
        {4, 3, PW_CH, 2, 2, -1, -1, -2, -1, 5, 2},
        {3, 4, PW_CH, 2, 2, -1, -1, -1, -4, 2, 3},
        {4, 3, PW_CH, 2, 2, -1, -1, -4, -1, 3, 2},
    };
    for (size_t i = 0; i < sizeof(valid) / sizeof(valid[0]); i++)
    {
        pw_check_valid(&valid[i], 1);
    }
    for (size_t i = 0; i < sizeof(empty) / sizeof(empty[0]); i++)
    {
        pw_check_untouched(&empty[i], 1, ARM_CMSIS_NN_ARG_ERROR);
    }
}

/* Negative padding, which offsets the first window into the input: computed against the reference, and rejected
   once a later window starts past the input. */
void PW_FN(negative_padding)(void)
{
    pw_fill_input();
    const pw_geom valid = {3, 4, PW_CH, 2, 2, 1, 1, -1, -1, 2, 2};
    const pw_geom empty_x = {3, 4, PW_CH, 2, 2, 1, 1, -1, -1, 2, 4};
    const pw_geom empty_y = {3, 4, PW_CH, 2, 2, 1, 1, -1, -1, 3, 2};
    pw_check_valid(&valid, 1);
    pw_check_untouched(&empty_x, 1, ARM_CMSIS_NN_ARG_ERROR);
    pw_check_untouched(&empty_y, 1, ARM_CMSIS_NN_ARG_ERROR);
}

/* Two and three batches, each pooled from its own input into its own output, and an empty window rejected for every
   batch before the first is written. */
void PW_FN(batches)(void)
{
    pw_fill_input();
    const pw_geom g = {3, 4, PW_CH, 2, 2, 1, 2, 0, 1, 2, 3};
    const pw_geom empty = {3, 4, PW_CH, 2, 2, 1, 2, 0, 1, 2, 4};
    pw_check_valid(&g, 2);
    pw_check_valid(&g, 3);
    pw_check_untouched(&empty, 2, ARM_CMSIS_NN_ARG_ERROR);
}

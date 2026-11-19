// SPDX-FileCopyrightText: Copyright 2026 Ambiq
// SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK

#include <stdio.h>
#include <string.h>

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

static const int8_t *input_base;
static unsigned called_rows;
static unsigned call_count;
static unsigned skipped_calls;

arm_cmsis_nn_status __real_arm_nn_transpose_conv_row_s8_s32(const int8_t *lhs,
                                                         const int8_t *rhs,
                                                         int32_t *output_start,
                                                         int32_t output_index,
                                                         int32_t output_max,
                                                         int32_t rhs_rows,
                                                         int32_t rhs_cols,
                                                         int32_t input_channels,
                                                         int32_t output_channels,
                                                         int32_t lhs_offset,
                                                         int32_t row_offset,
                                                         int32_t input_x,
                                                         int32_t stride_x,
                                                         int32_t skip_row_top,
                                                         int32_t skip_row_bottom);

arm_cmsis_nn_status __wrap_arm_nn_transpose_conv_row_s8_s32(const int8_t *lhs,
                                                         const int8_t *rhs,
                                                         int32_t *output_start,
                                                         int32_t output_index,
                                                         int32_t output_max,
                                                         int32_t rhs_rows,
                                                         int32_t rhs_cols,
                                                         int32_t input_channels,
                                                         int32_t output_channels,
                                                         int32_t lhs_offset,
                                                         int32_t row_offset,
                                                         int32_t input_x,
                                                         int32_t stride_x,
                                                         int32_t skip_row_top,
                                                         int32_t skip_row_bottom)
{
    ++call_count;
    called_rows |= 1u << (unsigned)(lhs - input_base);
    if (skip_row_top + skip_row_bottom >= rhs_rows)
    {
        ++skipped_calls;
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    return __real_arm_nn_transpose_conv_row_s8_s32(lhs, rhs, output_start, output_index, output_max, rhs_rows,
                                                  rhs_cols, input_channels, output_channels, lhs_offset, row_offset,
                                                  input_x, stride_x, skip_row_top, skip_row_bottom);
}

int main(void)
{
    static const struct
    {
        const char *name;
        int input_h, stride_h, pad_h, output_h;
        unsigned rows, calls;
    } cases[] = {
        {"top exceeds filter", 2, 3, 2, 2, 2, 1},
        {"top equals filter", 2, 3, 1, 3, 2, 1},
        {"all rows live", 2, 3, 0, 4, 3, 2},
        {"bottom equals filter", 2, 2, 0, 1, 1, 1},
        {"bottom exceeds filter", 3, 3, 0, 1, 1, 1},
    };
    const int8_t input[] = {3, 7, 11};
    const int8_t filter[] = {2};
    const int32_t bias[] = {1};
    int32_t multiplier[] = {1073741824}, shift[] = {1};
    const cmsis_nn_per_channel_quant_params quant = {multiplier, shift};
    const cmsis_nn_dims filter_dims = {1, 1, 1, 1}, bias_dims = {1, 1, 1, 1};
    input_base = input;
    for (unsigned i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i)
    {
        int32_t scratch[16];
        int8_t output[6];
        memset(output, 0x55, sizeof(output));
        const cmsis_nn_context ctx = {scratch, sizeof(scratch)};
        const cmsis_nn_dims input_dims = {1, cases[i].input_h, 1, 1};
        const cmsis_nn_dims output_dims = {1, cases[i].output_h, 1, 1};
        cmsis_nn_transpose_conv_params params = {0};
        params.stride.h = cases[i].stride_h;
        params.stride.w = 1;
        params.padding.h = cases[i].pad_h;
        params.dilation.h = params.dilation.w = 1;
        params.activation.min = -128;
        params.activation.max = 127;
        called_rows = call_count = skipped_calls = 0;
        const arm_cmsis_nn_status status = arm_transpose_conv_s8(
            &ctx, NULL, &params, &quant, &input_dims, input, &filter_dims, filter, &bias_dims, bias, &output_dims, output + 1);
        if (status != ARM_CMSIS_NN_SUCCESS || skipped_calls || called_rows != cases[i].rows || call_count != cases[i].calls)
        {
            fprintf(stderr, "%s: status=%d, skipped calls=%u, rows=%u, calls=%u\n",
                    cases[i].name, status, skipped_calls, called_rows, call_count);
            return 1;
        }
        for (int y = 0; y < cases[i].output_h; ++y)
        {
            int expected = bias[0];
            for (int row = 0; row < cases[i].input_h; ++row)
            {
                if (row * cases[i].stride_h - cases[i].pad_h == y)
                {
                    expected += input[row] * filter[0];
                }
            }
            if (output[y + 1] != expected)
            {
                fprintf(stderr, "%s: output row %d: %d != %d\n", cases[i].name, y, output[y + 1], expected);
                return 1;
            }
        }
        if (output[0] != 0x55 || output[cases[i].output_h + 1] != 0x55)
        {
            fprintf(stderr, "%s: output guard changed\n", cases[i].name);
            return 1;
        }
    }
    return 0;
}

/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

/* Expected weights are the independently enumerated signed pair, never decoded
 * from the packed tensor. Even products/bias and a half-scale requantizer make
 * the reference an exact division, independent of production rounding code. */
static int weight(int index, int low, int high) { return (index % 2) ? high : low; }

static int expected(int64_t sum, int control)
{
    if (sum % 2 != 0)
    {
        return 1000;
    }
    int result = (int)(sum / 2);
    if (control)
    {
        if (result < -4)
            result = -4;
        if (result > 4)
            result = 4;
    }
    else if (result < -127 || result > 126)
    {
        return 1000; /* An oracle-bound failure must not look like saturation. */
    }
    return result;
}

static void pack(int8_t *dst, size_t bytes, int low, int high)
{
    const unsigned lo = (unsigned)(low < 0 ? low + 16 : low);
    const unsigned hi = (unsigned)(high < 0 ? high + 16 : high);
    const uint8_t byte = (uint8_t)(lo + 16U * hi);
    memset(dst, byte, bytes);
}

static void quant(int32_t *mult, int32_t *shift, int32_t *bias, int channels, int control)
{
    for (int c = 0; c < channels; ++c)
    {
        mult[c] = 0x40000000;
        shift[c] = 0;
        bias[c] = control ? (c % 2 ? -2 : 2) : 0;
    }
}

static int activation(int index, int phase, int control)
{
    return control ? (index % 2 ? -2 : 2) : (index == phase ? 2 : 0);
}

static int check(const char *id, const int8_t *output, const int *reference, int count, int low, int high)
{
    for (int i = 0; i < count; ++i)
    {
        if (output[i + 16] != reference[i])
        {
            printf("%s pair(%d,%d) output%d: got%d expected%d\n", id, low, high, i, output[i + 16], reference[i]);
            return 1;
        }
    }
    for (int i = 0; i < 128; ++i)
    {
        if ((i < 16 || i >= count + 16) && output[i] != 85)
        {
            printf("%s output guard%d changed\n", id, i);
            return 1;
        }
    }
    return 0;
}

static int depthwise(const char *id, int channels, int multiplier, int low, int high, int phase, int control)
{
    const int out_channels = channels * multiplier;
    int8_t input[36], filter[16], output[128];
    int reference[36];
    int32_t mult[6], shift[6], bias[6];
    for (int i = 0; i < 12 * channels; ++i)
        input[i] = (int8_t)activation(i, phase, control);
    pack(filter, sizeof(filter), low, high);
    memset(output, 85, sizeof(output));
    quant(mult, shift, bias, out_channels, control);
    cmsis_nn_dims in = {1, 3, 4, channels}, kernel = {1, 2, 2, out_channels};
    cmsis_nn_dims out = {1, 2, 3, out_channels}, bias_dims = {1, 1, 1, out_channels};
    cmsis_nn_dw_conv_params params = {0};
    params.ch_mult = multiplier;
    params.stride.h = params.stride.w = 1;
    params.dilation.h = params.dilation.w = 1;
    params.activation.min = control ? -4 : -128;
    params.activation.max = control ? 4 : 127;
    cmsis_nn_per_channel_quant_params q = {mult, shift};
    if (arm_depthwise_conv_s4(NULL, &params, &q, &in, input, &kernel, filter, &bias_dims, bias, &out, output + 16) !=
        ARM_CMSIS_NN_SUCCESS)
        return 1;
    int index = 0;
    for (int y = 0; y < 2; ++y)
        for (int x = 0; x < 3; ++x)
            for (int c = 0; c < out_channels; ++c)
            {
                int64_t sum = bias[c];
                for (int ky = 0; ky < 2; ++ky)
                    for (int kx = 0; kx < 2; ++kx)
                        sum += (int64_t)input[((y + ky) * 4 + x + kx) * channels + c / multiplier] *
                            weight((ky * 2 + kx) * out_channels + c, low, high);
                reference[index++] = expected(sum, control);
            }
    return check(id, output, reference, index, low, high);
}

static int matrix(const char *id, int channels, int cols, int low, int high, int phase, int control)
{
    int8_t filter[32], output[128];
    int16_t input[14];
    int reference[14];
    int32_t mult[7], shift[7], bias[7];
    for (int p = 0; p < 2; ++p)
        for (int i = 0; i < cols; ++i)
            input[p * cols + i] = (int16_t)activation(i, phase ^ p, control);
    pack(filter, sizeof(filter), low, high);
    memset(output, 85, sizeof(output));
    quant(mult, shift, bias, channels, control);
    int8_t *end = arm_nn_mat_mult_kernel_s4_s16(filter,
                                                input,
                                                (uint16_t)channels,
                                                shift,
                                                mult,
                                                0,
                                                control ? -4 : -128,
                                                control ? 4 : 127,
                                                cols,
                                                bias,
                                                output + 16);
    if (end != output + 16 + 2 * channels)
        return 1;
    for (int p = 0; p < 2; ++p)
        for (int c = 0; c < channels; ++c)
        {
            int64_t sum = bias[c];
            for (int i = 0; i < cols; ++i)
                sum += (int64_t)input[p * cols + i] * weight(c * cols + i, low, high);
            reference[p * channels + c] = expected(sum, control);
        }
    return check(id, output, reference, 2 * channels, low, high);
}

static int convolution(int low, int high, int phase, int control)
{
    int8_t input[21], filter[16], output[128];
    int reference[9];
    int32_t mult[3], shift[3], bias[3];
    uint64_t scratch[514];
    for (int p = 0; p < 3; ++p)
        for (int i = 0; i < 7; ++i)
            input[p * 7 + i] = (int8_t)activation(i, phase, control);
    pack(filter, sizeof(filter), low, high);
    memset(output, 85, sizeof(output));
    memset(scratch, 85, sizeof(scratch));
    quant(mult, shift, bias, 3, control);
    cmsis_nn_dims in = {1, 1, 3, 7}, kernel = {3, 1, 1, 7};
    cmsis_nn_dims out = {1, 1, 3, 3}, bias_dims = {1, 1, 1, 3};
    cmsis_nn_conv_params params = {0};
    params.stride.h = params.stride.w = 1;
    params.dilation.h = params.dilation.w = 1;
    params.activation.min = control ? -4 : -128;
    params.activation.max = control ? 4 : 127;
    cmsis_nn_per_channel_quant_params q = {mult, shift};
    const int32_t size = arm_convolve_s4_get_buffer_size(&in, &kernel);
    if (size <= 0 || size > 4096)
        return 1;
    cmsis_nn_context ctx = {scratch + 1, size};
    if (arm_convolve_s4(&ctx, &params, &q, &in, input, &kernel, filter, &bias_dims, bias, &out, output + 16) !=
        ARM_CMSIS_NN_SUCCESS)
        return 1;
    const uint8_t *bytes = (const uint8_t *)scratch;
    for (size_t i = 0; i < sizeof(scratch); ++i)
        if ((i < 8 || i >= 8U + (size_t)size) && bytes[i] != 85)
        {
            printf("C7 scratch guard%u changed\n", (unsigned)i);
            return 1;
        }
    for (int p = 0; p < 3; ++p)
        for (int c = 0; c < 3; ++c)
        {
            int64_t sum = bias[c];
            for (int i = 0; i < 7; ++i)
                sum += (int64_t)input[p * 7 + i] * weight(c * 7 + i, low, high);
            reference[p * 3 + c] = expected(sum, control);
        }
    return check("C7", output, reference, 9, low, high);
}

static int run_case(int which, int low, int high, int phase, int control)
{
    switch (which)
    {
    case 0:
        return depthwise("D1", 2, 1, low, high, phase, control);
    case 1:
        return depthwise("D3", 2, 3, low, high, phase, control);
    case 2:
        return depthwise("D2", 2, 2, low, high, phase, control);
    case 3:
        return depthwise("DO", 3, 1, low, high, phase, control);
    case 4:
        return matrix("M46", 4, 6, low, high, phase, control);
    case 5:
        return matrix("M57", 5, 7, low, high, phase, control);
    case 6:
        return matrix("M66", 6, 6, low, high, phase, control);
    case 7:
        return matrix("M77", 7, 7, low, high, phase, control);
    default:
        return convolution(low, high, phase, control);
    }
}

int main(int argc, char **argv)
{
    const char *names[] = {"D1", "D3", "D2", "DO", "M46", "M57", "M66", "M77", "C7"};
    int executed = 0;
    for (int c = 0; c < 9; ++c)
    {
        if (argc > 1 && strcmp(argv[1], names[c]) != 0)
            continue;
        int pairs = 0;
        for (int low = -8; low <= 7; ++low)
            for (int high = -8; high <= 7; ++high)
            {
                for (int phase = 0; phase < 2; ++phase)
                    if (run_case(c, low, high, phase, 0))
                        return 1;
                ++pairs;
            }
        if (pairs != 256 || run_case(c, -7, 3, 0, 1))
            return 1;
        printf("%s PASS: 256 pairs, two phases, signed/bias/clamp control\n", names[c]);
        ++executed;
    }
    if (executed == 0)
        return 1;
    printf("INT4 PASS: %d route cases\n", executed);
    return 0;
}

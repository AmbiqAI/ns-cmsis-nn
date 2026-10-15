/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 */

#ifndef DEPTH_SPACE_S8_TEST_H
#define DEPTH_SPACE_S8_TEST_H

#include <arm_nnfunctions.h>
#include <string.h>

/* The oracle indexes each scalar output independently of the kernel's copy order. */
static void depth_space_permutations_s8(
    int depth_to_space,
    arm_cmsis_nn_status (*kernel)(const int8_t *, const cmsis_nn_dims *, int32_t, int8_t *, const cmsis_nn_dims *))
{
    static int8_t input_storage[6400];
    static int8_t output_storage[6400];
    const int channels[] = {1, 3, 7, 8, 15, 16, 17, 31, 32, 33};
    for (int block = 1; block <= 4; ++block)
    {
        for (unsigned k = 0; k < sizeof(channels) / sizeof(channels[0]); ++k)
        {
            const int c = channels[k];
            const cmsis_nn_dims space = {2, 2 * block, 3 * block, c};
            const cmsis_nn_dims depth = {2, 2, 3, c * block * block};
            const cmsis_nn_dims *in = depth_to_space ? &depth : &space;
            const cmsis_nn_dims *out = depth_to_space ? &space : &depth;
            const int count = space.n * space.h * space.w * space.c;
            const int input_offset = 16 + (k % 4);
            const int output_offset = 16 + ((k + 1) % 4);
            int8_t *input = input_storage + input_offset;
            int8_t *output = output_storage + output_offset;
            memset(input_storage, 0x5a, sizeof(input_storage));
            memset(output_storage, 0x6b, sizeof(output_storage));
            for (int i = 0; i < count; ++i)
            {
                input[i] = (int8_t)((i * 73 + i / 127) % 256 - 128);
            }
            TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, kernel(input, in, block, output, out));
            for (int n = 0; n < out->n; ++n)
            {
                for (int y = 0; y < out->h; ++y)
                {
                    for (int x = 0; x < out->w; ++x)
                    {
                        for (int channel = 0; channel < out->c; ++channel)
                        {
                            int iy, ix, ic;
                            if (depth_to_space)
                            {
                                iy = y / block;
                                ix = x / block;
                                ic = ((y % block) * block + x % block) * out->c + channel;
                            }
                            else
                            {
                                iy = y * block + channel / (block * in->c);
                                ix = x * block + (channel / in->c) % block;
                                ic = channel % in->c;
                            }
                            const int src = ((n * in->h + iy) * in->w + ix) * in->c + ic;
                            const int dst = ((n * out->h + y) * out->w + x) * out->c + channel;
                            TEST_ASSERT_EQUAL_INT8(input[src], output[dst]);
                        }
                    }
                }
            }
            for (int i = 0; i < count; ++i)
            {
                TEST_ASSERT_EQUAL_INT8((int8_t)((i * 73 + i / 127) % 256 - 128), input[i]);
            }
            for (int i = 0; i < output_offset; ++i)
            {
                TEST_ASSERT_EQUAL_INT8(0x6b, output_storage[i]);
            }
            for (unsigned i = output_offset + count; i < sizeof(output_storage); ++i)
            {
                TEST_ASSERT_EQUAL_INT8(0x6b, output_storage[i]);
            }
        }
    }
}

#endif

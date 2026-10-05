/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_nn_copy_fill_words_s8.c
 * Description:  Byte copy and fill a word at a time, for toolchains whose C library moves one byte at a time.
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "arm_nnsupportfunctions.h"

/*
 * Four words per iteration, then a loop-free tail: clang recognises a plain word or byte loop as memcpy or memset
 * and calls the C library, which these functions replace.
 */
void arm_nn_copy_words_s8(int8_t *dst, const int8_t *src, uint32_t block_size)
{
    while (block_size >= 16)
    {
        const int32_t a = arm_nn_read_s8x4_ia(&src);
        const int32_t b = arm_nn_read_s8x4_ia(&src);
        const int32_t c = arm_nn_read_s8x4_ia(&src);
        const int32_t d = arm_nn_read_s8x4_ia(&src);
        arm_nn_write_s8x4_ia(&dst, a);
        arm_nn_write_s8x4_ia(&dst, b);
        arm_nn_write_s8x4_ia(&dst, c);
        arm_nn_write_s8x4_ia(&dst, d);
        block_size -= 16;
    }
    if (block_size & 8)
    {
        const int32_t a = arm_nn_read_s8x4_ia(&src);
        const int32_t b = arm_nn_read_s8x4_ia(&src);
        arm_nn_write_s8x4_ia(&dst, a);
        arm_nn_write_s8x4_ia(&dst, b);
    }
    if (block_size & 4)
    {
        arm_nn_write_s8x4_ia(&dst, arm_nn_read_s8x4_ia(&src));
    }
    if (block_size & 2)
    {
        dst[0] = src[0];
        dst[1] = src[1];
        dst += 2;
        src += 2;
    }
    if (block_size & 1)
    {
        dst[0] = src[0];
    }
}

void arm_nn_fill_words_s8(int8_t *dst, const int32_t pattern, uint32_t block_size)
{
    while (block_size >= 16)
    {
        arm_nn_write_s8x4_ia(&dst, pattern);
        arm_nn_write_s8x4_ia(&dst, pattern);
        arm_nn_write_s8x4_ia(&dst, pattern);
        arm_nn_write_s8x4_ia(&dst, pattern);
        block_size -= 16;
    }
    if (block_size & 8)
    {
        arm_nn_write_s8x4_ia(&dst, pattern);
        arm_nn_write_s8x4_ia(&dst, pattern);
    }
    if (block_size & 4)
    {
        arm_nn_write_s8x4_ia(&dst, pattern);
    }
    if (block_size & 2)
    {
        memcpy(dst, &pattern, 2);
        dst += 2;
    }
    if (block_size & 1)
    {
        memcpy(dst, &pattern, 1);
    }
}

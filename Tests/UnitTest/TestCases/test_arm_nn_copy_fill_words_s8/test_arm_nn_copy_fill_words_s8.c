/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include <arm_nnsupportfunctions.h>
#include <string.h>

#include "unity.h"

/* This suite is built with ARM_NN_WORD_COPY defined (see CMakeLists.txt), so the copy and fill helpers take the
   word-at-a-time path on every toolchain, not only on the clang builds that select it by default. MVE builds keep
   their own path. */

/* Every size residue modulo 16 and a long size, at every source and destination offset within a word. The source
   ends at the end of its array, so a read past the source is a sanitizer error on the host, and the destination
   has guard bytes on both sides. */
#define CF_MAX (1027)
#define CF_GUARD (8)
#define CF_FILL (0x6B)

/* int16_t storage, so that the int16_t calls see aligned pointers at even offsets */
static int16_t cf_src_h[(CF_MAX + 4) / 2];
static int16_t cf_dst_h[(CF_MAX + 4 + 2 * CF_GUARD) / 2 + 1];
static int16_t cf_ref_h[(CF_MAX + 4 + 2 * CF_GUARD) / 2 + 1];
#define cf_src ((int8_t *)cf_src_h)
#define cf_dst ((int8_t *)cf_dst_h)
#define cf_ref ((int8_t *)cf_ref_h)

static const uint32_t cf_sizes[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 31, 32, 33, 1027};

static void cf_reset(void)
{
    for (uint32_t i = 0; i < sizeof(cf_src_h); i++)
    {
        cf_src[i] = (int8_t)(i * 37 + 11);
    }
    memset(cf_dst, CF_FILL, sizeof(cf_dst_h));
    memset(cf_ref, CF_FILL, sizeof(cf_ref_h));
}

void copy_words_arm_nn_copy_fill_words_s8(void)
{
    for (size_t k = 0; k < sizeof(cf_sizes) / sizeof(cf_sizes[0]); k++)
    {
        for (uint32_t so = 0; so < 4; so++)
        {
            for (uint32_t dof = 0; dof < 4; dof++)
            {
                const uint32_t n = cf_sizes[k];
                const int8_t *src = cf_src + (sizeof(cf_src_h) - n - so);
                cf_reset();
                for (uint32_t i = 0; i < n; i++)
                {
                    cf_ref[CF_GUARD + dof + i] = src[i];
                }
                arm_nn_copy_words_s8(cf_dst + CF_GUARD + dof, src, n);
                TEST_ASSERT_EQUAL_INT8_ARRAY(cf_ref, cf_dst, sizeof(cf_dst_h));

                memset(cf_dst, CF_FILL, sizeof(cf_dst_h));
                arm_memcpy_s8(cf_dst + CF_GUARD + dof, src, n);
                TEST_ASSERT_EQUAL_INT8_ARRAY(cf_ref, cf_dst, sizeof(cf_dst_h));

                if (((sizeof(cf_src_h) - n - so) | dof) % 2 == 0)
                {
                    memset(cf_dst, CF_FILL, sizeof(cf_dst_h));
                    arm_memcpy_q15((int16_t *)(void *)(cf_dst + CF_GUARD + dof), (const int16_t *)(const void *)src, n);
                    TEST_ASSERT_EQUAL_INT8_ARRAY(cf_ref, cf_dst, sizeof(cf_dst_h));
                }
                if (((sizeof(cf_src_h) - n - so) | dof | n) % 2 == 0)
                {
                    memset(cf_dst, CF_FILL, sizeof(cf_dst_h));
                    arm_memcpy_s16(
                        (int16_t *)(void *)(cf_dst + CF_GUARD + dof), (const int16_t *)(const void *)src, n / 2);
                    TEST_ASSERT_EQUAL_INT8_ARRAY(cf_ref, cf_dst, sizeof(cf_dst_h));
                }
            }
        }
    }
}

void fill_words_arm_nn_copy_fill_words_s8(void)
{
    const int8_t vals[] = {0, -1, 0x5A, -128};
    for (size_t k = 0; k < sizeof(cf_sizes) / sizeof(cf_sizes[0]); k++)
    {
        for (uint32_t dof = 0; dof < 4; dof++)
        {
            const uint32_t n = cf_sizes[k];
            for (size_t v = 0; v < sizeof(vals) / sizeof(vals[0]); v++)
            {
                cf_reset();
                memset(cf_ref + CF_GUARD + dof, vals[v], n);
                arm_memset_s8(cf_dst + CF_GUARD + dof, vals[v], n);
                TEST_ASSERT_EQUAL_INT8_ARRAY(cf_ref, cf_dst, sizeof(cf_dst_h));
            }

            /* A pattern of four different bytes is laid down in memory order from dst onwards */
            const int8_t pattern_bytes[4] = {0x01, 0x02, 0x03, 0x04};
            int32_t pattern;
            memcpy(&pattern, pattern_bytes, 4);
            cf_reset();
            for (uint32_t i = 0; i < n; i++)
            {
                cf_ref[CF_GUARD + dof + i] = pattern_bytes[i % 4];
            }
            arm_nn_fill_words_s8(cf_dst + CF_GUARD + dof, pattern, n);
            TEST_ASSERT_EQUAL_INT8_ARRAY(cf_ref, cf_dst, sizeof(cf_dst_h));
        }
    }
}

void fill_s16_arm_nn_copy_fill_words_s8(void)
{
    const int16_t vals[] = {0, -1, 0x1234, -32768};
    for (size_t k = 0; k < sizeof(cf_sizes) / sizeof(cf_sizes[0]); k++)
    {
        for (uint32_t dof = 0; dof < 4; dof += 2)
        {
            for (size_t v = 0; v < sizeof(vals) / sizeof(vals[0]); v++)
            {
                const uint32_t count = cf_sizes[k];
                if (2 * count + dof > sizeof(cf_dst_h) - 2 * CF_GUARD)
                {
                    continue;
                }
                cf_reset();
                for (uint32_t i = 0; i < count; i++)
                {
                    memcpy(cf_ref + CF_GUARD + dof + 2 * i, &vals[v], 2);
                }
                arm_memset_s16((int16_t *)(void *)(cf_dst + CF_GUARD + dof), vals[v], count);
                TEST_ASSERT_EQUAL_INT8_ARRAY(cf_ref, cf_dst, sizeof(cf_dst_h));
            }
        }
    }
}

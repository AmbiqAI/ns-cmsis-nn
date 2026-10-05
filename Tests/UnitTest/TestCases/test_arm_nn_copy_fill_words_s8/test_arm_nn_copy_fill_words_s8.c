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

/* Sizes across the 16-byte body and every 8/4/2/1 tail combination, at every source and destination offset
   within a word, with guard bytes on both sides of the destination. */
#define CF_MAX (1027)
#define CF_GUARD (8)
#define CF_FILL (0x6B)

/* Word-typed storage, so that offset 0 is word-aligned and the int16_t calls see aligned pointers at even offsets */
static int32_t cf_src_w[(CF_MAX + 4) / 4 + 1];
static int32_t cf_dst_w[(CF_MAX + 4 + 2 * CF_GUARD) / 4 + 1];
static int32_t cf_ref_w[(CF_MAX + 4 + 2 * CF_GUARD) / 4 + 1];
#define cf_src ((int8_t *)cf_src_w)
#define cf_dst ((int8_t *)cf_dst_w)
#define cf_ref ((int8_t *)cf_ref_w)

static const uint32_t cf_sizes[] = {0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 23, 30, 31, 32, 33, 63, 64, 65, 255, 1027};

static void cf_reset(void)
{
    for (uint32_t i = 0; i < sizeof(cf_src_w); i++)
    {
        cf_src[i] = (int8_t)(i * 37 + 11);
    }
    memset(cf_dst, CF_FILL, sizeof(cf_dst_w));
    memset(cf_ref, CF_FILL, sizeof(cf_ref_w));
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
                cf_reset();
                for (uint32_t i = 0; i < n; i++)
                {
                    cf_ref[CF_GUARD + dof + i] = cf_src[so + i];
                }
                arm_nn_copy_words_s8(cf_dst + CF_GUARD + dof, cf_src + so, n);
                TEST_ASSERT_EQUAL_INT8_ARRAY(cf_ref, cf_dst, sizeof(cf_dst_w));

                memset(cf_dst, CF_FILL, sizeof(cf_dst_w));
                arm_memcpy_s8(cf_dst + CF_GUARD + dof, cf_src + so, n);
                TEST_ASSERT_EQUAL_INT8_ARRAY(cf_ref, cf_dst, sizeof(cf_dst_w));

                if ((so | dof) % 2 == 0)
                {
                    memset(cf_dst, CF_FILL, sizeof(cf_dst_w));
                    arm_memcpy_q15(
                        (int16_t *)(void *)(cf_dst + CF_GUARD + dof), (const int16_t *)(const void *)(cf_src + so), n);
                    TEST_ASSERT_EQUAL_INT8_ARRAY(cf_ref, cf_dst, sizeof(cf_dst_w));
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
            for (size_t v = 0; v < sizeof(vals) / sizeof(vals[0]); v++)
            {
                const uint32_t n = cf_sizes[k];
                cf_reset();
                memset(cf_ref + CF_GUARD + dof, vals[v], n);
                arm_memset_s8(cf_dst + CF_GUARD + dof, vals[v], n);
                TEST_ASSERT_EQUAL_INT8_ARRAY(cf_ref, cf_dst, sizeof(cf_dst_w));

                memset(cf_dst, CF_FILL, sizeof(cf_dst_w));
                const int32_t pattern = (int32_t)((uint8_t)vals[v] * 0x01010101U);
                arm_nn_fill_words_s8(cf_dst + CF_GUARD + dof, pattern, n);
                TEST_ASSERT_EQUAL_INT8_ARRAY(cf_ref, cf_dst, sizeof(cf_dst_w));
            }
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
                const uint32_t count = cf_sizes[k] / 2;
                cf_reset();
                for (uint32_t i = 0; i < count; i++)
                {
                    memcpy(cf_ref + CF_GUARD + dof + 2 * i, &vals[v], 2);
                }
                arm_memset_s16((int16_t *)(void *)(cf_dst + CF_GUARD + dof), vals[v], count);
                TEST_ASSERT_EQUAL_INT8_ARRAY(cf_ref, cf_dst, sizeof(cf_dst_w));
            }
        }
    }
}

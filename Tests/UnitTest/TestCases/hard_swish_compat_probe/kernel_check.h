// SPDX-FileCopyrightText: 2026 Ambiq
// SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
#ifndef HARD_SWISH_COMPAT_KERNEL_CHECK_H
#define HARD_SWISH_COMPAT_KERNEL_CHECK_H

#include "arm_nnfunctions.h"
#include "goldens.h"
#include <stdio.h>
#include <string.h>

// Shared by the native sanitizer probe and the existing target Unity suite.
static int hs_check_litert(void)
{
    static int8_t input[4161], output[4161 + 16];
    const int sizes[] = {1, 7, 8, 9, 256, 4159, 4160, 4161};
    for (int i = 0; i < 4161; ++i)
        input[i] = (int8_t)((i % 256) - 128);
    for (unsigned c = 0; c < sizeof(hs_litert_cases) / sizeof(hs_litert_cases[0]); ++c)
    {
        const hs_litert_case *p = &hs_litert_cases[c];
        for (unsigned s = 0; s < sizeof(sizes) / sizeof(sizes[0]); ++s)
        {
            const int n = sizes[s];
            memset(output, 0x5a, sizeof(output));
            if (arm_hard_swish_compat_s8(input, p->zi, p->zo, p->ofp, p->oe, p->rfp, p->re, output, n) !=
                ARM_CMSIS_NN_SUCCESS)
                return 1;
            for (int i = 0; i < n; ++i)
                if (output[i] != p->expected[i % 256])
                {
                    printf("Hard Swish case %u size %d index %d: %d != %d\n", c, n, i, output[i], p->expected[i % 256]);
                    return 1;
                }
            for (int i = n; i < n + 16; ++i)
                if (output[i] != 0x5a)
                    return 1;
        }
    }
    return 0;
}
#endif

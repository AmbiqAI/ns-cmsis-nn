/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 */

#include "arm_nnfunctions.h"
#define ARM_NNSUPPORTFUNCTIONS_H /* Use the constexpr copy below instead of runtime helpers. */

/* Compile the production C bodies as constant expressions. Only their names,
 * constexpr declarations and copy primitive are adapted; pointer math is not. */
constexpr void copy_s8(int8_t *out, const int8_t *in, uint32_t count)
{
    for (uint32_t i = 0; i < count; ++i)
        out[i] = in[i];
}
#define arm_memcpy_s8 copy_s8
#define arm_concatenation_s8_x constexpr probe_s8_x
#define arm_concatenation_s8_y constexpr probe_s8_y
#define arm_concatenation_s8_z constexpr probe_s8_z
#include "arm_concatenation_s8_x.c"
#include "arm_concatenation_s8_y.c"
#include "arm_concatenation_s8_z.c"
#undef arm_concatenation_s8_x
#undef arm_concatenation_s8_y
#undef arm_concatenation_s8_z
#undef arm_memcpy_s8

template <int Axis, int Rows> constexpr bool copies()
{
    int8_t input[2 * Rows] = {};
    int8_t output[4 * Rows] = {};
    for (int i = 0; i < 2 * Rows; ++i)
        input[i] = i + 1;
    for (uint32_t offset = 0; offset < 4; offset += 2)
    {
        if (Axis == 0)
            probe_s8_x(input, 2, 1, 1, Rows, output, 4, offset);
        else if (Axis == 1)
            probe_s8_y(input, 1, 2, 1, Rows, output, 4, offset);
        else
            probe_s8_z(input, 1, 1, 2, Rows, output, 4, offset);
    }
    for (int row = 0; row < Rows; ++row)
    {
        for (int j = 0; j < 4; ++j)
        {
            if (output[4 * row + j] != input[2 * row + j % 2])
                return false;
        }
    }
    return true;
}

static_assert(copies<0, 1>() && copies<0, 3>(), "s8 x copies stay within the output object");
static_assert(copies<1, 1>() && copies<1, 3>(), "s8 y copies stay within the output object");
static_assert(copies<2, 1>() && copies<2, 3>(), "s8 z copies stay within the output object");

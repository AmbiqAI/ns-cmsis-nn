/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 */

#include "../../../../Include/Internal/arm_concatenation_common.h"

/* Clang constant evaluation diagnoses even an unused pointer beyond one-past;
 * ordinary memory sanitizers only observe accesses. Instantiate the production
 * loop unchanged, with a constexpr copy and constexpr function declarations. */
constexpr void arm_memcpy_probe(int *out, const int *in, uint32_t count)
{
    for (uint32_t i = 0; i < count; ++i)
    {
        out[i] = in[i];
    }
}
#define arm_concatenation_probe_x constexpr arm_concatenation_probe_x
#define arm_concatenation_probe_y constexpr arm_concatenation_probe_y
#define arm_concatenation_probe_z constexpr arm_concatenation_probe_z
#define arm_concatenation_probe_w constexpr arm_concatenation_probe_w
ARM_CONCATENATION_DEFINE(probe, int)
#undef arm_concatenation_probe_x
#undef arm_concatenation_probe_y
#undef arm_concatenation_probe_z
#undef arm_concatenation_probe_w

template <int Axis, int Rows> constexpr bool copies()
{
    int input[2 * Rows] = {};
    int output[4 * Rows] = {};
    for (int i = 0; i < 2 * Rows; ++i)
    {
        input[i] = i + 1;
    }
    for (uint32_t offset = 0; offset < 4; offset += 2)
    {
        if (Axis == 0)
            arm_concatenation_probe_x(input, 2, 1, 1, Rows, output, 4, offset);
        else if (Axis == 1)
            arm_concatenation_probe_y(input, 1, 2, 1, Rows, output, 4, offset);
        else
            arm_concatenation_probe_z(input, 1, 1, 2, Rows, output, 4, offset);
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

constexpr bool sibling_and_empty()
{
    const int input[2] = {1, 2};
    int output[4] = {};
    arm_concatenation_probe_w(input, 1, 1, 1, 2, output, 0);
    arm_concatenation_probe_w(input, 1, 1, 1, 2, output, 2);
    arm_concatenation_probe_x(input, 2, 0, 1, 1, output, 4, 0);
    arm_concatenation_probe_y(input, 1, 2, 0, 1, output, 4, 0);
    arm_concatenation_probe_z(input, 1, 1, 2, 0, output, 4, 0);
    arm_concatenation_probe_w(input, 1, 1, 1, 0, output, 0);
    return output[0] == 1 && output[1] == 2 && output[2] == 1 && output[3] == 2;
}

static_assert(copies<0, 1>() && copies<0, 3>(), "x copies form only in-bounds pointers");
static_assert(copies<1, 1>() && copies<1, 3>(), "y copies form only in-bounds pointers");
static_assert(copies<2, 1>() && copies<2, 3>(), "z copies form only in-bounds pointers");
static_assert(sibling_and_empty(), "w and zero-iteration behavior is preserved");

// SPDX-FileCopyrightText: 2026 Ambiq
// SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
#include "arm_nnsupportfunctions.h"
#include "kernel_check.h"
#include <limits.h>
#include <string.h>

// Deliberately independent oracle: repeated doubling stops after saturation.
static int16_t reference_shift(int16_t x, int shift)
{
    int64_t value = x;
    while (shift > 0 && value != 0 && value >= INT16_MIN && value <= INT16_MAX)
    {
        value *= 2;
        --shift;
    }
    if (value > INT16_MAX)
        return INT16_MAX;
    if (value < INT16_MIN)
        return INT16_MIN;
    return (int16_t)value;
}

static int16_t reference_multiply(int16_t a, int16_t b)
{
    // Mathematical floor division, including negative rounding ties.
    const int64_t numerator = (int64_t)a * b + 16384;
    int64_t result = numerator / 32768;
    if (numerator < 0 && numerator % 32768 != 0)
        --result;
    if (result > INT16_MAX)
        return INT16_MAX;
    if (result < INT16_MIN)
        return INT16_MIN;
    return (int16_t)result;
}

static int check_shift(void)
{
    const int extra_shifts[] = {INT_MIN, -1, 31, 32, INT_MAX};
    for (int x = INT16_MIN; x <= INT16_MAX; ++x)
    {
        for (int shift = 0; shift <= 16; ++shift)
            if (arm_nn_sat_lshift_s16((int16_t)x, shift) != reference_shift((int16_t)x, shift))
                return 1;
        for (unsigned i = 0; i < sizeof(extra_shifts) / sizeof(extra_shifts[0]); ++i)
            if (arm_nn_sat_lshift_s16((int16_t)x, extra_shifts[i]) != reference_shift((int16_t)x, extra_shifts[i]))
                return 1;
    }
    return 0;
}

static int check_multiply(void)
{
    const int16_t multipliers[] = {
        INT16_MIN, INT16_MIN + 1, -16385, -16384, -16383, -1, 0, 1, 16383, 16384, 16385, INT16_MAX - 1, INT16_MAX};
    if (arm_nn_sqrdmulh_s16(-1, 16384) != 0 || arm_nn_sqrdmulh_s16(-3, 16384) != -1)
        return 1;
    for (int a = INT16_MIN; a <= INT16_MAX; ++a)
        for (unsigned b = 0; b < sizeof(multipliers) / sizeof(multipliers[0]); ++b)
            if (arm_nn_sqrdmulh_s16((int16_t)a, multipliers[b]) != reference_multiply((int16_t)a, multipliers[b]))
                return 1;
    return 0;
}

int main(int argc, char **argv)
{
    // Separate modes let regression checks demonstrate each original failure.
    if (argc == 2 && strcmp(argv[1], "shift") == 0)
        return check_shift();
    if (argc == 2 && strcmp(argv[1], "multiply") == 0)
        return check_multiply();
    if (check_shift() || check_multiply() || hs_check_litert())
        return 1;
    puts("Hard Swish PASS: two helpers and eight LiteRT cases at eight sizes");
    return 0;
}

// SPDX-FileCopyrightText: Copyright 2026 Ambiq
// SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
#pragma once
#include <stdint.h>

// Arithmetic model for constant-evaluated pointer probes; target tests exercise
// the production helper. C++20 bit_cast models its runtime memcpy conversion.
constexpr int32_t arm_nn_mac_wrap_s32_bits(int32_t acc, int32_t a, int32_t b)
{
    const uint32_t bits = (uint32_t)acc + (uint32_t)a * (uint32_t)b;
    return __builtin_bit_cast(int32_t, bits);
}
static_assert(arm_nn_mac_wrap_s32_bits(INT32_MAX, 1, 1) == INT32_MIN);
static_assert(arm_nn_mac_wrap_s32_bits(INT32_MIN, -1, 1) == INT32_MAX);
static_assert(arm_nn_mac_wrap_s32_bits(0, INT32_MIN, -1) == INT32_MIN);

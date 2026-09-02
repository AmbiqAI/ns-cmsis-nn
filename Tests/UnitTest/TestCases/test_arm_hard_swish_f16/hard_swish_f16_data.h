/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq
 * SPDX-License-Identifier: Apache-2.0
 *
 * Generated golden data (numpy reference); see the PR for the generator model:
 * gate = RN32(fma(x, RN32(1/6), 0.5)) clamped to [0, 1], product rounded once.
 */
#pragma once

#define HARD_SWISH_F16_DST_SIZE 27

static const float16_t hard_swish_f16_input[] = {
    0.457519531f, 2.07617188f, 0.584472656f, -2.39453125f, -2.69921875f, -0.197998047f, -1.92285156f,
    0.498291016f, 2.54296875f, -0.110900879f, 1.95507812f, -0.450439453f, -0.876464844f, 0.00643539429f,
    -1.79101562f, -2.43164062f, 2.53320312f, 1.9609375f, -2.44140625f, -0.228637695f, -1.81445312f,
    -1.6484375f, 1.78320312f, 1.86914062f, -0.420410156f, -1.828125f, 1.53320312f};
static const float16_t hard_swish_f16_output_ref[] = {
    0.263671875f, 1.75683594f, 0.349121094f, -0.241577148f, -0.135253906f, -0.0924682617f, -0.345214844f,
    0.290527344f, 2.34960938f, -0.0534057617f, 1.61425781f, -0.19140625f, -0.310302734f, 0.00322532654f,
    -0.360839844f, -0.23034668f, 2.3359375f, 1.62109375f, -0.227294922f, -0.10559082f, -0.358398438f,
    -0.371337891f, 1.421875f, 1.51660156f, -0.180786133f, -0.356933594f, 1.15820312f};

#define HARD_SWISH_F16_IDENT_SIZE 6
static const float16_t hard_swish_f16_ident[] = {
    3.0f, 3.00195312f, 3.5f, 6.0f, 100.0f, 65504.0f};
static const float16_t hard_swish_f16_zero_region[] = {
    -3.0f, -3.00195312f, -3.5f, -6.0f, -100.0f, -65504.0f};

#define HARD_SWISH_F16_DENORMAL_SIZE 4
static const uint16_t hard_swish_f16_denormal_in_bits[] = {
    0x0001, 0x8001, 0x03ef, 0x0002};
static const uint16_t hard_swish_f16_denormal_ref_bits[] = {
    0x0000, 0x8000, 0x01f8, 0x0001};

/* Round-once witness: computing the gate and product natively in float16
 * (gate rounded to f16, product in f16) yields 0x3065 here; the kernel's
 * float32 computation with a single final rounding yields the correctly
 * rounded 0x3066. Input x = 0.25341796875. */
#define HARD_SWISH_F16_WITNESS_IN_BITS ((uint16_t)0x340e)
#define HARD_SWISH_F16_WITNESS_ONCE_BITS ((uint16_t)0x3066)
#define HARD_SWISH_F16_WITNESS_TWICE_BITS ((uint16_t)0x3065)

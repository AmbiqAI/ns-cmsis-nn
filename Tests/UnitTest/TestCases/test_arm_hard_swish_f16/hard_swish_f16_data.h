/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq
 * SPDX-License-Identifier: Apache-2.0
 *
 * Generated golden data (numpy reference); see the PR for the generator model:
 * gate = RN32(fma(x, RN32(1/6), 0.5)) clamped to [0, 1], product rounded once.
 * That model is the kernel's scalar leg. The first 27 curved-region entries are
 * random; the last four are the worst case of the pure-float16 MVE leg against
 * it over an exhaustive float16 sweep -- 2 ulp, 1.953e-3 absolute, 70.2% of the
 * combined band the test applies. See AmbiqAI/ns-cmsis-nn#427.
 */
#pragma once

#define HARD_SWISH_F16_DST_SIZE 31

static const float16_t hard_swish_f16_input[] = {
    0.457519531f, 2.07617188f, 0.584472656f, -2.39453125f, -2.69921875f, -0.197998047f, -1.92285156f,
    0.498291016f, 2.54296875f, -0.110900879f, 1.95507812f, -0.450439453f, -0.876464844f, 0.00643539429f,
    -1.79101562f, -2.43164062f, 2.53320312f, 1.9609375f, -2.44140625f, -0.228637695f, -1.81445312f,
    -1.6484375f, 1.78320312f, 1.86914062f, -0.420410156f, -1.828125f, 1.53320312f,
    /* 0x4031, 0x4055, 0x406d, 0x4079: the sweep's worst-case band. */
    2.09570312f, 2.16601562f, 2.21289062f, 2.23632812f};
static const float16_t hard_swish_f16_output_ref[] = {
    0.263671875f, 1.75683594f, 0.349121094f, -0.241577148f, -0.135253906f, -0.0924682617f, -0.345214844f,
    0.290527344f, 2.34960938f, -0.0534057617f, 1.61425781f, -0.19140625f, -0.310302734f, 0.00322532654f,
    -0.360839844f, -0.23034668f, 2.3359375f, 1.62109375f, -0.227294922f, -0.10559082f, -0.358398438f,
    -0.371337891f, 1.421875f, 1.51660156f, -0.180786133f, -0.356933594f, 1.15820312f,
    /* 0x3f1f, 0x3f76, 0x3fb1, 0x3fcf; the MVE leg lands 2 ulp below each. */
    1.78027344f, 1.86523438f, 1.92285156f, 1.95214844f};

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

/* Rounding-shape witness, x = 0.25341796875: the kernel's scalar leg computes
 * in float32 and rounds once, giving the correctly rounded 0x3066; its MVE leg
 * rounds the gate and the product separately in float16, giving 0x3065. One
 * ulp apart, which is what makes this input worth pinning. */
#define HARD_SWISH_F16_WITNESS_IN_BITS ((uint16_t)0x340e)
#define HARD_SWISH_F16_WITNESS_ONCE_BITS ((uint16_t)0x3066)
#define HARD_SWISH_F16_WITNESS_TWICE_BITS ((uint16_t)0x3065)

/* Fused-gate witness for the scalar leg: the single float16 input (exhaustive
 * sweep) where the correctly rounded fma gate differs observably from a
 * separately rounded x * (1/6f) + 0.5f gate after the final narrowing. Pinned
 * bit-exactly so an unfused-gate regression fails. The MVE leg's float16
 * evaluation lands on the same bits. */
#define HARD_SWISH_F16_FMA_WITNESS_IN_BITS ((uint16_t)0x3bff) /* 0.99951171875 */
#define HARD_SWISH_F16_FMA_WITNESS_FUSED_BITS ((uint16_t)0x3954)
#define HARD_SWISH_F16_FMA_WITNESS_UNFUSED_BITS ((uint16_t)0x3955)

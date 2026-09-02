/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq
 * SPDX-License-Identifier: Apache-2.0
 *
 * Generated golden data (numpy reference); see the PR for the generator model:
 * gate = RN32(fma(x, RN32(1/6), 0.5)) clamped to [0, 1], product rounded once.
 */
#pragma once

#define HARD_SWISH_F32_DST_SIZE 35

static const float32_t hard_swish_f32_input[] = {
    -0.307414472f, 2.65381289f, 2.48168707f, -0.330208659f, -0.603770852f, 1.47821856f, 2.24964523f,
    -0.528894782f, 1.91887379f, 1.11720371f, -2.17681885f, 1.44844449f, -3.22498178f, -2.63496137f,
    -2.69738269f, 0.670777202f, -2.60523152f, -1.45385158f, 2.68966913f, 1.68214023f, 1.45902181f,
    2.99345136f, -1.3939929f, -0.716884196f, 3.45700026f, 2.53866577f, 1.35133398f, 1.17074096f,
    2.10001421f, 3.0938971f, 2.65369248f, -2.3831768f, -2.13284588f, 3.44936824f, 2.62316155f};
static const float32_t hard_swish_f32_output_ref[] = {
    -0.137956634f, 2.50069356f, 2.26730537f, -0.146931365f, -0.241128892f, 1.10329771f, 1.96830654f,
    -0.21782577f, 1.5731163f, 0.766625881f, -0.298652709f, 1.07388747f, -0.0f, -0.160310417f,
    -0.136045754f, 0.410378933f, -0.171410516f, -0.374645025f, 2.55055451f, 1.3126694f, 1.08430171f,
    2.99018431f, -0.373127043f, -0.272788286f, 3.45700026f, 2.3434701f, 0.980017602f, 0.813809574f,
    1.78501713f, 3.0938971f, 2.50052691f, -0.244999766f, -0.308250993f, 3.44936824f, 2.45841026f};

#define HARD_SWISH_F32_IDENT_SIZE 7
static const float32_t hard_swish_f32_ident[] = {
    3.0f, 3.00000024f, 3.5f, 6.0f, 100.0f, 1.00000002e+30f, 3.40282347e+38f};
static const float32_t hard_swish_f32_zero_region[] = {
    -3.0f, -3.00000024f, -3.5f, -6.0f, -100.0f, -1.00000002e+30f, -3.40282347e+38f};

#define HARD_SWISH_F32_DENORMAL_SIZE 5
static const float32_t hard_swish_f32_denormal_in[] = {
    9.80908925e-45f, -9.80908925e-45f, 1.40129846e-45f, -1.40129846e-45f, 1.17549421e-38f};
static const uint32_t hard_swish_f32_denormal_ref_bits[] = {
    0x00000004, 0x80000004, 0x00000000, 0x80000000, 0x00400000};

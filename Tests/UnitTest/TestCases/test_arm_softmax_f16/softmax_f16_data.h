/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq
 * SPDX-License-Identifier: Apache-2.0
 *
 * Golden data for arm_softmax_f16, written by hand rather than emitted by a
 * generator: softmax_settings_flt.py's softmax_f16 dataset writes the
 * TestCases/TestData layout, which this suite does not use.
 *
 * The reference is softmax evaluated in float32 over the float16 inputs and
 * rounded once to float16 -- the same definition softmax_settings_flt.py
 * applies -- and is bit-exact for every element:
 *
 *   import numpy as np
 *   x = np.array(INPUT, dtype=np.float16).reshape(ROWS, COLS).astype(np.float32)
 *   e = np.exp(x - x.max(1, keepdims=True))
 *   (e / e.sum(1, keepdims=True)).astype(np.float16)
 *
 * Three row widths, because the MVE leg is an 8-lane loop and the row width
 * decides which arms of it run: 5 is predicated on the only pass, 16 is two
 * full unpredicated passes and no tail, 20 is two full passes and a 4-lane
 * tail. Only the full-vector passes exercise the half<->single conversions
 * with every lane live (AmbiqAI/ns-cmsis-nn#427).
 */
#pragma once

#define SOFTMAX_F16_NUM_ROWS 2
#define SOFTMAX_F16_ROW_SIZE 5
#define SOFTMAX_F16_DST_SIZE 10

static const float16_t softmax_f16_input[] = {1.0f, 0.25f, -0.5f, 2.0f, -1.25f, -2.0f, -0.125f, 0.75f, 3.0f, 1.5f};

static const float16_t softmax_f16_output_ref[] = {0.221313477f,
                                                   0.104553223f,
                                                   0.0493774414f,
                                                   0.6015625f,
                                                   0.0233154297f,
                                                   0.0048866272f,
                                                   0.0318603516f,
                                                   0.0764160156f,
                                                   0.725097656f,
                                                   0.161743164f};

#define SOFTMAX_F16_VEC_NUM_ROWS 2
#define SOFTMAX_F16_VEC_ROW_SIZE 16
#define SOFTMAX_F16_VEC_DST_SIZE 32

static const float16_t softmax_f16_vec_input[] = {
    -4.0f, -3.5f, -3.0f, -2.5f, -2.0f, -1.5f, -1.0f, -0.5f, 0.0f,   0.5f,  1.0f,   1.5f,  2.0f,   2.5f,  3.0f,   3.5f,
    1.75f, 1.5f,  1.25f, 1.0f,  0.75f, 0.5f,  0.25f, 0.0f,  -0.25f, -0.5f, -0.75f, -1.0f, -1.25f, -1.5f, -1.75f, -2.0f};

static const float16_t softmax_f16_vec_output_ref[] = {
    0.000217676163f, 0.000358819962f, 0.000591754913f, 0.000975608826f, 0.00160884857f, 0.0026512146f, 0.00437164307f,
    0.00720977783f,  0.0118865967f,   0.0195922852f,   0.0323181152f,   0.0532531738f,  0.0878295898f, 0.144775391f,
    0.238769531f,    0.393554688f,    0.225341797f,    0.175537109f,    0.13671875f,    0.106445312f,  0.0828857422f,
    0.0645751953f,   0.0502624512f,   0.0391540527f,   0.0304870605f,   0.0237426758f,  0.0184936523f, 0.0144042969f,
    0.01121521f,     0.00873565674f,  0.00680541992f,  0.0052986145f};

#define SOFTMAX_F16_WIDE_NUM_ROWS 2
#define SOFTMAX_F16_WIDE_ROW_SIZE 20
#define SOFTMAX_F16_WIDE_DST_SIZE 40

static const float16_t softmax_f16_wide_input[] = {
    -2.5f, -2.25f, -2.0f, -1.75f, -1.5f, -1.25f, -1.0f, -0.75f, -0.5f, -0.25f, 0.0f,  0.25f, 0.5f, 0.75f,
    1.0f,  1.25f,  1.5f,  1.75f,  2.0f,  2.25f,  3.0f,  2.75f,  2.5f,  2.25f,  2.0f,  1.75f, 1.5f, 1.25f,
    1.0f,  0.75f,  0.5f,  0.25f,  0.0f,  -0.25f, -0.5f, -0.75f, -1.0f, -1.25f, -1.5f, -1.75f};

static const float16_t softmax_f16_wide_output_ref[] = {
    0.00192642212f, 0.00247383118f, 0.00317573547f, 0.00407791138f, 0.00523757935f, 0.00672531128f, 0.00863647461f,
    0.0110855103f,  0.0142364502f,  0.0182800293f,  0.0234680176f,  0.0301361084f,  0.0386962891f,  0.0496826172f,
    0.0637817383f,  0.0819091797f,  0.105224609f,   0.135131836f,   0.173461914f,   0.22265625f,    0.22265625f,
    0.173461914f,   0.135131836f,   0.105224609f,   0.0819091797f,  0.0637817383f,  0.0496826172f,  0.0386962891f,
    0.0301361084f,  0.0234680176f,  0.0182800293f,  0.0142364502f,  0.0110855103f,  0.00863647461f, 0.00672531128f,
    0.00523757935f, 0.00407791138f, 0.00317573547f, 0.00247383118f, 0.00192642212f};

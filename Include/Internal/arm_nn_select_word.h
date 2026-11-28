/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 */

#ifndef ARM_NN_SELECT_WORD_H
#define ARM_NN_SELECT_WORD_H

#include "arm_nn_types.h"
#include <stdbool.h>

/* Rank and stride validation and bit-exact selection for 32-bit payloads. */
arm_cmsis_nn_status arm_nn_select_word(const bool *condition,
                                       const unsigned char *x,
                                       const unsigned char *y,
                                       const cmsis_nn_select_v2_params *params,
                                       unsigned char *output);

#endif

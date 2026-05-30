// Hand-crafted test data for arm_reverse_sequence_s16
#pragma once
#include <stdint.h>

// Input: shape [2, 4] (batch_dim=0, seq_dim=1)
// seq_lengths: [3, 2]
// Input:  [[100, 200, 300, 400], [500, 600, 700, 800]]
// Batch 0: reverse first 3 -> [300, 200, 100, 400]
// Batch 1: reverse first 2 -> [600, 500, 700, 800]
// Output: [[300, 200, 100, 400], [600, 500, 700, 800]]

#define REVERSE_SEQUENCE_2D_S16_RANK 2
#define REVERSE_SEQUENCE_2D_S16_SIZE 8
#define REVERSE_SEQUENCE_2D_S16_SEQ_DIM 1
#define REVERSE_SEQUENCE_2D_S16_BATCH_DIM 0

static const int32_t reverse_sequence_2d_s16_shape[2] = {2, 4};
static const int32_t reverse_sequence_2d_s16_seq_lengths[2] = {3, 2};

static const int16_t reverse_sequence_2d_s16_input[8] = {100, 200, 300, 400, 500, 600, 700, 800};
static const int16_t reverse_sequence_2d_s16_output[8] = {300, 200, 100, 400, 600, 500, 700, 800};

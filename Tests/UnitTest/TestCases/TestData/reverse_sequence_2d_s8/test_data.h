// Hand-crafted test data for arm_reverse_sequence_s8
#pragma once
#include <stdint.h>

// Input: shape [2, 4] (batch_dim=0, seq_dim=1)
// seq_lengths: [3, 2]
// Input:  [[1, 2, 3, 4], [5, 6, 7, 8]]
// Batch 0: reverse first 3 → [3, 2, 1, 4]
// Batch 1: reverse first 2 → [6, 5, 7, 8]
// Output: [[3, 2, 1, 4], [6, 5, 7, 8]]

#define REVERSE_SEQUENCE_2D_S8_RANK 2
#define REVERSE_SEQUENCE_2D_S8_SIZE 8
#define REVERSE_SEQUENCE_2D_S8_SEQ_DIM 1
#define REVERSE_SEQUENCE_2D_S8_BATCH_DIM 0

static const int32_t reverse_sequence_2d_s8_shape[2] = {2, 4};
static const int32_t reverse_sequence_2d_s8_seq_lengths[2] = {3, 2};

static const int8_t reverse_sequence_2d_s8_input[8] = {1, 2, 3, 4, 5, 6, 7, 8};
static const int8_t reverse_sequence_2d_s8_output[8] = {3, 2, 1, 4, 6, 5, 7, 8};

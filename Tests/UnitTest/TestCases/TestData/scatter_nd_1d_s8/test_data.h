// Hand-crafted test data for arm_scatter_nd_s8
#pragma once
#include <stdint.h>

// Output shape: [5], indices: [[1], [3]], updates: [10, 20]
// Expected output: [0, 10, 0, 20, 0]

#define SCATTER_ND_1D_S8_NUM_UPDATES 2
#define SCATTER_ND_1D_S8_INDEX_DEPTH 1
#define SCATTER_ND_1D_S8_SLICE_SIZE 1
#define SCATTER_ND_1D_S8_OUTPUT_SIZE 5

static const int32_t scatter_nd_1d_s8_output_strides[1] = {1};
static const int32_t scatter_nd_1d_s8_indices[2] = {1, 3};
static const int8_t scatter_nd_1d_s8_updates[2] = {10, 20};
static const int8_t scatter_nd_1d_s8_output[5] = {0, 10, 0, 20, 0};

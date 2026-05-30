// Hand-crafted test data for arm_scatter_nd_s16
#pragma once
#include <stdint.h>

// Output shape: [5], indices: [[1], [3]], updates: [100, 200]
// Expected output: [0, 100, 0, 200, 0]

#define SCATTER_ND_1D_S16_NUM_UPDATES 2
#define SCATTER_ND_1D_S16_INDEX_DEPTH 1
#define SCATTER_ND_1D_S16_SLICE_SIZE 1
#define SCATTER_ND_1D_S16_OUTPUT_SIZE 5

static const int32_t scatter_nd_1d_s16_output_strides[1] = {1};
static const int32_t scatter_nd_1d_s16_indices[2] = {1, 3};
static const int16_t scatter_nd_1d_s16_updates[2] = {100, 200};
static const int16_t scatter_nd_1d_s16_output[5] = {0, 100, 0, 200, 0};

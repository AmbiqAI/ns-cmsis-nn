# INT32 comparison and selection

The six direct `arm_equal_s32`, `arm_not_equal_s32`, `arm_less_s32`, `arm_less_equal_s32`, `arm_greater_s32` and `arm_greater_equal_s32` entries compare full-width signed integers without quantization. Outputs are one-byte Boolean values. They use positive NHWC singleton broadcasting; lower ranks are padded with leading ones. The broadcast output count is limited to INT32_MAX / 4 elements. Invalid metadata returns an error before writes.

`arm_select_v2_s32` copies either INT32 input according to a Boolean condition. Its existing `cmsis_nn_select_v2_params` representation supports rank 1 through 8, positive output dimensions, and nonnegative element strides (zero means broadcasting). Represent a scalar output as shape `[1]`. Output byte count and the maximum reachable byte span of every input must fit INT32_MAX; the caller remains responsible for allocating those buffers. Output must not overlap inputs or metadata.

Both families require no scratch or floating-point support. Generic and DSP targets use exact scalar operations; MVE builds use predicated loads/stores with no scalar tail unless compiler autovectorization is selected. Comparisons preserve operand order for either scalar input. Selection processes the longest suffix shared as contiguous or scalar by all three inputs and advances outer coordinates without per-element division; arbitrary valid strides retain a scalar-sized span.

These are unquantized INT32 entries. Existing INT8/INT16 quantized comparisons and selection retain their contracts.

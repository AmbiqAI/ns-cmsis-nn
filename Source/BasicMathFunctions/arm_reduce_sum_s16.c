#include "sum.h"

static inline int32_t math_mult_by_quant_mult(int64_t x, int32_t mult, int shift) {
    int left_shift = shift > 0 ? shift : 0;
    int right_shift = shift > 0 ? 0 : -shift;
    int64_t x_shifted = x * (1LL << left_shift);
    int64_t total = x_shifted * (int64_t)mult;
    int64_t result = (total + (1LL << 30)) >> 31;
    if (right_shift > 0) {
        int64_t rounding_offset = 1LL << (right_shift - 1);
        result = (result + rounding_offset) >> right_shift;
    }
    return (int32_t)result;
}

arm_cmsis_nn_status arm_reduce_sum_s16(
    const int16_t *input_data,
    const int32_t input_zp,
    const int* dims,
    const int num_dims,
    const int* axes,
    const int num_axes,
    int16_t *output,
    const int32_t out_offset,
    const int32_t out_mult,
    const int32_t out_shift,
    const int32_t out_activation_min,
    const int32_t out_activation_max)
{
    if (!input_data || !output || !dims || !axes) return ARM_CMSIS_NN_ARG_ERROR;

    int d4[4] = {1, 1, 1, 1};
    int r4[4] = {0, 0, 0, 0};
    int offset = 4 - num_dims;
    for (int i = 0; i < num_dims; i++) d4[offset + i] = dims[i];
    for (int i = 0; i < num_axes; i++) {
        int ax = axes[i] < 0 ? axes[i] + num_dims : axes[i];
        r4[offset + ax] = 1;
    }

    int s2 = d4[3];
    int s1 = d4[2] * d4[3];
    int s0 = d4[1] * d4[2] * d4[3];

    int out_dims[4];
    for (int i = 0; i < 4; i++) out_dims[i] = r4[i] ? 1 : d4[i];

    int out_idx = 0;
    for (int o0 = 0; o0 < out_dims[0]; o0++) {
        int st0 = r4[0] ? 0 : o0, en0 = r4[0] ? d4[0] : o0 + 1;
        for (int o1 = 0; o1 < out_dims[1]; o1++) {
            int st1 = r4[1] ? 0 : o1, en1 = r4[1] ? d4[1] : o1 + 1;
            for (int o2 = 0; o2 < out_dims[2]; o2++) {
                int st2 = r4[2] ? 0 : o2, en2 = r4[2] ? d4[2] : o2 + 1;
                for (int o3 = 0; o3 < out_dims[3]; o3++) {
                    int st3 = r4[3] ? 0 : o3, en3 = r4[3] ? d4[3] : o3 + 1;

                    int64_t acc = 0;
                    for (int i0 = st0; i0 < en0; i0++) {
                        int off0 = i0 * s0;
                        for (int i1 = st1; i1 < en1; i1++) {
                            int off1 = off0 + i1 * s1;
                            for (int i2 = st2; i2 < en2; i2++) {
                                int off2 = off1 + i2 * s2;
                                const int16_t* ptr = &input_data[off2 + st3];
                                for (int i3 = st3; i3 < en3; i3++) {
                                    acc += (int64_t)(*ptr++) - input_zp;
                                }
                            }
                        }
                    }

                    int32_t res = math_mult_by_quant_mult(acc, out_mult, out_shift);
                    res += out_offset;
                    if (res > out_activation_max) res = out_activation_max;
                    if (res < out_activation_min) res = out_activation_min;
                    output[out_idx++] = (int16_t)res;
                }
            }
        }
    }
    return ARM_CMSIS_NN_SUCCESS;
}
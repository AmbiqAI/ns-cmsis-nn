/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_fully_connected_per_channel_packed_s8.c
 * Description:  Direct-entry s8 per-channel fully connected layer on a weight stream packed ahead of time.
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

/**
 *  @ingroup Public
 */

/**
 * @addtogroup FC
 * @{
 */

/* Rows per block, and bytes of epilogue parameters per block: four kernel sums, four multipliers, four shifts */
#define FC_PACKED_ROWS (4)
#define FC_PACKED_PARAM_BYTES (3 * FC_PACKED_ROWS * (int32_t)sizeof(int32_t))

/* The accumulation depth rounded up to whole 16-byte groups */
static int32_t arm_fc_packed_depth(const int32_t k) { return (k + 15) & ~15; }

int32_t arm_fully_connected_per_channel_packed_s8_get_packed_size(const cmsis_nn_dims *filter_dims)
{
    if (filter_dims == NULL || filter_dims->n <= 0 || filter_dims->c <= 0 || filter_dims->n > INT32_MAX - 15)
    {
        return 0;
    }
    const int64_t blocks = ((int64_t)filter_dims->c + FC_PACKED_ROWS - 1) / FC_PACKED_ROWS;
    const int64_t size =
        blocks * ((int64_t)FC_PACKED_ROWS * arm_fc_packed_depth(filter_dims->n) + FC_PACKED_PARAM_BYTES);
    return size > INT32_MAX ? 0 : (int32_t)size;
}

arm_cmsis_nn_status
arm_fully_connected_per_channel_packed_s8_pack(const cmsis_nn_dims *filter_dims,
                                               const int8_t *filter_data,
                                               const int32_t *kernel_sum,
                                               const cmsis_nn_per_channel_quant_params *quant_params,
                                               int8_t *packed_data)
{
    if (arm_fully_connected_per_channel_packed_s8_get_packed_size(filter_dims) == 0 || filter_data == NULL ||
        kernel_sum == NULL || quant_params == NULL || quant_params->multiplier == NULL || quant_params->shift == NULL ||
        packed_data == NULL)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    const int32_t k = filter_dims->n;
    const int32_t n = filter_dims->c;
    if (!arm_nn_fc_packed_s8_rshift_only(quant_params, n))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    const int32_t kp = arm_fc_packed_depth(k);

    int8_t *o = packed_data;
    for (int32_t row0 = 0; row0 < n; row0 += FC_PACKED_ROWS)
    {
        const int32_t live = ARM_NN_MIN(FC_PACKED_ROWS, n - row0);
        for (int32_t g = 0; g < kp; g += 16)
        {
            for (int32_t r = 0; r < FC_PACKED_ROWS; r++)
            {
                for (int32_t j = 0; j < 16; j++)
                {
                    *o++ = (r < live && g + j < k) ? filter_data[(size_t)(row0 + r) * (size_t)k + (size_t)(g + j)] : 0;
                }
            }
        }
        const int32_t *const params[3] = {kernel_sum, quant_params->multiplier, quant_params->shift};
        for (int32_t p = 0; p < 3; p++)
        {
            for (int32_t r = 0; r < FC_PACKED_ROWS; r++)
            {
                const int32_t v = r < live ? params[p][row0 + r] : 0;
                arm_memcpy_s8(o, (const int8_t *)&v, (uint32_t)sizeof(v));
                o += sizeof(v);
            }
        }
    }
    return ARM_CMSIS_NN_SUCCESS;
}

#if defined(ARM_MATH_MVEI) && !defined(ARM_MATH_AUTOVECTORIZE) && !defined(CMSIS_NN_USE_SINGLE_ROUNDING)
/*
 * One block of four output channels: the dot products over the interleaved weight groups, then the block's
 * parameters, which follow its weights in the stream. Returns the stream position after the block.
 */
__STATIC_FORCEINLINE const int8_t *arm_fc_packed_s8_block(const int8_t *input,
                                                          const int8_t *w,
                                                          const int32_t k,
                                                          const int32_t live,
                                                          const int32x4_t v_off,
                                                          const int32x4_t v_min,
                                                          const int32x4_t v_max,
                                                          int8_t *dst)
{
    int32_t a0 = 0;
    int32_t a1 = 0;
    int32_t a2 = 0;
    int32_t a3 = 0;
    const int8_t *ip = input;
    __ASM volatile(" .p2align 2                         \n"
                   "   wlstp.8     lr, %[cnt], 1f       \n"
                   "2:                                  \n"
                   "   vldrb.8     q0, [%[ip]], #16     \n"
                   "   vldrb.8     q1, [%[w]], #16      \n"
                   "   vmladava.s8 %[a0], q0, q1        \n"
                   "   vldrb.8     q2, [%[w]], #16      \n"
                   "   vmladava.s8 %[a1], q0, q2        \n"
                   "   vldrb.8     q3, [%[w]], #16      \n"
                   "   vmladava.s8 %[a2], q0, q3        \n"
                   "   vldrb.8     q4, [%[w]], #16      \n"
                   "   vmladava.s8 %[a3], q0, q4        \n"
                   "   letp        lr, 2b               \n"
                   "1:                                  \n"
                   : [ip] "+r"(ip), [w] "+r"(w), [a0] "+Te"(a0), [a1] "+Te"(a1), [a2] "+Te"(a2), [a3] "+Te"(a3)
                   : [cnt] "r"(k)
                   : "q0", "q1", "q2", "q3", "q4", "memory", "r14");

    const int32_t *p = (const int32_t *)(const void *)w;
    int32x4_t acc = vdupq_n_s32(0);
    acc = vsetq_lane_s32(a0, acc, 0);
    acc = vsetq_lane_s32(a1, acc, 1);
    acc = vsetq_lane_s32(a2, acc, 2);
    acc = vsetq_lane_s32(a3, acc, 3);
    acc = vaddq_s32(acc, vldrwq_s32(p));
    acc = arm_requantize_mve_32x4_rshift(acc, vldrwq_s32(p + FC_PACKED_ROWS), vldrwq_s32(p + 2 * FC_PACKED_ROWS));
    acc = vaddq_s32(acc, v_off);
    acc = vmaxq_s32(acc, v_min);
    acc = vminq_s32(acc, v_max);
    if (live == FC_PACKED_ROWS)
    {
        vstrbq_s32(dst, acc);
    }
    else
    {
        vstrbq_p_s32(dst, acc, vctp32q((uint32_t)live));
    }
    return w + FC_PACKED_PARAM_BYTES;
}
#endif

/*
 * s8 per-channel fully connected layer on a packed weight stream.
 *
 * Refer header file for details.
 *
 */
arm_cmsis_nn_status arm_fully_connected_per_channel_packed_s8(const cmsis_nn_fc_params *fc_params,
                                                              const cmsis_nn_dims *input_dims,
                                                              const int8_t *input_data,
                                                              const cmsis_nn_dims *filter_dims,
                                                              const int8_t *packed_data,
                                                              const cmsis_nn_dims *output_dims,
                                                              int8_t *output_data)
{
    if (fc_params == NULL || input_dims == NULL || input_data == NULL || packed_data == NULL || output_dims == NULL ||
        output_data == NULL || arm_fully_connected_per_channel_packed_s8_get_packed_size(filter_dims) == 0 ||
        output_dims->c != filter_dims->c || input_dims->n <= 0 || ((uintptr_t)packed_data & 3U) != 0U)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    if (fc_params->filter_offset != 0)
    {
        return ARM_CMSIS_NN_NO_IMPL_ERROR;
    }
#if defined(ARM_MATH_MVEI) && !defined(ARM_MATH_AUTOVECTORIZE) && !defined(CMSIS_NN_USE_SINGLE_ROUNDING)
    const int32_t k = filter_dims->n;
    const int32_t n = filter_dims->c;
    const int32x4_t v_off = vdupq_n_s32(fc_params->output_offset);
    const int32x4_t v_min = vdupq_n_s32(fc_params->activation.min);
    const int32x4_t v_max = vdupq_n_s32(fc_params->activation.max);

    for (int32_t b = 0; b < input_dims->n; b++)
    {
        const int8_t *w = packed_data;
        for (int32_t row0 = 0; row0 < n; row0 += FC_PACKED_ROWS)
        {
            w = arm_fc_packed_s8_block(
                input_data, w, k, ARM_NN_MIN(FC_PACKED_ROWS, n - row0), v_off, v_min, v_max, output_data + row0);
        }
        input_data += k;
        output_data += n;
    }
    return ARM_CMSIS_NN_SUCCESS;
#else
    return ARM_CMSIS_NN_NO_IMPL_ERROR;
#endif
}

/**
 * @} end of FC group
 */

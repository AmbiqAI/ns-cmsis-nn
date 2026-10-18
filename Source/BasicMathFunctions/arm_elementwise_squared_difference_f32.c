/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 */

#include "Internal/arm_nn_fp_control.h"
#include "arm_nnfunctions.h"

#if ARM_NN_ENABLE_F32

    #if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
// The scalar FPU's (a - b) * (a - b), so FPSCR.FZ acts as on the scalar path, kept out of reach of the vectorizer
static inline float32_t arm_squared_difference_scalar_f32(const float32_t a, const float32_t b)
{
    float32_t d, r;
    __asm("vsub.f32 %1, %2, %3\n\t"
          "vmul.f32 %0, %1, %1"
          : "=t"(r), "=&t"(d)
          : "t"(a), "t"(b));
    return r;
}

// Block by block, redoing on the scalar FPU the lanes whose square MVE flushed; exact in place too
static void arm_squared_difference_exact_f32(const float32_t *input_1_vect,
                                             const float32_t *input_2_vect,
                                             float32_t *output,
                                             int32_t block_size)
{
    for (int32_t i = 0; i < block_size; i += 4)
    {
        const mve_pred16_t p = vctp32q((uint32_t)(block_size - i));
        const float32x4_t a = vld1q_z(&input_1_vect[i], p);
        const float32x4_t b = vld1q_z(&input_2_vect[i], p);
        const float32x4_t difference = vsubq(a, b);
        const float32x4_t square = vmulq(difference, difference);
        vstrwq_p(&output[i], square, p);
        const mve_pred16_t redo = vcmpeqq_n_u32(vreinterpretq_u32_f32(square), 0) &
            vcmpneq_n_u32(vshlq_n_u32(vreinterpretq_u32_f32(difference), 1), 0) & p;
        if (redo)
        {
            // From the loaded lanes: output may be either input, already overwritten by the store above
            float32_t as[4], bs[4];
            vst1q(as, a);
            vst1q(bs, b);
            for (int32_t j = 0; j < 4; ++j)
            {
                if (redo & (1u << (4 * j)))
                {
                    output[i + j] = arm_squared_difference_scalar_f32(as[j], bs[j]);
                }
            }
        }
    }
}
    #endif

arm_cmsis_nn_status arm_elementwise_squared_difference_f32(const float32_t *input_1_vect,
                                                           const float32_t *input_2_vect,
                                                           float32_t *output,
                                                           int32_t block_size)
{
    if (!input_1_vect || !input_2_vect || !output || block_size < 1 ||
        block_size > INT32_MAX / (int32_t)sizeof(float32_t))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    #if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
        #if defined(ARM_NN_FP_CONTROL_AVAILABLE)
    const int32_t flush = (arm_nn_get_fp_control() & (1u << 24)) != 0;
        #else
    const int32_t flush = 0;
        #endif
    /*
     * MVE f32 arithmetic flushes subnormal operands and results whatever FPSCR.FZ holds; with FZ set, so does the
     * scalar path. With FZ clear only a square can differ, and only when the difference is a nonzero normal below
     * 2^-63: a flushed operand is below half an ulp of a difference of 2^-101 or more, and a smaller difference
     * squares to +0 either way. The plain loop tracks the smallest (bits << 1) - 1, which takes zeros to UINT32_MAX
     * and exactly those differences below 2^30 - 1. If one turned up, the exact loop reruns from the inputs; an
     * in-place call has overwritten one, so with FZ clear it runs the exact loop from the start. The merging
     * vminq_m matches implicit tail predication, so LLVM keeps the low-overhead loop. Refs #877.
     */
    if (flush || (output != input_1_vect && output != input_2_vect))
    {
        uint32x4_t smallest = vdupq_n_u32(UINT32_MAX);
        for (int32_t i = 0; i < block_size; i += 4)
        {
            const mve_pred16_t p = vctp32q((uint32_t)(block_size - i));
            const float32x4_t difference = vsubq(vld1q_z(&input_1_vect[i], p), vld1q_z(&input_2_vect[i], p));
            vstrwq_p(&output[i], vmulq(difference, difference), p);
            const uint32x4_t key = vsubq_n_u32(vshlq_n_u32(vreinterpretq_u32_f32(difference), 1), 1);
            smallest = vminq_m(smallest, smallest, key, p);
        }
        if (flush || vminvq(UINT32_MAX, smallest) >= 0x3FFFFFFFu)
        {
            return ARM_CMSIS_NN_SUCCESS;
        }
    }
    arm_squared_difference_exact_f32(input_1_vect, input_2_vect, output, block_size);
    #else
    for (int32_t i = 0; i < block_size; ++i)
    {
        const float32_t difference = input_1_vect[i] - input_2_vect[i];
        output[i] = difference * difference;
    }
    #endif
    return ARM_CMSIS_NN_SUCCESS;
}

#endif /* ARM_NN_ENABLE_F32 */

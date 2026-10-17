/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 */

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
    /*
     * MVE f32 arithmetic flushes subnormal operands and results whatever FPSCR.FZ holds. A flushed square is redone:
     * a lane whose difference is normal and whose product came back +0 goes to the scalar FPU, which gives the
     * subnormal (or, with FZ set, the same +0). A flushed operand needs nothing more: next to an operand of 2^-101 or
     * more it is below half an ulp of the difference, next to a smaller normal one the square underflows and is
     * redone, and with no normal operand the square is +0 either way. The tests are on integer lanes; the difference
     * and the product are already flushed, so a compiler that turns them into float compares agrees. Refs #877.
     */
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

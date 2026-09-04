/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_nn_vcvt_f16_fixup.h
 * Description:  Correctly encoded MVE float16<->float32 Q-form conversions
 *
 * $Date:        03 September 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#ifndef ARM_NN_VCVT_F16_FIXUP_H
#define ARM_NN_VCVT_F16_FIXUP_H

#include "arm_nn_compiler.h"
#include "arm_nn_types_flt.h"

#if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)

    /*
     * arm_nn_vcvt{b,t}q_f{16_f32,32_f16}: the four MVE half<->single Q-register
     * conversions. Every float16 kernel that widens or narrows goes through these
     * rather than calling the intrinsics directly.
     *
     * The gas bundled with Arm GNU Toolchain releases before 14.2.Rel1
     * mis-encodes the Q-register form of VCVTB/VCVTT.F16<->F32: each Q operand is
     * emitted with its D-register alias number (Qn as Q2n), so operands land in
     * the wrong registers and any doubled operand past q7 overflows the field
     * into an UNDEFINED word, which faults on first execution. A pragma cannot
     * reach the assembler, so on those compilers the conversion is emitted as a
     * raw instruction word instead: a gas macro maps the register names the
     * compiler substitutes for %q0/%q1 back to numbers and places them in the
     * encoding by hand. Register allocation and scheduling stay with the
     * compiler, only the encoder is bypassed, and the emitted word is identical
     * to what a correct assembler produces. See AmbiqAI/ns-cmsis-nn#427.
     *
     * The preprocessor cannot see the assembler version, so this keys on the
     * compiler major that each Arm GNU release pairs with its binutils. That
     * proxy holds for the Arm GNU releases; a GCC 14 or newer built by hand
     * against an older binutils is outside what this covers. Clang encodes MVE
     * itself and is never affected; there the names below expand straight to
     * the intrinsics.
     */
    #if defined(__GNUC__) && !defined(__clang__) && (__GNUC__ < 14)
        #define ARM_NN_VCVT_F16_FIXUP 1
    #else
        #define ARM_NN_VCVT_F16_FIXUP 0
    #endif

    #if ARM_NN_VCVT_F16_FIXUP

        /*
         * Vector VCVTB/VCVTT.F16<->F32 is a fixed word plus the two register numbers,
         * Qd at bit 13 and Qm at bit 1. The words below are the q0, q0 forms: bit 28
         * selects the direction and bit 12 selects the top half-lanes.
         */
        #define ARM_NN_VCVTB_F16_F32_WORD "0xee3f0e01"
        #define ARM_NN_VCVTT_F16_F32_WORD "0xee3f1e01"
        #define ARM_NN_VCVTB_F32_F16_WORD "0xfe3f0e01"
        #define ARM_NN_VCVTT_F32_F16_WORD "0xfe3f1e01"

        /*
         * Defined and purged inside the asm block so that every expansion, including
         * any copy the compiler makes, is self-contained. The .set names are .L-local
         * so they never reach the symbol table.
         *
         * They do outlive .purgem, though, so a name the .irp fails to match would
         * silently inherit the previous expansion's register number and encode a
         * conversion between the wrong registers. Seeding them out of range makes
         * that a build failure instead.
         */
        #define ARM_NN_VCVT_FIXUP_ASM(word)                                                                            \
            ".macro ns_vcvt_fixup enc, qd, qm\n"                                                                       \
            ".set .Lns_vcvt_qd, -1\n"                                                                                  \
            ".set .Lns_vcvt_qm, -1\n"                                                                                  \
            ".irp num,0,1,2,3,4,5,6,7\n"                                                                               \
            ".ifc \\qd, q\\num\n"                                                                                      \
            ".set .Lns_vcvt_qd, \\num\n"                                                                               \
            ".endif\n"                                                                                                 \
            ".ifc \\qm, q\\num\n"                                                                                      \
            ".set .Lns_vcvt_qm, \\num\n"                                                                               \
            ".endif\n"                                                                                                 \
            ".endr\n"                                                                                                  \
            ".if (.Lns_vcvt_qd < 0) || (.Lns_vcvt_qm < 0)\n"                                                           \
            ".error \"ns_vcvt_fixup: operand is not one of q0-q7\"\n"                                                  \
            ".endif\n"                                                                                                 \
            ".inst.w \\enc + (.Lns_vcvt_qd << 13) + (.Lns_vcvt_qm << 1)\n"                                             \
            ".endm\n"                                                                                                  \
            "ns_vcvt_fixup " word ", %q0, %q1\n"                                                                       \
            ".purgem ns_vcvt_fixup\n"

/**
 * @brief Narrow four float32 lanes into the bottom halves of a float16 vector.
 * @param[in] inactive  Value supplying the untouched top half-lanes.
 * @param[in] a         Source float32 vector.
 * @return Merge of @p inactive and the narrowed lanes.
 */
__STATIC_FORCEINLINE float16x8_t arm_nn_vcvtbq_f16_f32(float16x8_t inactive, float32x4_t a)
{
    float16x8_t d = inactive;
    __asm(ARM_NN_VCVT_FIXUP_ASM(ARM_NN_VCVTB_F16_F32_WORD) : "+w"(d) : "w"(a));
    return d;
}

/**
 * @brief Narrow four float32 lanes into the top halves of a float16 vector.
 * @param[in] inactive  Value supplying the untouched bottom half-lanes.
 * @param[in] a         Source float32 vector.
 * @return Merge of @p inactive and the narrowed lanes.
 */
__STATIC_FORCEINLINE float16x8_t arm_nn_vcvttq_f16_f32(float16x8_t inactive, float32x4_t a)
{
    float16x8_t d = inactive;
    __asm(ARM_NN_VCVT_FIXUP_ASM(ARM_NN_VCVTT_F16_F32_WORD) : "+w"(d) : "w"(a));
    return d;
}

/**
 * @brief Widen the bottom half-lanes of a float16 vector to float32.
 * @param[in] a  Source float16 vector.
 * @return The four widened lanes.
 */
__STATIC_FORCEINLINE float32x4_t arm_nn_vcvtbq_f32_f16(float16x8_t a)
{
    float32x4_t d;
    __asm(ARM_NN_VCVT_FIXUP_ASM(ARM_NN_VCVTB_F32_F16_WORD) : "=w"(d) : "w"(a));
    return d;
}

/**
 * @brief Widen the top half-lanes of a float16 vector to float32.
 * @param[in] a  Source float16 vector.
 * @return The four widened lanes.
 */
__STATIC_FORCEINLINE float32x4_t arm_nn_vcvttq_f32_f16(float16x8_t a)
{
    float32x4_t d;
    __asm(ARM_NN_VCVT_FIXUP_ASM(ARM_NN_VCVTT_F32_F16_WORD) : "=w"(d) : "w"(a));
    return d;
}

    #else

        #define arm_nn_vcvtbq_f16_f32(inactive, a) vcvtbq_f16_f32((inactive), (a))
        #define arm_nn_vcvttq_f16_f32(inactive, a) vcvttq_f16_f32((inactive), (a))
        #define arm_nn_vcvtbq_f32_f16(a) vcvtbq_f32_f16((a))
        #define arm_nn_vcvttq_f32_f16(a) vcvttq_f32_f16((a))

    #endif

#endif

#endif /* ARM_NN_VCVT_F16_FIXUP_H */

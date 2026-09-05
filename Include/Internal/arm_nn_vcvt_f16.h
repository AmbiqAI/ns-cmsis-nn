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
 * Title:        arm_nn_vcvt_f16.h
 * Description:  MVE half<->single conversions that encode on every assembler
 *
 * $Date:        04 September 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#ifndef ARM_NN_VCVT_F16_H
#define ARM_NN_VCVT_F16_H

#include "arm_nn_compiler.h"
#include "arm_nn_types_flt.h"

#if ARM_NN_ENABLE_F16 && defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)

    /*
     * arm_nn_vcvt{b,t}q_f{16_f32,32_f16}: the four MVE half<->single Q-register
     * conversions. Every float16 kernel that widens or narrows goes through these
     * rather than naming the intrinsics directly.
     *
     * gas before binutils 2.43 mis-encodes the vector form of VCVTB/VCVTT.F16<->F32:
     * it writes each Q operand with its D-register alias number (Qn as Q2n), so the
     * operands land in the wrong registers and a doubled number past q7 overflows
     * the field into an UNDEFINED word that faults on first execution. All four
     * directions are affected, the predicated forms included. The scalar VFP S-form
     * of the same four instructions encodes correctly on every gas release, and Qn
     * aliases S(4n)..S(4n+3), so one Q-wide conversion is four S-form instructions
     * over the same bits. On an assembler that needs it that is what the wrappers
     * below emit; everywhere else they are the intrinsics.
     * See AmbiqAI/ns-cmsis-nn#427.
     *
     * NaN payloads are the one behavioural difference between the two forms: the
     * vector form always returns the default NaN, while the scalar form honours
     * FPSCR.DN and so can carry a payload through. NaN stays NaN either way, but a
     * kernel that pinned a particular payload would not be portable between them.
     *
     * The preprocessor cannot see the assembler, so which arm applies is decided by
     * two definitions that the CMake probe in cmake/check_gas_mve_encoding.cmake
     * stamps on the target after measuring it: ARM_NN_GAS_VCVT_F16_BROKEN selects
     * the S-form, ARM_NN_GAS_F16_VERIFIED selects the intrinsics. With neither -- a
     * build that never runs the probe -- the compiler major stands in as a proxy
     * for the binutils each Arm GNU release pairs with. That proxy is conservative
     * in one direction and blind in the other: a GCC 13 driver over a 2.43
     * assembler gets an S-form it does not need, and a GCC 14 driver assembled by
     * hand against an older binutils is not caught. Clang encodes MVE itself and
     * always takes the intrinsics.
     *
     * TODO(#435): delete this header and the wrapper names when GCC 13 leaves the
     * support matrix.
     */
    #if defined(__GNUC__) && !defined(__clang__) &&                                                                    \
        (defined(ARM_NN_GAS_VCVT_F16_BROKEN) || (__GNUC__ < 14 && !defined(ARM_NN_GAS_F16_VERIFIED)))

        /*
         * Assembler-side operand rewrite, carried by every use site rather than
         * defined once at file scope. A top-level asm does not follow the functions of
         * its translation unit into whichever LTRANS partition ends up assembling
         * them, so under -flto a file-scope definition can be absent where the call
         * lands and the assemble stops on an unknown mnemonic. .macro is an error on
         * redefinition, so the .ifndef guard is what lets every site carry its own and
         * still leaves one definition per assembly file. The .set and label names are
         * .L-local so none of this reaches the symbol table, and `%%' is `%' because
         * this is an extended-asm template. An operand the .irp fails to match leaves
         * its number at -1 and stops the assemble, rather than silently encoding the
         * wrong register.
         *
         * The cost is that gcc sizes an asm by counting the lines of its template, so
         * each conversion now looks far larger than the four instructions it becomes
         * and nearby branches get their wide encodings. That is a few bytes on this
         * arm only; the other arm is the intrinsics and is untouched.
         */
        #define ARM_NN_SFORM_DEF                                                                                       \
            ".ifndef .L_nn_sform_defined\n"                                                                            \
            "	.set .L_nn_sform_defined, 1\n"                                                                           \
            "	.macro _nn_sform_emit op, d, m\n"                                                                        \
            "	\\op s\\d, s\\m\n"                                                                                       \
            "	.endm\n"                                                                                                 \
            "	.macro _nn_sform op, d, m\n"                                                                             \
            "	.set .L_nn_sform_d, -1\n"                                                                                \
            "	.set .L_nn_sform_m, -1\n"                                                                                \
            "	.irp i, 0,1,2,3,4,5,6,7\n"                                                                               \
            "	.ifc \\d,q\\i\n"                                                                                         \
            "	.set .L_nn_sform_d, \\i\n"                                                                               \
            "	.endif\n"                                                                                                \
            "	.ifc \\m,q\\i\n"                                                                                         \
            "	.set .L_nn_sform_m, \\i\n"                                                                               \
            "	.endif\n"                                                                                                \
            "	.endr\n"                                                                                                 \
            "	.if (.L_nn_sform_d < 0) || (.L_nn_sform_m < 0)\n"                                                        \
            "	.error \"_nn_sform: operand is not one of q0-q7\"\n"                                                     \
            "	.endif\n"                                                                                                \
            "	.altmacro\n"                                                                                             \
            "	_nn_sform_emit \\op, %%(.L_nn_sform_d*4+0), %%(.L_nn_sform_m*4+0)\n"                                     \
            "	_nn_sform_emit \\op, %%(.L_nn_sform_d*4+1), %%(.L_nn_sform_m*4+1)\n"                                     \
            "	_nn_sform_emit \\op, %%(.L_nn_sform_d*4+2), %%(.L_nn_sform_m*4+2)\n"                                     \
            "	_nn_sform_emit \\op, %%(.L_nn_sform_d*4+3), %%(.L_nn_sform_m*4+3)\n"                                     \
            "	.noaltmacro\n"                                                                                           \
            "	.endm\n"                                                                                                 \
            "	.endif\n"

/*
 * Four S-form instructions per call, one per 32-bit lane: the S registers of a
 * Q register are consecutive, so lane i of the destination and lane i of the
 * source share an offset from their Q bases. Each emitted instruction therefore
 * reads its own source before writing its own destination, and input and output
 * may share a Q register without an earlyclobber constraint.
 */

/**
 * @brief Widen the bottom half-lanes of a float16 vector to float32.
 * @param[in] a  Source float16 vector.
 * @return The four widened lanes.
 */
__STATIC_FORCEINLINE float32x4_t arm_nn_vcvtbq_f32_f16(float16x8_t a)
{
    float32x4_t r;
    __asm__(ARM_NN_SFORM_DEF "_nn_sform vcvtb.f32.f16, %q0, %q1" : "=w"(r) : "w"(a));
    return r;
}

/**
 * @brief Widen the top half-lanes of a float16 vector to float32.
 * @param[in] a  Source float16 vector.
 * @return The four widened lanes.
 */
__STATIC_FORCEINLINE float32x4_t arm_nn_vcvttq_f32_f16(float16x8_t a)
{
    float32x4_t r;
    __asm__(ARM_NN_SFORM_DEF "_nn_sform vcvtt.f32.f16, %q0, %q1" : "=w"(r) : "w"(a));
    return r;
}

/**
 * @brief Narrow four float32 lanes into the bottom halves of a float16 vector.
 * @param[in] inactive  Value supplying the untouched top half-lanes.
 * @param[in] a         Source float32 vector.
 * @return Merge of @p inactive and the narrowed lanes.
 *
 * `vcvtb.f16.f32 Sd, Sm' leaves Sd<31:16> alone, which is the intrinsic's
 * contract that the odd float16 lanes of @p inactive survive.
 */
__STATIC_FORCEINLINE float16x8_t arm_nn_vcvtbq_f16_f32(float16x8_t inactive, float32x4_t a)
{
    float16x8_t r;
    __asm__(ARM_NN_SFORM_DEF "_nn_sform vcvtb.f16.f32, %q0, %q2" : "=w"(r) : "0"(inactive), "w"(a));
    return r;
}

/**
 * @brief Narrow four float32 lanes into the top halves of a float16 vector.
 * @param[in] inactive  Value supplying the untouched bottom half-lanes.
 * @param[in] a         Source float32 vector.
 * @return Merge of @p inactive and the narrowed lanes.
 */
__STATIC_FORCEINLINE float16x8_t arm_nn_vcvttq_f16_f32(float16x8_t inactive, float32x4_t a)
{
    float16x8_t r;
    __asm__(ARM_NN_SFORM_DEF "_nn_sform vcvtt.f16.f32, %q0, %q2" : "=w"(r) : "0"(inactive), "w"(a));
    return r;
}

        #undef ARM_NN_SFORM_DEF

    #else

/*
 * Same four names and the same contracts as the S-form arm above, straight
 * through to the intrinsics. Wrappers rather than macros so that both arms
 * present one shape to a reader and to anything that walks call graphs over
 * this tree.
 */

__STATIC_FORCEINLINE float32x4_t arm_nn_vcvtbq_f32_f16(float16x8_t a) { return vcvtbq_f32_f16(a); }

__STATIC_FORCEINLINE float32x4_t arm_nn_vcvttq_f32_f16(float16x8_t a) { return vcvttq_f32_f16(a); }

__STATIC_FORCEINLINE float16x8_t arm_nn_vcvtbq_f16_f32(float16x8_t inactive, float32x4_t a)
{
    return vcvtbq_f16_f32(inactive, a);
}

__STATIC_FORCEINLINE float16x8_t arm_nn_vcvttq_f16_f32(float16x8_t inactive, float32x4_t a)
{
    return vcvttq_f16_f32(inactive, a);
}

    #endif

#endif /* ARM_NN_ENABLE_F16 && ARM_MATH_MVE_FLOAT16 && !ARM_MATH_AUTOVECTORIZE */

#endif /* ARM_NN_VCVT_F16_H */

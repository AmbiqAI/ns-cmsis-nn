/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_hard_swish_precise_s8.c
 * Description:  Hard Swish function for int8_t data type
 *
 * $Date:        09 September 2025
 * $Revision:    V.1.0.0
 *
 * Target Processor:  Cortex-M cores
 *
 * -------------------------------------------------------------------- */

#include "arm_nn_types.h"
#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

/**
*  @ingroup groupNN
*/

/**
* @addtogroup Acti
* @{
*/

/*
* Hard Swish activation function for int8_t data type.
*
* Refer header file for details.
*
*/
/**
 * @brief Hard-Swish activation (quantized, int8)
 *
 * Implements the piecewise activation:
 *
 *     y = x * relu6(x + 3) / 6
 *
 * using affine per-tensor quantization. Let:
 *   - x_q, y_q : quantized input/output values (int8)
 *   - z_in, z_out : input/output zero points
 *   - s_in, s_out : input/output scales
 *
 * The kernel operates in the input integer domain:
 *
 *   x_i = x_q - z_in
 *   r_i = clamp(x_i + q3, 0, q6)                 // q3 = round(3 / s_in), q6 = round(6 / s_in)
 *   y_i = x_i * r_i
 *   y_q = clamp( requantize(y_i, M) + z_out, -128, 127 )
 *
 * where the real multiplier is:
 *
 *   M_real = s_in^2 / (6 * s_out)
 *
 * and is represented by (output_multiplier, output_shift) for arm_nn_requantize().
 *
 * @param[in]  act_params         Hard-swish parameters in input LSBs:
 *                                - pre_activation_offset == q3 (round(3 / s_in))
 *                                - relu6_cap            == q6 (round(6 / s_in))
 * @param[in]  input              Pointer to input tensor (int8)
 * @param[in]  input_offset       z_in (input zero point)
 * @param[in]  output_offset      z_out (output zero point)
 * @param[in]  output_multiplier  Quantized multiplier for M_real (Q31)
 * @param[in]  output_shift       Base-2 exponent for M_real (see arm_nn_requantize)
 * @param[out] output             Pointer to output tensor (int8)
 * @param[in]  output_size        Number of elements
 *
 * @return     ARM_CMSIS_NN_SUCCESS on success
 *
 * @note To precompute constants:
 *   // in floating-point host code
 *   params.pre_activation_offset = (int32_t)lrint(3.0 / s_in);   // q3
 *   params.relu6_cap             = (int32_t)lrint(6.0 / s_in);   // q6
 *   double M_real = (s_in * s_in) / (6.0 * s_out);
 *   quantize_multiplier(M_real, &output_multiplier, &output_shift);
 *
 * @note An MVE/Helium path is used when ARM_MATH_MVEI is defined; otherwise a scalar fallback is used.
 */
arm_cmsis_nn_status arm_hard_swish_precise_s8(
    const int8_t *input,
    const int32_t input_offset,
    const int32_t output_offset,
    const int32_t output_multiplier,
    const int32_t output_shift,
    const int32_t relu_q3,
    const int32_t relu_q6,
    int8_t *output,
    const int32_t output_size)
{

    if (!input || !output || output_size < 0)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    int32_t flat_size = output_size;
    const int32_t qmin = INT8_MIN;
    const int32_t qmax = INT8_MAX;

#if defined(ARM_MATH_MVEI)

    // Perform 4 operations in parallel
    uint32_t blkCnt = (flat_size + 3) / 4U;
    while (blkCnt > 0U)
    {
        // Load input and create predicate
        mve_pred16_t pred = vctp32q((uint32_t)flat_size);
        int32x4_t x = vldrbq_z_s32(input, pred);

        // Center input to remove zero-point
        x = vsubq_s32(x, vdupq_n_s32(input_offset));

        // Compute xr = clamp(x + relu_q3, 0, relu_q6)
        int32x4_t xr = vaddq_n_s32(x, relu_q3);
        xr = vmaxq_s32(xr, vdupq_n_s32(0));
        xr = vminq_s32(xr, vdupq_n_s32(relu_q6));

        // Integer product x * xr (unit: s_in^2)
        int32x4_t y = vmulq_s32(x, xr);

        // Requantize the result (includes division by 6)
        y = arm_requantize_mve(y, output_multiplier, output_shift);
        y = vaddq_n_s32(y, output_offset);

        // Clamp and store
        y = vmaxq_s32(y, vdupq_n_s32(qmin));
        y = vminq_s32(y, vdupq_n_s32(qmax));
        vstrbq_p_s32(output, y, pred);

        // Increment pointers
        input += 4;
        output += 4;
        blkCnt -= 1;
        flat_size -= 4;
    }

#else

    for (int32_t i = 0; i < flat_size; i++)
    {
        // Center input to remove zero-point
        int32_t x = (int32_t)input[i] - input_offset;

        // Compute xr = clamp(x + relu_q3, 0, relu_q6)
        int32_t xr = CLAMP(x + relu_q3, relu_q6, 0);

        // Integer product x * xr (unit: s_in^2)
        int32_t y = x * xr;

        // Requantize the result (includes division by 6)
        y = arm_nn_requantize(y, output_multiplier, output_shift);
        y += output_offset;

        // Clamp and store
        y = CLAMP(y, qmax, qmin);
        output[i] = (int8_t)y;
    }

#endif

    return ARM_CMSIS_NN_SUCCESS;

}

/**
 * @} end of Doxygen group
 */

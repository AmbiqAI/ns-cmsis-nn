/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_gru_unidirectional_f32.c
 * Description:  Unidirectional GRU (float32)
 *
 * $Date:        14 August 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

/**
 * @ingroup Public
 */

/**
 * @addtogroup LSTM
 * @{
 */

#if ARM_NN_ENABLE_F32

/*
 * Unidirectional GRU (float32)
 *
 * Refer to header file for details.
 *
 */

arm_cmsis_nn_status arm_gru_unidirectional_f32(const float32_t *input,
                                               float32_t *output,
                                               const cmsis_nn_gru_params_f32 *params,
                                               cmsis_nn_gru_context_f32 *buffers)
{
    if (!input || !output || !params)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    // Reject non-positive dimensions: a negative size would otherwise reach
    // (size_t)h * (size_t)size pointer arithmetic (wraps; formal UB) and
    // return SUCCESS with garbage or no output at all. time_steps == 0 stays
    // legal: a zero-length streaming chunk is a well-defined no-op.
    if (params->batch_size < 1 || params->time_steps < 0 || params->input_size < 1 || params->hidden_size < 1)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    // The pre-reset formulation needs scratch; reset-after does not.
    if (!params->reset_after && (!buffers || !buffers->temp1))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    // Streaming state carry: when a hidden_state buffer is supplied it is used
    // as the initial state (seed to zero for a fresh sequence) and updated with
    // the final hidden state on return. This carries state across calls (e.g.
    // chunked inference). It is only defined for batch_size == 1.
    const int stateful = (buffers != NULL && buffers->hidden_state != NULL);
    if (stateful && params->batch_size != 1)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    const float32_t *hidden_in = stateful ? buffers->hidden_state : NULL;
    float32_t *last_hidden = NULL;

    if (params->time_major)
    {
        for (int32_t t = 0; t < params->time_steps; t++)
        {
            const float32_t *data_in = input + (size_t)t * (size_t)params->batch_size * (size_t)params->input_size;
            float32_t *hidden_out = output + (size_t)t * (size_t)params->batch_size * (size_t)params->hidden_size;
            arm_cmsis_nn_status status = arm_nn_gru_step_f32(data_in, hidden_in, hidden_out, params, buffers, 1);
            if (status != ARM_CMSIS_NN_SUCCESS)
            {
                return status;
            }
            hidden_in = hidden_out;
            last_hidden = hidden_out;
        }
    }
    else
    {
        for (int32_t t = 0; t < params->time_steps; t++)
        {
            const float32_t *data_in = input + (size_t)t * (size_t)params->input_size;
            float32_t *hidden_out = output + (size_t)t * (size_t)params->hidden_size;
            arm_cmsis_nn_status status =
                arm_nn_gru_step_f32(data_in, hidden_in, hidden_out, params, buffers, params->time_steps);
            if (status != ARM_CMSIS_NN_SUCCESS)
            {
                return status;
            }
            hidden_in = hidden_out;
            last_hidden = hidden_out;
        }
    }

    // Persist the final hidden state for the next streaming call.
    if (stateful && last_hidden != NULL)
    {
        for (int32_t h = 0; h < params->hidden_size; h++)
        {
            buffers->hidden_state[h] = last_hidden[h];
        }
    }

    return ARM_CMSIS_NN_SUCCESS;
}

/*
 * Bytes written through buffers->temp1 by arm_gru_unidirectional_f32(). Derived from the kernel: temp1 is
 * dereferenced only on the pre-reset (reset_after == 0) path of arm_nn_gru_step_f32(), where the reset-gate
 * loop stores reset_buf[h] for h in [0, hidden_size) and the candidate matmul reads the same extent back; the
 * vector is reused across batches and time steps, so neither batch_size nor time_steps enters. On the
 * reset-after path (reset_after != 0) the reset gate is folded per hidden unit in automatics, temp1 is never
 * dereferenced on any build path, and both the wrapper and the step accept it as NULL - so the requirement is
 * zero there.
 */
int32_t arm_gru_unidirectional_f32_temp1_get_buffer_size(const cmsis_nn_gru_params_f32 *gru_params)
{
    if (gru_params == NULL || gru_params->hidden_size < 0)
    {
        return -1;
    }

    if (gru_params->reset_after != 0)
    {
        return 0;
    }

    // Folded one factor at a time so the accumulator stays bounded; see arm_nn_size_mul().
    int64_t required_bytes = arm_nn_size_mul(1, gru_params->hidden_size);
    required_bytes = arm_nn_size_mul(required_bytes, (int32_t)sizeof(float32_t));

    return (int32_t)required_bytes;
}

#endif /* ARM_NN_ENABLE_F32 */

/** @} */

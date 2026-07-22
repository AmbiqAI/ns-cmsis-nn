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
 * Title:        arm_gru_unidirectional_f16.c
 * Description:  Unidirectional GRU (float16)
 *
 * $Date:        22 July 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture with FP16 support
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

#if ARM_NN_ENABLE_F16

/*
 * Unidirectional GRU (float16)
 *
 * Refer to header file for details.
 *
 */

arm_cmsis_nn_status arm_gru_unidirectional_f16(const float16_t *input,
                                               float16_t *output,
                                               const cmsis_nn_gru_params_f16 *params,
                                               cmsis_nn_gru_context_f16 *buffers)
{
    if (!input || !output || !params)
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

    const float16_t *hidden_in = stateful ? buffers->hidden_state : NULL;
    float16_t *last_hidden = NULL;

    if (params->time_major)
    {
        for (int32_t t = 0; t < params->time_steps; t++)
        {
            const float16_t *data_in = input + (size_t)t * (size_t)params->batch_size * (size_t)params->input_size;
            float16_t *hidden_out = output + (size_t)t * (size_t)params->batch_size * (size_t)params->hidden_size;
            arm_cmsis_nn_status status = arm_nn_gru_step_f16(data_in, hidden_in, hidden_out, params, buffers, 1);
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
            const float16_t *data_in = input + (size_t)t * (size_t)params->input_size;
            float16_t *hidden_out = output + (size_t)t * (size_t)params->hidden_size;
            arm_cmsis_nn_status status =
                arm_nn_gru_step_f16(data_in, hidden_in, hidden_out, params, buffers, params->time_steps);
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

#endif /* ARM_NN_ENABLE_F16 */

/** @} */

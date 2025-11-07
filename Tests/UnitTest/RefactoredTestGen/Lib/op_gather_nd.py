# SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import numpy as np
import tensorflow as tf
import tf_keras as keras

import Lib.op_utils


class Op_gather_nd(Lib.op_utils.Op_type):

    @staticmethod
    def get_shapes(params):
        full_params_shape = tuple(params["params_dim"])
        full_indices_shape = tuple(params["indices_dim"])

        if len(full_params_shape) != 4 or len(full_indices_shape) != 4:
            raise ValueError("Op_gather_nd expects params_dim/indices_dim with 4 entries (n, h, w, c)")

        params_rank = params.get("params_rank", len(full_params_shape))
        indices_rank = params.get("indices_rank", len(full_indices_shape))
        indices_last_dim = params["indices_last_dim"]

        if not (1 <= params_rank <= 4) or not (1 <= indices_rank <= 4):
            raise ValueError("params_rank and indices_rank must be within [1, 4]")

        params_shape = tuple(full_params_shape[:params_rank])
        indices_shape = tuple(full_indices_shape[:indices_rank])

        if indices_shape[-1] != indices_last_dim:
            raise ValueError("indices_last_dim must match the last dimension of indices_shape")

        indices_values = np.array(params["indices_values"], dtype=np.int32).reshape(indices_shape)

        shapes = {
            "input_tensor": params_shape,
            "representational_dataset": params_shape,
            "indices_tensor": indices_shape,
            "indices_values": indices_values,
            "params_rank": params_rank,
            "indices_rank": indices_rank,
            "indices_last_dim": indices_last_dim,
            "batch_dims": params.get("batch_dims", 0),
        }

        return shapes

    @staticmethod
    def generate_keras_model(shapes, params):
        tf.keras.backend.clear_session()
        input_tensor = keras.layers.Input(batch_input_shape=shapes["input_tensor"], name="input_tensor")

        indices_const = tf.constant(shapes["indices_values"], dtype=tf.int32)
        batch_dims = params.get("batch_dims", 0)

        def gather_nd_fn(t):
            return tf.gather_nd(t, indices_const, batch_dims=batch_dims)

        output = keras.layers.Lambda(gather_nd_fn, name="gather_nd")(input_tensor)
        return keras.Model(inputs=input_tensor, outputs=output)

    @staticmethod
    def generate_data_tflite(tflite_fname, params):
        del tflite_fname  # Not needed for parameter extraction

        tensors = {}
        effective_scales = {}
        scales = {}

        full_params_shape = tuple(params["params_dim"])
        full_indices_shape = tuple(params["indices_dim"])
        params_rank = params.get("params_rank", len(full_params_shape))
        indices_rank = params.get("indices_rank", len(full_indices_shape))
        indices_last_dim = params["indices_last_dim"]
        batch_dims = params.get("batch_dims", 0)

        params_shape = tuple(full_params_shape[:params_rank])
        indices_shape = tuple(full_indices_shape[:indices_rank])

        if indices_shape[-1] != indices_last_dim:
            raise ValueError("indices_last_dim must match the last dimension of indices_shape")

        if (batch_dims < 0) or (batch_dims > params_rank) or (batch_dims > indices_rank - 1):
            raise ValueError("Invalid batch_dims for gather_nd")

        indices_values = np.array(params["indices_values"], dtype=np.int32).reshape(indices_shape)
        tensors["indices_tensor"] = indices_values

        # Output shape calculation per TensorFlow gather_nd spec
        output_shape = list(params_shape[:batch_dims])
        if indices_rank - 1 > batch_dims:
            output_shape += list(indices_shape[batch_dims:-1])
        tail_start = batch_dims + indices_last_dim
        if tail_start < len(params_shape):
            output_shape += list(params_shape[tail_start:])

        if not output_shape:
            output_shape = [1]

        padded_output_shape = (output_shape + [1] * 4)[:4]
        output_rank = len(output_shape)

        generated_params = {
            "params_n": full_params_shape[0],
            "params_h": full_params_shape[1],
            "params_w": full_params_shape[2],
            "params_c": full_params_shape[3],
            "params_rank": params_rank,
            "indices_n": full_indices_shape[0],
            "indices_h": full_indices_shape[1],
            "indices_w": full_indices_shape[2],
            "indices_c": full_indices_shape[3],
            "indices_rank": indices_rank,
            "indices_last_dim": indices_last_dim,
            "indices_length": int(indices_values.size),
            "batch_dims": batch_dims,
            "output_rank": output_rank,
            "output_n": padded_output_shape[0],
            "output_h": padded_output_shape[1],
            "output_w": padded_output_shape[2],
            "output_c": padded_output_shape[3],
            "output_size": int(np.prod(output_shape)),
        }

        return Lib.op_utils.Generated_data(generated_params, tensors, scales, effective_scales)

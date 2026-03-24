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


class Op_gather(Lib.op_utils.Op_type):

    @staticmethod
    def get_shapes(params):
        shapes = {}
        full_input = tuple(params["input_dim"])
        full_indices = tuple(params["indices_dim"])
        if len(full_input) != 4 or len(full_indices) != 4:
            raise ValueError("Op_gather expects input_dim and indices_dim with 4 entries (n, h, w, c)")

        input_rank = params.get("input_rank", len(full_input))
        indices_rank = params.get("indices_rank", len(full_indices))
        if not (1 <= input_rank <= 4):
            raise ValueError("input_rank must be between 1 and 4")
        if not (1 <= indices_rank <= 4):
            raise ValueError("indices_rank must be between 1 and 4")

        input_shape = tuple(full_input[:input_rank])
        indices_shape = tuple(full_indices[:indices_rank])
        indices_values = np.array(params["indices_values"], dtype=np.int32).reshape(indices_shape)

        shapes["input_tensor"] = input_shape
        shapes["representational_dataset"] = input_shape
        shapes["indices_tensor"] = indices_shape
        shapes["indices_values"] = indices_values
        shapes["axis"] = params["axis"]
        shapes["batch_dims"] = params.get("batch_dims", 0)
        shapes["input_rank"] = input_rank
        shapes["indices_rank"] = indices_rank

        return shapes

    @staticmethod
    def generate_keras_model(shapes, params):
        tf.keras.backend.clear_session()
        input_tensor = keras.layers.Input(batch_input_shape=shapes["input_tensor"], name="input_tensor")

        axis_param = params["axis"]
        rank = shapes["input_rank"]
        axis_runtime = axis_param
        if axis_runtime < 0:
            axis_runtime += rank
        axis_runtime = axis_runtime % rank
        batch_dims = params.get("batch_dims", 0)
        indices_const = tf.constant(shapes["indices_values"], dtype=tf.int32)

        def gather_fn(t):
            return tf.gather(t, indices_const, axis=axis_runtime, batch_dims=batch_dims)

        output = keras.layers.Lambda(gather_fn, name="gather")(input_tensor)
        return keras.Model(inputs=input_tensor, outputs=output)

    @staticmethod
    def generate_data_tflite(tflite_fname, params):
        del tflite_fname  # not required for parameter extraction

        tensors = {}
        effective_scales = {}
        scales = {}

        full_input = tuple(params["input_dim"])
        full_indices = tuple(params["indices_dim"])
        input_rank = params.get("input_rank", len(full_input))
        indices_rank = params.get("indices_rank", len(full_indices))
        input_shape = tuple(full_input[:input_rank])
        indices_shape = tuple(full_indices[:indices_rank])
        indices_values = np.array(params["indices_values"], dtype=np.int32).reshape(indices_shape)
        tensors["indices_tensor"] = indices_values

        axis_param = params["axis"]
        axis_runtime = axis_param
        if axis_runtime < 0:
            axis_runtime += input_rank
        axis_runtime = axis_runtime % input_rank
        batch_dims = params.get("batch_dims", 0)

        output_shape_full = (list(input_shape[:axis_runtime]) +
                             list(indices_shape[batch_dims:]) +
                             list(input_shape[axis_runtime + 1:]))
        if not output_shape_full:
            output_shape_full = [1]

        output_size = int(np.prod(output_shape_full))
        if len(output_shape_full) >= 3:
            output_n = output_shape_full[0]
            output_h = output_shape_full[1]
            output_w = output_shape_full[2]
            output_c = int(np.prod(output_shape_full[3:])) if len(output_shape_full) > 3 else 1
        else:
            padded = output_shape_full + [1] * (3 - len(output_shape_full))
            output_n, output_h, output_w = padded[:3]
            output_c = 1

        indices_shape_full = full_indices
        generated_params = {
            "input_n": full_input[0],
            "input_h": full_input[1],
            "input_w": full_input[2],
            "input_c": full_input[3],
            "input_rank": input_rank,
            "indices_n": indices_shape_full[0],
            "indices_h": indices_shape_full[1],
            "indices_w": indices_shape_full[2],
            "indices_c": indices_shape_full[3],
            "coords_rank": indices_rank,
            "indices_length": int(indices_values.size),
            "axis": axis_param,
            "batch_dims": batch_dims,
            "output_n": output_n,
            "output_h": output_h,
            "output_w": output_w,
            "output_c": output_c,
            "output_size": output_size,
        }

        return Lib.op_utils.Generated_data(generated_params, tensors, scales, effective_scales)

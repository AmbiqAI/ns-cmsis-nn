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
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import numpy as np
import tensorflow as tf
from tensorflow.lite.python.interpreter import OpResolverType
from Lib.keras_compat import keras

import Lib.op_utils


class Op_arg_min_max(Lib.op_utils.Op_type):

    def get_shapes(params):
        shapes = {}
        shapes["lhs_input_tensor"] = (params["lhs_n"], params["lhs_h"], params["lhs_w"], params["lhs_c"])
        shapes["representational_dataset"] = shapes["lhs_input_tensor"]
        return shapes

    def generate_keras_model(shapes, params):
        tf.keras.backend.clear_session()
        input_tensor = keras.layers.Input(batch_input_shape=shapes["lhs_input_tensor"], name="input_tensor")

        axis = params["axis"]
        mode = params["mode"]

        def arg_min_max_fn(x):
            if mode == "argmax":
                result = tf.math.argmax(x, axis=axis)
            else:
                result = tf.math.argmin(x, axis=axis)
            return tf.cast(result, tf.int32)

        def compute_output_shape(input_shape):
            shape_list = list(input_shape)
            # Remove the axis dimension
            shape_list.pop(axis)
            if len(shape_list) == 0:
                return (1,)
            return tuple(shape_list)

        output = keras.layers.Lambda(arg_min_max_fn,
                                     output_shape=compute_output_shape,
                                     name="arg_" + mode)(input_tensor)
        model = tf.keras.Model(inputs=input_tensor, outputs=output, name="arg_min_max_model")
        return model

    def generate_data_tflite(tflite_fname, params):
        tensors = {}
        effective_scales = {}
        scales = {}
        generated_params = {}

        interpreter = tf.lite.Interpreter(str(tflite_fname), experimental_op_resolver_type=OpResolverType.BUILTIN_REF)
        interpreter.allocate_tensors()
        tensor_details = interpreter.get_tensor_details()

        # Input tensor is first entry, output is last
        input_info = tensor_details[0]
        output_info = tensor_details[-1]

        interpreter_tensor = interpreter.tensor(input_info["index"])()
        tensors["lhs_input_tensor"] = interpreter_tensor.copy()

        output_tensor = interpreter.tensor(output_info["index"])().astype(np.int32)
        tensors["output"] = output_tensor

        generated_params["axis"] = params["axis"]
        generated_params["mode"] = params["mode"]
        generated_params["dst_size"] = tensors["output"].size
        generated_params["output_dims"] = list(tensors["output"].shape)
        generated_params["output_data_type"] = "int32_t"

        return Lib.op_utils.Generated_data(generated_params, tensors, scales, effective_scales)

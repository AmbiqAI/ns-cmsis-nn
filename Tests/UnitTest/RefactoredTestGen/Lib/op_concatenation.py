# SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import Lib.op_utils
import numpy as np
import tensorflow as tf

from tensorflow.lite.python.interpreter import Interpreter
from tensorflow.lite.python.interpreter import OpResolverType
from Lib.keras_compat import keras


class Op_concatenation(Lib.op_utils.Op_type):
    @staticmethod
    def get_shapes(params):
        shapes = {}

        input_shape_1 = (
            params["batch_1"],
            params["height_1"],
            params["width_1"],
            params["channel_1"],
        )
        input_shape_2 = (
            params["batch_2"],
            params["height_2"],
            params["width_2"],
            params["channel_2"],
        )

        shapes["input_tensor_1"] = input_shape_1
        shapes["input_tensor_2"] = input_shape_2
        shapes["representational_dataset"] = input_shape_1
        shapes["representational_dataset2"] = input_shape_2
        shapes["different_in_shapes"] = True

        return shapes

    @staticmethod
    def generate_keras_model(shapes, params):
        dtype = Lib.op_utils.get_tf_dtype(params["input_data_type"])
        axis = params["axis"]

        input_1 = keras.layers.Input(batch_input_shape=shapes["input_tensor_1"], dtype=dtype, name="input_1")
        input_2 = keras.layers.Input(batch_input_shape=shapes["input_tensor_2"], dtype=dtype, name="input_2")
        output = tf.concat([input_1, input_2], axis=axis, name="concat")
        model = keras.Model([input_1, input_2], [output])

        return model

    @staticmethod
    def generate_data_tflite(tflite_fname, params):
        tensors = {}
        effective_scales = {}
        scales = {}
        aliases = {}
        generated_params = {}

        interpreter = Interpreter(str(tflite_fname), experimental_op_resolver_type=OpResolverType.BUILTIN_REF)
        interpreter.allocate_tensors()
        output_shape = interpreter.get_output_details()[0]["shape"].tolist()

        input_shape_1 = [
            params["batch_1"],
            params["height_1"],
            params["width_1"],
            params["channel_1"],
        ]
        input_shape_2 = [
            params["batch_2"],
            params["height_2"],
            params["width_2"],
            params["channel_2"],
        ]

        generated_params["inputs_count"] = 2
        generated_params["axis"] = params["axis"]
        generated_params["output_dims"] = len(output_shape)
        generated_params["output_shape"] = output_shape

        generated_params["input_concat_dims"] = [
            input_shape_1[params["axis"]],
            input_shape_2[params["axis"]],
        ]

        generated_params["output_size"] = int(np.prod(output_shape))
        generated_params["output_n"] = output_shape[0]
        generated_params["output_h"] = output_shape[1]
        generated_params["output_w"] = output_shape[2]
        generated_params["output_c"] = output_shape[3]

        return Lib.op_utils.Generated_data(generated_params, tensors, scales, effective_scales, aliases)

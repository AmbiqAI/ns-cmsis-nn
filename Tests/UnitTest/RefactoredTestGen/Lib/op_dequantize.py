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
import tensorflow as tf
import math
import numpy as np

from tensorflow.lite.python.interpreter import Interpreter
from tensorflow.lite.python.interpreter import OpResolverType
import tf_keras as keras

class Op_dequantize(Lib.op_utils.Op_type):
    def get_shapes(params):
        shapes = {}
        shapes["input_tensor_1"] = (
            params["batch_1"],
            params["height_1"],
            params["width_1"],
            params["channel_1"]
        )

        shapes["representational_dataset"] = (
            params["batch_1"],
            params["height_1"],
            params["width_1"],
            params["channel_1"]
        )

        shapes["different_in_shapes"] = False

        return shapes


    def generate_keras_model(shapes, params):
        model = keras.Sequential()
        model.add(keras.Input(shape=shapes["input_tensor_1"][1:]))

        # Flatten it to a 1D vector
        model.add(keras.layers.Flatten())

        # Insert Dense(params["height_1"] * params["width_1"] * params["channel_1"]) with identity weights
        dense = keras.layers.Dense(units=params["height_1"] * params["width_1"] * params["channel_1"], use_bias=False, activation=None)
        model.add(dense)
        dense.build((None,params["height_1"] * params["width_1"] * params["channel_1"]))

        identity_weights = np.eye(params["height_1"] * params["width_1"] * params["channel_1"], dtype=np.float32)
        dense.set_weights([identity_weights])


        return model

    def generate_data_tflite(tflite_fname, params):
        """
        Generate random input data for the quantize op (float32, int8, int16, etc.)
        based on params["input_data_type"], read quantization details (scale,
        zero-point), and store them for downstream use.
        """
        tensors = {}
        effective_scales = {}
        scales = {}
        generated_params = {}
        aliases = {}

        input_shape = (
            params["batch_1"],
            params["height_1"],
            params["width_1"],
            params["channel_1"]
        )

        input_dtype_str = params["input_data_type"]
        output_dtype_str = params["output_data_type"]

        input_dtype_tf = Lib.op_utils.get_tf_dtype(input_dtype_str)

        if input_dtype_tf == tf.int8:
            # Generate random int8 data in range [-128, 127]
            random_data = np.random.randint(-128, 128, size=input_shape).astype(np.int8)
        elif input_dtype_tf == tf.int16:
            # Generate random int16 data in range [-32768, 32767]
            random_data = np.random.randint(-32768, 32768, size=input_shape).astype(np.int16)
        else:
            raise ValueError(f"Unsupported input dtype: {input_dtype_tf}")

        tensors["input_tensor_1"] = random_data


        from tensorflow.lite.python.interpreter import Interpreter, OpResolverType
        interpreter = Interpreter(
            model_path=str(tflite_fname),
            experimental_op_resolver_type=OpResolverType.BUILTIN_REF
        )
        interpreter.allocate_tensors()

        # 4. Get input quantization info.
        input_details = interpreter.get_input_details()
        (input_scale, input_zero_point) = input_details[0]['quantization']
        scales["input_scale_" + output_dtype_str] = input_scale
        scales["input_zero_point_" + output_dtype_str] = input_zero_point
        generated_params["quant_input_scale_" + output_dtype_str] = input_scale
        generated_params["quant_input_zero_point_" + output_dtype_str] = input_zero_point

        # 5. Get output quantization info.
        output_details = interpreter.get_output_details()
        (output_scale, output_zero_point) = output_details[0]['quantization']
        scales["output_scale_" + output_dtype_str] = output_scale
        scales["output_zero_point_" + output_dtype_str] = output_zero_point
        generated_params["quant_output_scale_" + output_dtype_str] = output_scale
        generated_params["quant_output_zero_point_" + output_dtype_str] = output_zero_point
        

        out_shape = output_details[0]["shape"]
        generated_params["output_len"] = int(np.prod(out_shape))
        
        return Lib.op_utils.Generated_data(
            generated_params,
            tensors,
            scales,
            effective_scales,
            aliases
        )


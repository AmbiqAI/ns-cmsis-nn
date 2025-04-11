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
        # The real input shape for your op:
        shapes["input_tensor_1"] = (
            params["batch_1"],
            params["height_1"],
            params["width_1"],
            params["channel_1"]
        )

        # Provide the same shape to representational_dataset unless you want something smaller:
        shapes["representational_dataset"] = (
            params["batch_1"],
            params["height_1"],
            params["width_1"],
            params["channel_1"]
        )

        # Usually for a single-input op, you set:
        shapes["different_in_shapes"] = False

        return shapes


    def generate_keras_model(shapes, params):
        model = keras.Sequential()
        # Accept shape = (1,8,1) as (H,W,C)
        model.add(keras.Input(shape=(1,8,1)))

        # Flatten it to a 1D vector of length 8
        model.add(keras.layers.Flatten())  # Now we have shape=(None, 8)

        # Insert Dense(8) with identity weights
        dense = keras.layers.Dense(units=8, use_bias=False, activation=None)
        model.add(dense)
        dense.build((None,8))

        identity_weights = np.eye(8, dtype=np.float32)
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

        # 1. Figure out input shape from the test parameters (single input example).
        input_shape = (
            params["batch_1"],
            params["height_1"],
            params["width_1"],
            params["channel_1"]
        )

        # -------------------------------------------------------------------------
        # 2. Determine the input dtype from params["input_data_type"] and generate
        #    random data in an appropriate numeric range.
        # -------------------------------------------------------------------------
        input_dtype_str = params["input_data_type"]  # e.g. "float32_to_int8", "int8_t_to_int8_t", "int16_t_to_int16_t"
        output_dtype_str = ""
        if "_to_" in input_dtype_str:
            tokens = input_dtype_str.split("_to_")
            # tokens[0] is the "left" side
            # tokens[1] is the "right" side
            input_dtype_str = tokens[0]
            output_dtype_str = tokens[1]
        input_dtype_tf = Lib.op_utils.get_tf_dtype(input_dtype_str)

        # Generate random data according to the chosen dtype
        if input_dtype_tf == tf.int8:
            # Generate random int8 data in range [-128, 127]
            random_data = np.random.randint(-128, 128, size=input_shape).astype(np.int8)
        elif input_dtype_tf == tf.int16:
            # Generate random int16 data in range [-32768, 32767]
            random_data = np.random.randint(-32768, 32768, size=input_shape).astype(np.int16)
        else:
            # Fallback or raise an error for unsupported dtypes
            raise ValueError(f"Unsupported input dtype: {input_dtype_tf}")

        tensors["input_tensor_1"] = random_data

        # -------------------------------------------------------------------------
        # 3. Create a TFLite interpreter to read the scale and zero-point from
        #    the TFLite file.
        # -------------------------------------------------------------------------
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
        generated_params["output_shape"] = out_shape.tolist()  # for example

        # 8. Return the data container
        return Lib.op_utils.Generated_data(
            generated_params,
            tensors,
            scales,
            effective_scales,
            aliases
        )


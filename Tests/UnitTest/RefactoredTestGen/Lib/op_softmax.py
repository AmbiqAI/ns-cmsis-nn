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
import math
import tensorflow as tf
import numpy as np

from tensorflow.lite.python.interpreter import Interpreter
from tensorflow.lite.python.interpreter import OpResolverType
import tf_keras as keras

class Op_softmax(Lib.op_utils.Op_type):
    
    def get_shapes(params):
        shapes = {}

        shapes["input_tensor"] = (
            params["num_rows"],
            params["row_size"]
        )

        shapes["representational_dataset"] = (
            params["num_rows"],
            params["row_size"]
        )

        shapes["different_in_shapes"] = False

        return shapes


    def generate_keras_model(shapes, params):
        tf.keras.backend.clear_session()

        input_1 = keras.layers.Input(batch_input_shape=shapes["input_tensor"])
        layer = tf.nn.softmax(input_1)
        model = keras.Model([input_1], [layer])
        return model

    def quantize_scale(scale):
        significand, shift = math.frexp(scale)
        significand_q31 = round(significand * (1 << 31))
        return significand_q31, shift

    def generate_data_tflite(tflite_fname, params):
        """
        Generate random input data for the quantize/ dequantize / requantize ops,
        read TFLite's scale + zero-point (and optionally compute a Q31 multiplier+shift),
        then store them for downstream use.
        """
        tensors = {}
        effective_scales = {}
        scales = {}
        generated_params = {}
        aliases = {}
        
        softmax_input_integer_bits = params["input_integer_bits"]
        input_scale = params["input_scale"]
        input_real_multiplier = min(input_scale * (1 << (31 - softmax_input_integer_bits)), (1 << 31) - 1)
        input_multiplier, input_left_shift = Op_softmax.quantize_scale(input_real_multiplier)

        input_shape = (
            params["num_rows"],
            params["row_size"]
        )

        input_dtype_str = params["input_data_type"]

        input_dtype_tf = Lib.op_utils.get_tf_dtype(input_dtype_str)
        
        if input_dtype_tf == tf.float32:
            random_data = np.random.uniform(-1.0, 1.0, size=input_shape).astype(np.float32)
        elif input_dtype_tf == tf.int8:
            random_data = np.random.randint(-128, 128, size=input_shape).astype(np.int8)
        elif input_dtype_tf == tf.int16:
            random_data = np.random.randint(-32768, 32768, size=input_shape).astype(np.int16)
        else:
            raise ValueError(f"Unsupported input dtype: {input_dtype_tf}")

        tensors["input_tensor"] = random_data

        from tensorflow.lite.python.interpreter import Interpreter, OpResolverType
        interpreter = Interpreter(
            model_path=str(tflite_fname),
            experimental_op_resolver_type=OpResolverType.BUILTIN_REF
        )
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        (input_scale, input_zero_point) = input_details[0]['quantization']
        scales["input_scale"] = input_scale
        scales["input_zero_point"] = input_zero_point

        generated_params["input_mult"] = input_multiplier
        generated_params["input_left_shift"] = input_left_shift
        generated_params["diff_min"] = math.floor(((1 << softmax_input_integer_bits) - 1) * \
                            (1 << (31 - softmax_input_integer_bits)) / \
                            (1 << input_left_shift))
        generated_params["dst_size"] = params["num_rows"] * params["row_size"]
        
        return Lib.op_utils.Generated_data(
            generated_params,
            tensors,
            scales,
            effective_scales,
            aliases
        )

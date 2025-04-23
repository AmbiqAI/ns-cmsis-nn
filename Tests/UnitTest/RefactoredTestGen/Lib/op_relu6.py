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

def compute_multiplier_shift(scale):
    """
    Convert a float 'scale' into a Q31 'multiplier' and an integer 'shift'
    so that:
        int_val = (val * multiplier) * 2^shift / 2^31

    Used to test requantize ops
    """
    if scale == 0.0:
        return 0, 0

    significand, exponent = math.frexp(scale)

    q31 = round(significand * (1 << 31))

    # If rounding pushed q31 to 2^31, reduce it by factor of 2 and increase exponent
    if q31 == (1 << 31):
        q31 >>= 1
        exponent += 1

    # The net 'shift' is exponent - 31
    shift = exponent - 31
    multiplier = q31

    return multiplier, shift

class Op_relu6(Lib.op_utils.Op_type):

    def get_shapes(params):
        shapes = {}

        shapes["input_tensor"] = (
            params["batch_1"],
            params["height_1"],
            params["width_1"],
            params["channel_1"]
        )

        shapes["representational_dataset"] = (
            params["batch_size"],
            params["height_1"],
            params["width_1"],
            params["channel_1"]
        )

        shapes["different_in_shapes"] = False

        return shapes


    def generate_keras_model(shapes, params):
        tf.keras.backend.clear_session()

        input_1 = keras.layers.Input(batch_input_shape=shapes["input_tensor"])
        layer = tf.nn.relu6(input_1)
        model = keras.Model([input_1], [layer])
        return model



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

        input_shape = (
            params["batch_1"],
            params["height_1"],
            params["width_1"],
            params["channel_1"]
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

        generated_params["input_offset"] = input_zero_point

        output_details = interpreter.get_output_details()
        (output_scale, output_zero_point) = output_details[0]['quantization']
        scales["output_scale"] = output_scale
        scales["output_zero_point"] = output_zero_point
        generated_params["output_offset"] = output_zero_point

        if (input_dtype_tf == tf.int8):
            ratio = 1.0
            if output_scale != 0.0:  # just to avoid division by zero
                ratio = input_scale / output_scale

            multiplier, shift = compute_multiplier_shift(ratio)
            generated_params["output_multiplier"] = multiplier
            generated_params["output_shift"] = shift

        out_shape = output_details[0]["shape"]
        generated_params["output_len"] = int(np.prod(out_shape))

        return Lib.op_utils.Generated_data(
            generated_params,
            tensors,
            scales,
            effective_scales,
            aliases
        )

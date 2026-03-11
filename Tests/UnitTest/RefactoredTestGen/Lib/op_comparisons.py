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
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import Lib.op_utils
import tensorflow as tf
import numpy as np

from tensorflow.lite.python.interpreter import Interpreter
from tensorflow.lite.python.interpreter import OpResolverType
from Lib.keras_compat import keras


class Op_comparisons(Lib.op_utils.Op_type):

    _COMPARISON_MAP = {
        "equal": (tf.math.equal, "ARM_COMPARE_EQUAL"),
        "not_equal": (tf.math.not_equal, "ARM_COMPARE_NOT_EQUAL"),
        "greater": (tf.math.greater, "ARM_COMPARE_GREATER"),
        "greater_equal": (tf.math.greater_equal, "ARM_COMPARE_GREATER_EQUAL"),
        "less": (tf.math.less, "ARM_COMPARE_LESS"),
        "less_equal": (tf.math.less_equal, "ARM_COMPARE_LESS_EQUAL"),
    }

    @staticmethod
    def get_shapes(params):
        shapes = {}
        shapes["input_1_input_tensor"] = (
            params["input_1_n"],
            params["input_1_h"],
            params["input_1_w"],
            params["input_1_c"],
        )
        shapes["input_2_input_tensor"] = (
            params["input_2_n"],
            params["input_2_h"],
            params["input_2_w"],
            params["input_2_c"],
        )

        shapes["representational_dataset"] = shapes["input_1_input_tensor"]
        shapes["representational_dataset2"] = shapes["input_2_input_tensor"]
        shapes["different_in_shapes"] = True

        return shapes

    @staticmethod
    def generate_keras_model(shapes, params):
        tf.keras.backend.clear_session()

        comparison_type = params["comparison_type"]
        if comparison_type not in Op_comparisons._COMPARISON_MAP:
            raise ValueError(f"Unsupported comparison type '{comparison_type}'")

        comparison_op, _ = Op_comparisons._COMPARISON_MAP[comparison_type]

        input1 = keras.layers.Input(
            batch_input_shape=shapes["input_1_input_tensor"], name="input_1"
        )
        input2 = keras.layers.Input(
            batch_input_shape=shapes["input_2_input_tensor"], name="input_2"
        )

        comparison = comparison_op(input1, input2)
        comparison = tf.cast(comparison, tf.float32)
        model = keras.Model([input1, input2], comparison)

        return model

    @staticmethod
    def _compute_multipliers(params, lhs_scale, rhs_scale, output_scale):
        if params["input_data_type"] == "int16_t":
            left_shift = 15
        else:
            left_shift = 20

        twice_max_input_scale = 2 * max(lhs_scale, rhs_scale)
        lhs_multiplier = lhs_scale / twice_max_input_scale
        rhs_multiplier = rhs_scale / twice_max_input_scale
        # The output is only used for reference comparison. The scale is still needed to
        # align the intermediate quantized values.
        out_multiplier = twice_max_input_scale / ((1 << left_shift) * output_scale)

        lhs_mult, lhs_shift = Lib.op_utils.compute_multiplier_shift(lhs_multiplier)
        rhs_mult, rhs_shift = Lib.op_utils.compute_multiplier_shift(rhs_multiplier)
        out_mult, out_shift = Lib.op_utils.compute_multiplier_shift(out_multiplier)

        return left_shift, lhs_mult, lhs_shift, rhs_mult, rhs_shift, out_mult, out_shift

    @staticmethod
    def generate_data_tflite(tflite_fname, params):
        tensors = {}
        effective_scales = {}
        scales = {}
        generated_params = {}
        aliases = {}
        aliases["output"] = "output_ref"

        interpreter = Interpreter(
            str(tflite_fname), experimental_op_resolver_type=OpResolverType.BUILTIN_REF
        )
        interpreter.allocate_tensors()
        tensor_details = interpreter.get_tensor_details()

        input_1_details = tensor_details[0]
        input_2_details = tensor_details[1]

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        lhs_scale = input_details[0]["quantization_parameters"]["scales"][0]
        lhs_zp = input_details[0]["quantization_parameters"]["zero_points"][0]

        rhs_scale = input_details[1]["quantization_parameters"]["scales"][0]
        rhs_zp = input_details[1]["quantization_parameters"]["zero_points"][0]

        # Output is quantized back to the same dtype as inputs so the interpreter exposes scale/zp.
        out_scale = output_details[0]["quantization_parameters"]["scales"][0]
        out_zp = output_details[0]["quantization_parameters"]["zero_points"][0]

        (
            left_shift,
            input_1_mult,
            input_1_shift,
            input_2_mult,
            input_2_shift,
            out_mult,
            out_shift,
        ) = Op_comparisons._compute_multipliers(
            params, lhs_scale, rhs_scale, out_scale
        )

        tensors["input_1_input_tensor"] = interpreter.get_tensor(
            input_1_details["index"]
        )
        tensors["input_2_input_tensor"] = interpreter.get_tensor(
            input_2_details["index"]
        )

        output_shape = output_details[0]["shape"]
        generated_params["dst_size"] = int(np.prod(output_shape))
        generated_params["output_n"] = output_shape[0]
        generated_params["output_h"] = output_shape[1]
        generated_params["output_w"] = output_shape[2]
        generated_params["output_c"] = output_shape[3]

        generated_params["input_1_offset"] = -lhs_zp
        generated_params["input_1_mult"] = input_1_mult
        generated_params["input_1_shift"] = input_1_shift

        generated_params["input_2_offset"] = -rhs_zp
        generated_params["input_2_mult"] = input_2_mult
        generated_params["input_2_shift"] = input_2_shift

        generated_params["left_shift"] = left_shift
        generated_params["output_offset"] = out_zp
        generated_params["output_mult"] = out_mult
        generated_params["output_shift"] = out_shift
        generated_params["output_data_type"] = "bool"

        comparison_type = params["comparison_type"]
        _, enum_token = Op_comparisons._COMPARISON_MAP[comparison_type]
        generated_params["operation"] = enum_token

        return Lib.op_utils.Generated_data(
            generated_params, tensors, scales, effective_scales, aliases
        )

    @staticmethod
    def post_model_update(tflite_fname, generated_data, params):
        # Ensure the enum token is propagated back so it can be stored in config data.
        comparison_type = params["comparison_type"]
        _, enum_token = Op_comparisons._COMPARISON_MAP[comparison_type]
        generated_data.params["operation"] = enum_token
        # TFLite quantizes boolean outputs to full-range int8/int16 (0 or 127/32767).
        # Convert back to logical bool so the CMSIS-NN reference data matches kernel behaviour.
        if "output" in generated_data.tensors:
            output_tensor = np.asarray(generated_data.tensors["output"])
            generated_data.tensors["output"] = (output_tensor != 0)

        return generated_data

# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

from tensorflow.lite.python.interpreter import Interpreter
from tensorflow.lite.python.interpreter import OpResolverType
from Lib.keras_compat import keras


class Op_squared_difference(Lib.op_utils.Op_type):

    def get_shapes(params):
        shapes = {}
        shapes["lhs_input_tensor"] = (params["lhs_n"], params["lhs_h"], params["lhs_w"], params["lhs_c"])
        shapes["rhs_input_tensor"] = (params["rhs_n"], params["rhs_h"], params["rhs_w"], params["rhs_c"])
        shapes["representational_dataset"] = (params["lhs_n"], params["lhs_h"], params["lhs_w"], params["lhs_c"])
        shapes["representational_dataset2"] = (params["rhs_n"], params["rhs_h"], params["rhs_w"], params["rhs_c"])
        shapes["different_in_shapes"] = True

        return shapes

    def generate_keras_model(shapes, params):
        tf.keras.backend.clear_session()
        input_lhs = keras.layers.Input(batch_input_shape=shapes["lhs_input_tensor"], name="lhs")
        input_rhs = keras.layers.Input(batch_input_shape=shapes["rhs_input_tensor"], name="rhs")

        squared_diff = tf.keras.layers.Lambda(
            lambda x: tf.math.squared_difference(x[0], x[1]), name="lhs_squared_difference_rhs"
        )([input_lhs, input_rhs])

        model = tf.keras.Model(
            inputs=[input_lhs, input_rhs],
            outputs=squared_diff,
            name="simple_squared_difference_model"
        )

        return model

    def generate_data_tflite(tflite_fname, params):
        tensors = {}
        effective_scales = {}
        scales = {}
        generated_params = {}
        aliases = {}

        # To be removed
        aliases["output_multiplier"] = "output_mult"
        aliases["output"] = "output_ref"

        interpreter = Interpreter(str(tflite_fname), experimental_op_resolver_type=OpResolverType.BUILTIN_REF)
        interpreter.allocate_tensors()
        in_det = interpreter.get_input_details()
        out_det = interpreter.get_output_details()

        lhs_scale = in_det[0]["quantization_parameters"]["scales"][0]
        lhs_zp = in_det[0]["quantization_parameters"]["zero_points"][0]
        rhs_scale = in_det[1]["quantization_parameters"]["scales"][0]
        rhs_zp = in_det[1]["quantization_parameters"]["zero_points"][0]
        out_scale = out_det[0]["quantization_parameters"]["scales"][0]
        out_zp = out_det[0]["quantization_parameters"]["zero_points"][0]

        if params.get("input_data_type") == "int16_t":
            left_shift = 0
        else:
            left_shift = 7

        twice_max_input_scale = 2 * max(lhs_scale, rhs_scale)
        lhs_multiplier = lhs_scale / twice_max_input_scale
        rhs_multiplier = rhs_scale / twice_max_input_scale
        out_multiplier = (twice_max_input_scale * twice_max_input_scale) / (
            (1 << (left_shift * 2)) * out_scale
        )
        lhs_mult, lhs_shift = Lib.op_utils.compute_multiplier_shift(lhs_multiplier)
        rhs_mult, rhs_shift = Lib.op_utils.compute_multiplier_shift(rhs_multiplier)
        out_mult, out_shift = Lib.op_utils.compute_multiplier_shift(out_multiplier)

        (scales["lhs_scale"], scales["lhs_zero_point"]) = in_det[0]["quantization"]
        (scales["rhs_scale"], scales["rhs_zero_point"]) = in_det[1]["quantization"]
        (scales["output_scale"], scales["output_zero_point"]) = out_det[0]["quantization"]

        minval = Lib.op_utils.get_dtype_min(params["input_data_type"])
        maxval = Lib.op_utils.get_dtype_max(params["input_data_type"])

        n_output = out_det[0]["shape"][0]
        h_output = out_det[0]["shape"][1]
        w_output = out_det[0]["shape"][2]
        c_output = out_det[0]["shape"][3]

        generated_params["dst_size"] = n_output * h_output * w_output * c_output
        generated_params["output_n"] = n_output
        generated_params["output_h"] = h_output
        generated_params["output_w"] = w_output
        generated_params["output_c"] = c_output

        generated_params["lhs_offset"] = -lhs_zp
        generated_params["lhs_mult"] = lhs_mult
        generated_params["lhs_shift"] = lhs_shift
        generated_params["rhs_offset"] = -rhs_zp
        generated_params["rhs_mult"] = rhs_mult
        generated_params["rhs_shift"] = rhs_shift

        generated_params["input1_offset"] = -lhs_zp
        generated_params["input1_mult"] = lhs_mult
        generated_params["input1_shift"] = lhs_shift
        generated_params["input2_offset"] = -rhs_zp
        generated_params["input2_mult"] = rhs_mult
        generated_params["input2_shift"] = rhs_shift

        generated_params["output_offset"] = out_zp
        generated_params["output_mult"] = out_mult
        generated_params["output_shift"] = out_shift
        generated_params["output_multiplier"] = out_mult
        generated_params["left_shift"] = left_shift

        generated_params["activation_min"] = minval
        generated_params["activation_max"] = maxval

        return Lib.op_utils.Generated_data(generated_params, tensors, scales, effective_scales, aliases)

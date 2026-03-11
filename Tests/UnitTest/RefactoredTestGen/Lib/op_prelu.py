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
import numpy as np
import tensorflow as tf

from tensorflow.lite.python.interpreter import Interpreter
from tensorflow.lite.python.interpreter import OpResolverType
from Lib.keras_compat import keras

import Lib.op_utils


class Op_prelu(Lib.op_utils.Op_type):

    @staticmethod
    def get_shapes(params):
        shapes = {}
        shapes["input_tensor"] = (
            params["batch_size"],
            params["height_1"],
            params["width_1"],
            params["channel_1"],
        )
        shapes["alpha_input_tensor"] = (
            params["alpha_n"],
            params["alpha_h"],
            params["alpha_w"],
            params["alpha_c"],
        )
        shapes["representational_dataset"] = shapes["input_tensor"]
        shapes["representational_dataset2"] = shapes["alpha_input_tensor"]
        shapes["different_in_shapes"] = True

        return shapes

    @staticmethod
    def generate_keras_model(shapes, params):
        tf.keras.backend.clear_session()

        x = keras.layers.Input(batch_input_shape=shapes["input_tensor"], name="input_tensor")
        a = keras.layers.Input(batch_input_shape=shapes["alpha_input_tensor"], name="alpha_tensor")

        # y = x if x >= 0 else a * x
        y = keras.layers.Lambda(lambda t: tf.where(t[0] >= 0, t[0], t[0] * t[1]),
                                name="prelu_with_alpha")([x, a])

        model = keras.Model([x, a], y, name="prelu_with_alpha")
        return model

    @staticmethod
    def _quantize(data, scale, zero_point, dtype_str):
        if scale == 0.0:
            raise ValueError("Quantization scale cannot be zero.")
        qmin = Lib.op_utils.get_dtype_min(dtype_str)
        qmax = Lib.op_utils.get_dtype_max(dtype_str)
        np_dtype = Lib.op_utils.get_np_dtype(dtype_str)
        quantized = np.round(data / scale + zero_point)
        quantized = np.clip(quantized, qmin, qmax).astype(np_dtype)
        return quantized

    @staticmethod
    def generate_data_tflite(tflite_fname, params):
        tensors = {}
        effective_scales = {}
        scales = {}
        generated_params = {}
        aliases = {}

        interpreter = Interpreter(
            model_path=str(tflite_fname),
            experimental_op_resolver_type=OpResolverType.BUILTIN_REF,
        )
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        input_scale, input_zero_point = input_details[0]["quantization"]
        alpha_scale, alpha_zero_point = input_details[1]["quantization"]
        output_scale, output_zero_point = output_details[0]["quantization"]

        scales["input_scale"] = input_scale
        scales["input_zero_point"] = input_zero_point
        scales["alpha_scale"] = alpha_scale
        scales["alpha_zero_point"] = alpha_zero_point
        scales["output_scale"] = output_scale
        scales["output_zero_point"] = output_zero_point

        input_shape = (
            params["batch_size"],
            params["height_1"],
            params["width_1"],
            params["channel_1"],
        )
        alpha_shape = (
            params["alpha_n"],
            params["alpha_h"],
            params["alpha_w"],
            params["alpha_c"],
        )

        rng = np.random.default_rng()
        input_float = rng.uniform(-1.0, 1.0, size=input_shape).astype(np.float32)
        alpha_min = params.get("alpha_min", 0.01)
        alpha_max = params.get("alpha_max", 0.8)
        alpha_float = rng.uniform(alpha_min, alpha_max, size=alpha_shape).astype(
            np.float32
        )

        input_dtype = params["input_data_type"]
        alpha_dtype = params.get("alpha_data_type", params["input_data_type"])

        tensors["input_tensor"] = Op_prelu._quantize(
            input_float, input_scale, input_zero_point, input_dtype
        )
        tensors["alpha_input_tensor"] = Op_prelu._quantize(
            alpha_float, alpha_scale, alpha_zero_point, alpha_dtype
        )

        ratio_positive = 0.0 if output_scale == 0.0 else input_scale / output_scale
        ratio_negative = (
            0.0 if output_scale == 0.0 else (input_scale * alpha_scale) / output_scale
        )

        output_multiplier_1, output_shift_1 = Lib.op_utils.compute_multiplier_shift(
            ratio_positive
        )
        output_multiplier_2, output_shift_2 = Lib.op_utils.compute_multiplier_shift(
            ratio_negative
        )

        output_shape = output_details[0]["shape"]

        generated_params["input_offset"] = -input_zero_point
        generated_params["alpha_offset"] = -alpha_zero_point
        generated_params["output_offset"] = output_zero_point
        generated_params["output_multiplier_1"] = output_multiplier_1
        generated_params["output_shift_1"] = output_shift_1
        generated_params["output_multiplier_2"] = output_multiplier_2
        generated_params["output_shift_2"] = output_shift_2
        generated_params["input_n"] = input_shape[0]
        generated_params["input_h"] = input_shape[1]
        generated_params["input_w"] = input_shape[2]
        generated_params["input_c"] = input_shape[3]
        generated_params["alpha_n"] = alpha_shape[0]
        generated_params["alpha_h"] = alpha_shape[1]
        generated_params["alpha_w"] = alpha_shape[2]
        generated_params["alpha_c"] = alpha_shape[3]
        generated_params["output_n"] = int(output_shape[0])
        generated_params["output_h"] = int(output_shape[1])
        generated_params["output_w"] = int(output_shape[2])
        generated_params["output_c"] = int(output_shape[3])
        generated_params["dst_size"] = int(np.prod(output_shape))
        generated_params["output_len"] = generated_params["dst_size"]

        return Lib.op_utils.Generated_data(
            generated_params,
            tensors,
            scales,
            effective_scales,
            aliases,
        )

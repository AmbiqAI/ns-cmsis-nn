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
from Lib.keras_compat import keras


class Op_resize_nearest_neighbor(Lib.op_utils.Op_type):

    @staticmethod
    def get_shapes(params):
        shapes = {}
        shapes["input_tensor"] = (params["input_n"], params["input_h"], params["input_w"], params["input_c"])
        shapes["representational_dataset"] = shapes["input_tensor"]
        return shapes

    @staticmethod
    def generate_keras_model(shapes, params):
        input_tensor = keras.layers.Input(batch_input_shape=shapes["input_tensor"], name="input_tensor")
        output_tensor = keras.layers.Lambda(
            lambda t: tf.raw_ops.ResizeNearestNeighbor(
                images=t,
                size=[params["output_h"], params["output_w"]],
                align_corners=params.get("align_corners", False),
                half_pixel_centers=params.get("half_pixel_centers", False),
            ),
            name="resize_nearest_neighbor",
        )(input_tensor)
        return keras.Model(inputs=input_tensor, outputs=output_tensor)

    @staticmethod
    def generate_data_tflite(tflite_fname, params):
        del tflite_fname

        tensors = {}
        effective_scales = {}
        scales = {}

        input_n = params["input_n"]
        input_h = params["input_h"]
        input_w = params["input_w"]
        input_c = params["input_c"]
        output_h = params["output_h"]
        output_w = params["output_w"]

        output_n = input_n
        output_c = input_c

        generated_params = {
            "input_n": input_n,
            "input_h": input_h,
            "input_w": input_w,
            "input_c": input_c,
            "output_n": output_n,
            "output_h": output_h,
            "output_w": output_w,
            "output_c": output_c,
            "output_size": output_n * output_h * output_w * output_c,
            "output_size_n": 1,
            "output_size_h": 1,
            "output_size_w": 1,
            "output_size_c": 2,
            "output_size_data": [output_h, output_w],
        }

        return Lib.op_utils.Generated_data(generated_params, tensors, scales, effective_scales)

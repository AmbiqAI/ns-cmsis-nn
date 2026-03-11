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
import numpy as np

from tensorflow.lite.python.interpreter import Interpreter
from tensorflow.lite.python.interpreter import OpResolverType
from Lib.keras_compat import keras

class Op_strided_slice(Lib.op_utils.Op_type):
    """
    Generate and capture data for the StridedSlice operator.
    Expects params to contain:
      - batch_size, input_h, input_w, input_c   : input dims
      - begin   : list/tuple of 4 ints
      - end     : list/tuple of 4 ints
      - strides : list/tuple of 4 ints
    """

    @staticmethod
    def get_shapes(params):
        # we take a 4-D NHWC tensor
        shapes = {}
        shapes["input_tensor"] = (
            params["batch_size"],
            params["input_h"],
            params["input_w"],
            params["input_c"],
        )
        # TFLite static‐op tests often use the input as the rep‐dataset
        shapes["representational_dataset"] = shapes["input_tensor"]
        return shapes

    @staticmethod
    def generate_keras_model(shapes, params):
        inp_shape = shapes["input_tensor"]
        if params["input_data_type"] == "int32_t":
            x = keras.Input(shape=inp_shape[1:], batch_size=inp_shape[0], name="input", dtype=tf.int32)
        else:
            x = keras.Input(shape=inp_shape[1:], batch_size=inp_shape[0], name="input")
        y = tf.keras.layers.Lambda(
            lambda t: tf.strided_slice(
                t,
                begin=params["begin"],
                end=params["end"],
                strides=params["strides"],
                begin_mask=0,
                end_mask=0,
                ellipsis_mask=0,
                new_axis_mask=0,
                shrink_axis_mask=0,
            ),
            name="strided_slice",
        )(x)
        model = keras.Model(inputs=x, outputs=y)
        return model

    @staticmethod
    def generate_data_tflite(tflite_fname, params):
        input_shape = (
            params["batch_size"],
            params["input_h"],
            params["input_w"],
            params["input_c"],
        )
        input_dtype = params["input_data_type"]
        input_np_dtype = Lib.op_utils.get_np_dtype(input_dtype)
        input_max = Lib.op_utils.get_dtype_max(input_dtype)
        dtype_info = np.iinfo(input_np_dtype)
        high = input_max + 1 if input_max < dtype_info.max else input_max
        inp_data = np.random.randint(
            low=0,
            high=high,
            size=input_shape,
            dtype=input_np_dtype,
        )

        # run the TFLite model
        interpreter = Interpreter(
            model_path=str(tflite_fname),
            experimental_op_resolver_type=OpResolverType.BUILTIN_REF,
        )
        interpreter.allocate_tensors()
        inp_detail = interpreter.get_input_details()[0]
        interpreter.set_tensor(inp_detail["index"], inp_data)
        interpreter.invoke()
        out_detail = interpreter.get_output_details()[0]
        out_data = interpreter.get_tensor(out_detail["index"])

        # collect params & shapes for the C test harness
        generated_params = {
            "input_batches": params["batch_size"],
            "input_h": params["input_h"],
            "input_w": params["input_w"],
            "input_c": params["input_c"],
            "begin_n": params["begin"][0],
            "begin_h": params["begin"][1],
            "begin_w": params["begin"][2],
            "begin_c": params["begin"][3],
            "end_n": params["end"][0],
            "end_h": params["end"][1],
            "end_w": params["end"][2],
            "end_c": params["end"][3],
            "strides_n": params["strides"][0],
            "strides_h": params["strides"][1],
            "strides_w": params["strides"][2],
            "strides_c": params["strides"][3],
            "output_size": out_data.size,
            "output_n":  out_data.shape[0],
            "output_h":  out_data.shape[1],
            "output_w":  out_data.shape[2],
            "output_c":  out_data.shape[3],
        }
        # tensors maps the names your C harness will expect
        tensors = {
            "input_tensor": inp_data,
        }
        scales = {}              # no quant here
        effective_scales = {}    # bare‐float testing

        return Lib.op_utils.Generated_data(
            generated_params, tensors, scales, effective_scales
        )

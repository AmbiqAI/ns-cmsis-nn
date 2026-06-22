# SPDX-FileCopyrightText: Copyright 2026 Ambiq
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import tensorflow as tf
import tf_keras as keras

import Lib.op_utils


def _broadcast_strides(input_shape, output_shape):
    rank = len(output_shape)
    padded_shape = [1] * (rank - len(input_shape)) + list(input_shape)
    element_strides = [1] * rank
    for index in range(rank - 2, -1, -1):
        element_strides[index] = element_strides[index + 1] * padded_shape[index + 1]

    return [0 if dim == 1 else stride for dim, stride in zip(padded_shape, element_strides)]


class Op_select_v2(Lib.op_utils.Op_type):

    @staticmethod
    def get_shapes(params):
        return {
            "condition_input_tensor": tuple(params["condition_shape"]),
            "y_input_tensor": tuple(params["y_shape"]),
            "x_input_tensor": tuple(params["x_shape"]),
        }

    @staticmethod
    def generate_keras_model(shapes, params):
        tf.keras.backend.clear_session()

        condition_dtype = Lib.op_utils.get_tf_dtype(params.get("condition_data_type", "bool"))
        data_dtype = Lib.op_utils.get_tf_dtype(params["input_data_type"])

        condition = keras.layers.Input(batch_input_shape=shapes["condition_input_tensor"], dtype=condition_dtype, name="condition")
        x = keras.layers.Input(batch_input_shape=shapes["x_input_tensor"], dtype=data_dtype, name="x")
        y = keras.layers.Input(batch_input_shape=shapes["y_input_tensor"], dtype=data_dtype, name="y")
        output = keras.layers.Lambda(
            lambda tensors: tf.raw_ops.SelectV2(condition=tensors[0], t=tensors[1], e=tensors[2]),
            name="select_v2",
        )([condition, x, y])

        return keras.Model(inputs=[condition, x, y], outputs=[output])

    @staticmethod
    def generate_data_tflite(tflite_fname, params):
        del tflite_fname

        condition_shape = tuple(params["condition_shape"])
        x_shape = tuple(params["x_shape"])
        y_shape = tuple(params["y_shape"])
        output_shape = tuple(params["output_shape"])
        data_dtype = Lib.op_utils.get_np_dtype(params["input_data_type"])
        condition_dtype = Lib.op_utils.get_np_dtype(params.get("condition_data_type", "bool"))

        condition = np.asarray(params["condition_values"], dtype=condition_dtype).reshape(condition_shape)
        x = np.asarray(params["x_values"], dtype=data_dtype).reshape(x_shape)
        y = np.asarray(params["y_values"], dtype=data_dtype).reshape(y_shape)

        generated_params = {
            "rank": len(output_shape),
            "output_size": int(np.prod(output_shape)),
            "output_shape": list(output_shape),
            "cond_strides": _broadcast_strides(condition_shape, output_shape),
            "x_strides": _broadcast_strides(x_shape, output_shape),
            "y_strides": _broadcast_strides(y_shape, output_shape),
        }
        tensors = {
            "condition_input_tensor": condition,
            "x_input_tensor": x,
            "y_input_tensor": y,
        }

        return Lib.op_utils.Generated_data(generated_params, tensors, {}, {})
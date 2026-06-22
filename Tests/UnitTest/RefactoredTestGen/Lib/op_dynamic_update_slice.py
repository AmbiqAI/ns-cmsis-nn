# SPDX-FileCopyrightText: Copyright 2026 Ambiq
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import tensorflow as tf
import tf_keras as keras
from tensorflow.compiler.tf2xla.python import xla

import Lib.op_utils


def _row_major_strides(shape):
    strides = [1] * len(shape)
    for index in range(len(shape) - 2, -1, -1):
        strides[index] = strides[index + 1] * shape[index + 1]
    return strides


class Op_dynamic_update_slice(Lib.op_utils.Op_type):

    @staticmethod
    def get_shapes(params):
        operand_shape = (1,) + tuple(params["operand_shape"])
        update_shape = (1,) + tuple(params["update_shape"])
        shapes = {
            "update_input_tensor": update_shape,
            "operand_input_tensor": operand_shape,
            "representational_dataset": operand_shape,
            "representational_dataset2": update_shape,
            "different_in_shapes": True,
        }
        return shapes

    @staticmethod
    def generate_keras_model(shapes, params):
        tf.keras.backend.clear_session()

        operand = keras.layers.Input(batch_input_shape=shapes["operand_input_tensor"], name="operand")
        update = keras.layers.Input(batch_input_shape=shapes["update_input_tensor"], name="update")
        start_indices = [0] + [int(index) for index in params["start_indices"]]
        start_indices_tensor = tf.constant(start_indices, dtype=tf.int32)

        output = keras.layers.Lambda(
            lambda tensors: xla.dynamic_update_slice(tensors[0], tensors[1], start_indices_tensor),
            name="dynamic_update_slice",
        )([operand, update])

        return keras.Model(inputs=[operand, update], outputs=[output])

    @staticmethod
    def generate_data_tflite(tflite_fname, params):
        del tflite_fname

        operand_shape = tuple(params["operand_shape"])
        update_shape = tuple(params["update_shape"])
        start_indices = np.asarray(params["start_indices"], dtype=np.int32)
        dtype = Lib.op_utils.get_np_dtype(params["input_data_type"])

        operand = np.asarray(params["operand_values"], dtype=dtype).reshape((1,) + operand_shape)
        update = np.asarray(params["update_values"], dtype=dtype).reshape((1,) + update_shape)
        generated_params = {
            "rank": len(operand_shape),
            "operand_size": int(operand.size),
            "update_size": int(update.size),
            "operand_shape": list(operand_shape),
            "update_shape": list(update_shape),
            "operand_strides": _row_major_strides(list(operand_shape)),
        }
        tensors = {
            "operand_input_tensor": operand,
            "update_input_tensor": update,
            "start_indices": start_indices,
        }

        return Lib.op_utils.Generated_data(generated_params, tensors, {}, {})
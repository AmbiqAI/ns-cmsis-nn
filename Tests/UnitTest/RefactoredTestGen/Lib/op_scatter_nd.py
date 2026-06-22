# SPDX-FileCopyrightText: Copyright 2026 Ambiq
#
# SPDX-License-Identifier: Apache-2.0

import math
import numpy as np
import tensorflow as tf
import tf_keras as keras

import Lib.op_utils


def _row_major_strides(shape):
    strides = [1] * len(shape)
    for index in range(len(shape) - 2, -1, -1):
        strides[index] = strides[index + 1] * shape[index + 1]
    return strides


class Op_scatter_nd(Lib.op_utils.Op_type):

    @staticmethod
    def get_shapes(params):
        output_shape = tuple(params["output_shape"])
        index_depth = int(params["index_depth"])
        num_updates = int(params["num_updates"])
        slice_shape = output_shape[index_depth:]
        updates_shape = (num_updates,) + slice_shape
        return {"updates_input_tensor": updates_shape}

    @staticmethod
    def generate_keras_model(shapes, params):
        tf.keras.backend.clear_session()

        updates = keras.layers.Input(batch_input_shape=shapes["updates_input_tensor"], name="updates")
        indices = tf.constant(params["indices_values"], dtype=tf.int32, shape=[params["num_updates"], params["index_depth"]])
        output_shape = tf.constant(params["output_shape"], dtype=tf.int32)
        output = tf.scatter_nd(indices, updates, output_shape)

        return keras.Model(inputs=[updates], outputs=[output])

    @staticmethod
    def generate_data_tflite(tflite_fname, params):
        del tflite_fname

        output_shape = tuple(params["output_shape"])
        index_depth = int(params["index_depth"])
        num_updates = int(params["num_updates"])
        dtype = Lib.op_utils.get_np_dtype(params["input_data_type"])

        indices = np.asarray(params["indices_values"], dtype=np.int32).reshape(num_updates, index_depth)
        slice_shape = output_shape[index_depth:]
        slice_size = int(math.prod(slice_shape)) if slice_shape else 1
        updates_shape = (num_updates,) + slice_shape
        updates = np.asarray(params["updates_values"], dtype=dtype).reshape(updates_shape)

        generated_params = {
            "num_updates": num_updates,
            "index_depth": index_depth,
            "slice_size": slice_size,
            "output_size": int(np.prod(output_shape)),
            "output_shape": list(output_shape),
            "output_strides": _row_major_strides(list(output_shape))[:index_depth],
        }

        return Lib.op_utils.Generated_data(generated_params, {"updates_input_tensor": updates, "indices": indices}, {}, {})
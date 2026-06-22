# SPDX-FileCopyrightText: Copyright 2026 Ambiq
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import tensorflow as tf
import tf_keras as keras

import Lib.op_utils


class Op_where(Lib.op_utils.Op_type):

    @staticmethod
    def get_shapes(params):
        return {"condition_input_tensor": tuple(params["shape"])}

    @staticmethod
    def generate_keras_model(shapes, params):
        tf.keras.backend.clear_session()

        del params

        condition = keras.layers.Input(batch_input_shape=shapes["condition_input_tensor"], dtype=tf.bool)
        output = keras.layers.Lambda(
            lambda tensor: tf.raw_ops.Where(condition=tensor),
            name="where",
        )(condition)

        return keras.Model(inputs=[condition], outputs=[output])

    @staticmethod
    def generate_data_tflite(tflite_fname, params):
        del tflite_fname

        condition_shape = tuple(params["shape"])
        condition_dtype = Lib.op_utils.get_np_dtype(params.get("condition_data_type", params["input_data_type"]))
        condition = np.asarray(params["condition_values"], dtype=condition_dtype).reshape(condition_shape)

        generated_params = {
            "rank": len(condition_shape),
            "size": int(condition.size),
            "cond_size": int(condition.size),
            "shape": list(condition_shape),
            "output_data_type": "int64_t",
        }

        return Lib.op_utils.Generated_data(generated_params, {"condition_input_tensor": condition}, {}, {})
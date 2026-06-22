# SPDX-FileCopyrightText: Copyright 2026 Ambiq
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import tensorflow as tf
import tf_keras as keras

import Lib.op_utils


class Op_tile(Lib.op_utils.Op_type):

    @staticmethod
    def get_shapes(params):
        return {"input_tensor": tuple(params["input_shape"])}

    @staticmethod
    def generate_keras_model(shapes, params):
        tf.keras.backend.clear_session()

        input_tensor = keras.layers.Input(batch_input_shape=shapes["input_tensor"])
        output_tensor = tf.tile(input_tensor, params["multiples"])

        return keras.Model(inputs=[input_tensor], outputs=[output_tensor])

    @staticmethod
    def generate_data_tflite(tflite_fname, params):
        del tflite_fname

        input_shape = tuple(params["input_shape"])
        multiples = list(params["multiples"])
        output_shape = [dimension * multiple for dimension, multiple in zip(input_shape, multiples)]
        dtype = Lib.op_utils.get_np_dtype(params["input_data_type"])
        input_tensor = np.asarray(params["input_values"], dtype=dtype).reshape(input_shape)

        generated_params = {
            "rank": len(input_shape),
            "input_size": int(input_tensor.size),
            "output_size": int(np.prod(output_shape)),
            "input_shape": list(input_shape),
            "multiples": multiples,
        }

        return Lib.op_utils.Generated_data(generated_params, {"input_tensor": input_tensor}, {}, {})
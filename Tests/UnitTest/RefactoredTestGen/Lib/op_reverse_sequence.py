# SPDX-FileCopyrightText: Copyright 2026 Ambiq
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import tensorflow as tf
import tf_keras as keras

import Lib.op_utils


class Op_reverse_sequence(Lib.op_utils.Op_type):

    @staticmethod
    def get_shapes(params):
        return {"input_tensor": tuple(params["shape"])}

    @staticmethod
    def generate_keras_model(shapes, params):
        tf.keras.backend.clear_session()

        input_tensor = keras.layers.Input(batch_input_shape=shapes["input_tensor"])
        seq_lengths = tf.constant(params["seq_lengths"], dtype=tf.int32)
        output_tensor = tf.reverse_sequence(
            input_tensor,
            seq_lengths=seq_lengths,
            seq_axis=int(params["seq_dim"]),
            batch_axis=int(params["batch_dim"]),
        )

        return keras.Model(inputs=[input_tensor], outputs=[output_tensor])

    @staticmethod
    def generate_data_tflite(tflite_fname, params):
        del tflite_fname

        input_shape = tuple(params["shape"])
        seq_lengths = np.asarray(params["seq_lengths"], dtype=np.int32)
        dtype = Lib.op_utils.get_np_dtype(params["input_data_type"])
        input_tensor = np.asarray(params["input_values"], dtype=dtype).reshape(input_shape)

        generated_params = {
            "rank": len(input_shape),
            "size": int(input_tensor.size),
            "shape": list(input_shape),
            "seq_dim": int(params["seq_dim"]),
            "batch_dim": int(params["batch_dim"]),
        }
        tensors = {
            "input_tensor": input_tensor,
            "seq_lengths": seq_lengths,
        }

        return Lib.op_utils.Generated_data(generated_params, tensors, {}, {})
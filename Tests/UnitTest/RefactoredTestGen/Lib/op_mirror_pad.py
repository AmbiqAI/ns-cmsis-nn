# SPDX-FileCopyrightText: Copyright 2026 Ambiq
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import tensorflow as tf
import tf_keras as keras

import Lib.op_utils


class Op_mirror_pad(Lib.op_utils.Op_type):

    @staticmethod
    def get_shapes(params):
        return {"input_tensor": tuple(params["input_shape"])}

    @staticmethod
    def generate_keras_model(shapes, params):
        tf.keras.backend.clear_session()

        input_tensor = keras.layers.Input(batch_input_shape=shapes["input_tensor"])
        pad_before = list(params["pad_before"])
        pad_after = list(params.get("pad_after", pad_before))
        paddings = tf.constant([[before, after] for before, after in zip(pad_before, pad_after)], dtype=tf.int32)
        output_tensor = keras.layers.Lambda(
            lambda tensor: tf.raw_ops.MirrorPad(input=tensor, paddings=paddings, mode="REFLECT"),
            name="mirror_pad",
        )(input_tensor)

        return keras.Model(inputs=[input_tensor], outputs=[output_tensor])

    @staticmethod
    def generate_data_tflite(tflite_fname, params):
        del tflite_fname

        input_shape = tuple(params["input_shape"])
        pad_before = list(params["pad_before"])
        pad_after = list(params.get("pad_after", pad_before))
        output_shape = [dimension + before + after for dimension, before, after in zip(input_shape, pad_before, pad_after)]
        dtype = Lib.op_utils.get_np_dtype(params["input_data_type"])

        input_tensor = np.asarray(params["input_values"], dtype=dtype).reshape(input_shape)
        generated_params = {
            "rank": len(input_shape),
            "mode": int(params.get("mode", 0)),
            "input_size": int(input_tensor.size),
            "output_size": int(np.prod(output_shape)),
            "input_shape": list(input_shape),
            "output_shape": output_shape,
            "pad_before": pad_before,
        }

        return Lib.op_utils.Generated_data(generated_params, {"input_tensor": input_tensor}, {}, {})
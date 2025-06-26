# SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache License, Version 2.0
#
# www.apache.org/licenses/LICENSE-2.0
#
# Licensed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
#

import math
import numpy as np
import copy
import tensorflow as tf
import tf_keras as keras
import Lib.op_utils

from tensorflow.lite.python.interpreter import Interpreter
from tensorflow.lite.python.interpreter import OpResolverType


def compute_multiplier_shift(scale):
    """
    Convert a float 'scale' into a Q31 'multiplier' and an integer 'shift'
    so that:
        int_val = (val * multiplier) * 2^shift / 2^31

    Used to test requantize ops
    """
    if scale == 0.0:
        return 0, 0

    significand, exponent = math.frexp(scale)

    # q31 = round(significand * (1 << 31))
    q31 = int(math.floor(significand * (1 << 31) + 0.5))

    # If rounding pushed q31 to 2^31, reduce it by factor of 2 and increase exponent
    if q31 == (1 << 31):
        q31 //= 2
        exponent += 1

    shift = exponent
    multiplier = q31

    return multiplier, shift


class Op_reduce_max(Lib.op_utils.Op_type):

    def get_shapes(params):
        shapes = {}
        input_shape = copy.deepcopy(params["in_dim"])
        shapes["input_tensor"] = input_shape
        shapes["representational_dataset"] = input_shape
        return shapes

    def generate_keras_model(shapes, params):
        input_shape = shapes["input_tensor"]
        axis = params["axis"]
        keepdims = params.get("keepdims", True)

        input_tensor = keras.layers.Input(batch_input_shape=input_shape)
        output_tensor = tf.math.reduce_max(input_tensor, axis=axis, keepdims=keepdims)
        model = keras.Model(inputs=[input_tensor], outputs=[output_tensor])

        return model

    def generate_data_tflite(tflite_fname, params):
        tensors = {}
        effective_scales = {}
        scales = {}
        generated_params = {}
        aliases = {}

        axis = sorted(params["axis"])
        keepdims = params.get("keepdims", True)
        input_shape = params["in_dim"]
        input_rank = len(input_shape)

        # Derive output shape
        if keepdims:
            output_shape = [1 if i in axis else input_shape[i] for i in range(input_rank)]
        else:
            output_shape = [input_shape[i] for i in range(input_rank) if i not in axis]

        # Pad both to 4D
        padded_input = input_shape + [1] * (4 - len(input_shape))
        padded_output = output_shape + [1] * (4 - len(output_shape))

        # Axis dimension mask for [n,h,w,c]
        axis_dim = [1 if i in axis else 0 for i in range(4)]
        axis_mask = sum([1 << (3 - i) for i in axis])

        generated_params["in_dim"] = padded_input
        generated_params["out_dim"] = padded_output
        generated_params["axis"] = axis
        generated_params["axis_dim"] = axis_dim
        generated_params["axis_mask"] = axis_mask
        generated_params["input_size"] = int(np.prod(padded_input))
        generated_params["output_size"] = int(np.prod(padded_output))

        # Quantization extraction from TFLite
        interpreter = Interpreter(str(tflite_fname), experimental_op_resolver_type=OpResolverType.BUILTIN_REF)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]

        input_scale, input_zp = input_details['quantization']
        output_scale, output_zp = output_details['quantization']

        scales["input_scale"] = input_scale
        scales["input_zero_point"] = input_zp
        scales["output_scale"] = output_scale
        scales["output_zero_point"] = output_zp

        generated_params["input_offset"] = -input_zp
        generated_params["output_offset"] = output_zp

        # Compute the number of elements being reduced
        reduction_size = 1
        for i in axis:
            reduction_size *= input_shape[i]

        # Compute base scale and quantize
        real_multiplier = input_scale / output_scale
        base_mult, base_shift = compute_multiplier_shift(real_multiplier)
        print(f"Input scale: {input_scale}, Output scale: {output_scale}")
        print(f"Computed multiplier: {base_mult}, shift: {base_shift} for scale: {real_multiplier}")
        real_multiplier = input_scale / output_scale
        out_mult, out_shift = compute_multiplier_shift(real_multiplier)
        generated_params["output_multiplier"] = out_mult
        generated_params["output_shift"] = out_shift
        generated_params["reduction_size"] = reduction_size  # Optional for debugging

        return Lib.op_utils.Generated_data(generated_params, tensors, scales, effective_scales, aliases)

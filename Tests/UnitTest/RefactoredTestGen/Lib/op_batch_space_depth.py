# SPDX-FileCopyrightText: Copyright 2024 Arm
# SPDX-License-Identifier: Apache-2.0

import copy
import numpy as np
import tensorflow as tf
from Lib.keras_compat import keras
import Lib.op_utils
from tensorflow.lite.python.interpreter import Interpreter
from tensorflow.lite.python.interpreter import OpResolverType


def _pad_to_4d(shape):
    return shape + [1] * (4 - len(shape))


def _extract_io_scales(tflite_fname):
    interpreter = Interpreter(str(tflite_fname), experimental_op_resolver_type=OpResolverType.BUILTIN_REF)
    interpreter.allocate_tensors()
    in_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]

    in_scale, in_zp = in_det['quantization']
    out_scale, out_zp = out_det['quantization']

    scales = {
        "input_scale": in_scale,
        "input_zero_point": in_zp,
        "output_scale": out_scale,
        "output_zero_point": out_zp,
    }
    gen = {
        "input_offset": -in_zp,
        "output_offset": out_zp,
    }
    return scales, gen


# ----------------------------- SPACE_TO_DEPTH -----------------------------

class Op_space_to_depth(Lib.op_utils.Op_type):

    def get_shapes(params):
        shapes = {}
        shapes["input_tensor"] = copy.deepcopy(params["in_dim"])
        shapes["representational_dataset"] = shapes["input_tensor"]
        return shapes

    def generate_keras_model(shapes, params):
        input_shape = shapes["input_tensor"]
        bs = int(params["block_size"])
        x = keras.layers.Input(batch_input_shape=input_shape)
        y = tf.nn.space_to_depth(x, block_size=bs, data_format="NHWC")
        return keras.Model(inputs=[x], outputs=[y])

    def generate_data_tflite(tflite_fname, params):
        tensors = {}
        effective_scales = {}
        scales = {}
        generated_params = {}
        aliases = {}

        in_shape = list(params["in_dim"])
        bs = int(params["block_size"])

        # Derived out shape (NHWC)
        N, H, W, C = in_shape
        assert H % bs == 0 and W % bs == 0, "H and W must be divisible by block_size"
        OH, OW, OC = H // bs, W // bs, C * bs * bs
        out_shape = [N, OH, OW, OC]

        # 4D padding (kept for consistency with your framework)
        generated_params["in_dim"] = _pad_to_4d(in_shape)
        generated_params["out_dim"] = _pad_to_4d(out_shape)
        generated_params["block_size"] = bs
        generated_params["input_size"] = int(np.prod(in_shape))
        generated_params["output_size"] = int(np.prod(out_shape))

        scales, qo = _extract_io_scales(tflite_fname)
        generated_params.update(qo)

        return Lib.op_utils.Generated_data(generated_params, tensors, scales, effective_scales, aliases)


# ----------------------------- DEPTH_TO_SPACE -----------------------------

class Op_depth_to_space(Lib.op_utils.Op_type):

    def get_shapes(params):
        shapes = {}
        shapes["input_tensor"] = copy.deepcopy(params["in_dim"])
        shapes["representational_dataset"] = shapes["input_tensor"]
        return shapes

    def generate_keras_model(shapes, params):
        input_shape = shapes["input_tensor"]
        bs = int(params["block_size"])
        x = keras.layers.Input(batch_input_shape=input_shape)
        y = tf.nn.depth_to_space(x, block_size=bs, data_format="NHWC")
        return keras.Model(inputs=[x], outputs=[y])

    def generate_data_tflite(tflite_fname, params):
        tensors = {}
        effective_scales = {}
        scales = {}
        generated_params = {}
        aliases = {}

        in_shape = list(params["in_dim"])
        bs = int(params["block_size"])

        # Derived out shape (NHWC)
        N, H, W, C = in_shape
        denom = bs * bs
        assert C % denom == 0, "C must be divisible by block_size^2"
        OH, OW, OC = H * bs, W * bs, C // denom
        out_shape = [N, OH, OW, OC]

        generated_params["in_dim"] = _pad_to_4d(in_shape)
        generated_params["out_dim"] = _pad_to_4d(out_shape)
        generated_params["block_size"] = bs
        generated_params["input_size"] = int(np.prod(in_shape))
        generated_params["output_size"] = int(np.prod(out_shape))

        # scales, qo = _extract_io_scales(tflite_fname)
        # generated_params.update(qo)

        return Lib.op_utils.Generated_data(generated_params, tensors, scales, effective_scales, aliases)


# ----------------------------- SPACE_TO_BATCH_ND -----------------------------

class Op_space_to_batch_nd(Lib.op_utils.Op_type):

    def get_shapes(params):
        shapes = {}
        input_shape = copy.deepcopy(params["in_dim"])
        shapes["input_tensor"] = input_shape
        shapes["representational_dataset"] = input_shape
        return shapes

    def generate_keras_model(shapes, params):
        in_shape = shapes["input_tensor"]
        bh, bw = map(int, params["block_shape"])
        top, left, bottom, right = map(int, params["paddings"])

        # TF expects H/W 2×2: [[top, bottom], [left, right]]
        paddings_hw = tf.constant([[int(top), int(bottom)], [int(left), int(right)]], dtype=tf.int32)
        block_hw = tf.constant([int(bh), int(bw)], dtype=tf.int32)

        x = keras.layers.Input(batch_input_shape=in_shape)
        y = tf.space_to_batch(x, block_shape=block_hw, paddings=paddings_hw)
        model = keras.Model(inputs=[x], outputs=[y])
        return model

    def generate_data_tflite(tflite_fname, params):
        tensors = {}
        effective_scales = {}
        generated_params = {}
        aliases = {}

        N, H, W, C = params["in_dim"]
        bh, bw = map(int, params["block_shape"])
        top, left, bottom, right = map(int, params["paddings"])

        # Compute output dims
        H_pad = H + top + bottom
        W_pad = W + left + right

        if H_pad % bh != 0 or W_pad % bw != 0:
            raise ValueError(
                f"SPACE_TO_BATCH_ND: padded H/W ({H_pad},{W_pad}) not divisible by block ({bh},{bw})"
            )

        out_N = N * bh * bw
        out_H = H_pad // bh
        out_W = W_pad // bw
        out_C = C

        generated_params["in_dim"] = [N, H, W, C]
        generated_params["out_dim"] = [out_N, out_H, out_W, out_C]
        generated_params["block_shape"] = [bh, bw]

        # For your C API: cmsis_nn_dims used as (n,h,w,c) → (top,left,bottom,right)
        generated_params["pad_nhwc4"] = [top, left, bottom, right]

        generated_params["input_size"] = int(np.prod([N, H, W, C]))
        generated_params["output_size"] = int(np.prod([out_N, out_H, out_W, out_C]))

        # Quantization (pass-through op; still record what converter gives)
        scales, qo = _extract_io_scales(tflite_fname)
        generated_params.update(qo)
        return Lib.op_utils.Generated_data(generated_params, tensors, scales, effective_scales, aliases)

# ----------------------------- BATCH_TO_SPACE_ND -----------------------------

class Op_batch_to_space_nd(Lib.op_utils.Op_type):

    def get_shapes(params):
        shapes = {}
        input_shape = copy.deepcopy(params["in_dim"])
        shapes["input_tensor"] = input_shape
        shapes["representational_dataset"] = input_shape
        return shapes

    def generate_keras_model(shapes, params):
        in_shape = shapes["input_tensor"]

        bh, bw = map(int, params["block_shape"])
        top, left, bottom, right = map(int, params["crops"])

        # TF expects H/W 2×2: [[top, bottom], [left, right]]
        crops_hw = tf.constant([[int(top), int(bottom)], [int(left), int(right)]], dtype=tf.int32)
        block_hw = tf.constant([int(bh), int(bw)], dtype=tf.int32)

        x = keras.layers.Input(batch_input_shape=in_shape)
        y = tf.batch_to_space(x, block_shape=block_hw, crops=crops_hw)
        model = keras.Model(inputs=[x], outputs=[y])
        return model

    def generate_data_tflite(tflite_fname, params):
        tensors = {}
        effective_scales = {}
        generated_params = {}
        aliases = {}

        N, H, W, C = params["in_dim"]
        bh, bw = map(int, params["block_shape"])
        top, left, bottom, right = map(int, params["crops"])

        if (N % (bh * bw)) != 0:
            raise ValueError(
                f"BATCH_TO_SPACE_ND: input N={N} not divisible by bh*bw={bh*bw}"
            )

        out_N = N // (bh * bw)
        out_H = H * bh - (top + bottom)
        out_W = W * bw - (left + right)
        out_C = C

        if out_H < 0 or out_W < 0:
            raise ValueError(
                f"BATCH_TO_SPACE_ND: negative output size H={out_H}, W={out_W} "
                f"with crops (t,l,b,r)=({top},{left},{bottom},{right})"
            )

        generated_params["in_dim"] = [N, H, W, C]
        generated_params["out_dim"] = [out_N, out_H, out_W, out_C]
        generated_params["block_shape"] = [bh, bw]

        # For your C API: cmsis_nn_dims used as (n,h,w,c) → (top,left,bottom,right)
        generated_params["crop_nhwc4"] = [top, left, bottom, right]

        generated_params["input_size"] = int(np.prod([N, H, W, C]))
        generated_params["output_size"] = int(np.prod([out_N, out_H, out_W, out_C]))

        # Quantization (pass-through op; still record what converter gives)
        scales, qo = _extract_io_scales(tflite_fname)
        generated_params.update(qo)

        return Lib.op_utils.Generated_data(generated_params, tensors, scales, effective_scales, aliases)

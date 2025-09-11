# SPDX-FileCopyrightText: Copyright 2024 Arm Limited
# SPDX-License-Identifier: Apache-2.0
#
# Hard-Swish test-vector generator (TFLite-backed, TFLM-style params)
#
# This generates parameters for a TFLM-accurate int8 hard-swish kernel that uses:
#   - input_zero_point, output_zero_point
#   - output_multiplier_fixedpoint_q15, output_multiplier_exponent (<= 0)
#   - reluish_multiplier_fixedpoint_q15, reluish_multiplier_exponent
#
# where:
#   hires_input_scale = (1/128) * s_in
#   output_multiplier_real = hires_input_scale / s_out
#   reluish_multiplier_real = hires_input_scale / (3 / 32768)
#
# Both multipliers are quantized like TFLM:
#   QuantizeMultiplier(real) -> (q31, exp)
#   DownScaleInt32ToInt16Multiplier(q31) -> q15 (keeps same exp)
#
import math
import numpy as np
import tensorflow as tf
from tensorflow.lite.python.interpreter import Interpreter, OpResolverType
import keras

import Lib.op_utils


class Op_hard_swish(Lib.op_utils.Op_type):

    # ------------------------ Shapes / Keras model ------------------------ #

    def get_shapes(params):
        shapes = {}
        shapes["input_tensor"] = (
            params["batch_size"],
            params["height_1"],
            params["width_1"],
            params["channel_1"],
        )
        shapes["representational_dataset"] = (
            params["batch_size"],
            params["height_1"],
            params["width_1"],
            params["channel_1"],
        )
        shapes["different_in_shapes"] = False
        return shapes

    def generate_keras_model(shapes, params):
        inp = keras.layers.Input(batch_shape=shapes["input_tensor"])
        out = keras.layers.Activation("hard_swish")(inp)
        return keras.Model([inp], [out])

    # ------------------------ Multiplier helpers (TFLM-like) -------------- #

    @staticmethod
    def _quantize_multiplier_q31(real_scale: float) -> tuple[int, int]:
        """
        TFLM-like QuantizeMultiplier:
        Given real_scale, produce (q31, exp) such that:
          real_scale ≈ q31 / 2^31 * 2^exp    with q31 in [2^30, 2^31)
        """
        if real_scale == 0.0:
            return 0, 0

        significand, exponent = math.frexp(real_scale)  # real_scale = significand * 2^exponent, significand in [0.5,1)
        q31 = int(math.floor(significand * (1 << 31) + 0.5))
        if q31 == (1 << 31):  # handle rounding overflow
            q31 //= 2
            exponent += 1
        return q31, exponent

    @staticmethod
    def _downscale_q31_to_q15(q31: int) -> int:
        """
        TFLM-like DownScaleInt32ToInt16Multiplier:
        Convert a Q31 multiplier to Q15 with rounding, keeping the same exponent.
        """
        # round(q31 / 2^16)
        q15 = int((q31 + (1 << 15)) >> 16)
        if q15 == (1 << 15):  # saturate if it rounded up to 2^15
            q15 = (1 << 15) - 1
        return q15

    @staticmethod
    def _to_q15_exp(real_scale: float) -> tuple[int, int]:
        """
        Convenience: real_scale -> (q15, exp) via Q31 + downscale.
        """
        q31, exp = Op_hard_swish._quantize_multiplier_q31(real_scale)
        q15 = Op_hard_swish._downscale_q31_to_q15(q31)
        return q15, exp

    # ------------------------ Stimulus (int8) ----------------------------- #

    @staticmethod
    def _pattern_int8(shape, kind: str) -> np.ndarray:
        """
        Build edge-case int8 patterns in the *quantized* domain.
        """
        N = int(np.prod(shape))

        def reshape(arr):
            return arr.astype(np.int8).reshape(shape)

        kind = (kind or "random_uniform").lower()

        if kind == "random_uniform":
            return reshape(np.random.randint(-128, 128, size=N))

        if kind == "sweep_edges":
            vals = np.linspace(-128, 127, num=N, dtype=np.int16)
            return reshape(vals)

        if kind == "plateau_edges":
            q_neg = -128
            q_pos = 127
            near_zero = [-2, -1, 0, 1, 2]
            near_l = [-4, -3, -2]   # around -3
            near_r = [2, 3, 4]      # around +3
            quarters = max(1, N // 4)
            a = np.full(quarters, q_neg, dtype=np.int16)
            b = np.full(quarters, q_pos, dtype=np.int16)
            c = np.random.choice(near_l + near_zero + near_r, size=quarters).astype(np.int16)
            d = np.random.randint(-16, 17, size=N - 3 * quarters).astype(np.int16)
            return reshape(np.concatenate([a, b, c, d]))

        if kind == "dense_mixture":
            vals = np.random.randint(-128, 128, size=N).astype(np.int16)
            idx = np.random.choice(N, size=max(4, N // 20), replace=False)
            spikes = np.random.choice(
                [-128, -127, -4, -3, -2, 0, 2, 3, 4, 126, 127],
                size=idx.size
            ).astype(np.int16)
            vals[idx] = spikes
            return reshape(vals)

        if kind == "adversarial_step":
            half = N // 2
            left = np.random.randint(-6, -2, size=half).astype(np.int16)      # around -3
            right = np.random.randint(2, 7, size=N - half).astype(np.int16)   # around +3
            vals = np.concatenate([left, right])
            if N >= 8:
                vals[:3] = [-3, 0, 3]
            return reshape(vals)

        # Fallback
        return reshape(np.random.randint(-128, 128, size=N))

    # ------------------------ Main generator ------------------------------ #

    def generate_data_tflite(tflite_fname, params):
        """
        Build test vectors + TFLM-style kernel parameters.

        Exports (for int8):
          input_offset, output_offset
          output_multiplier_fixedpoint_q15, output_multiplier_exponent
          reluish_multiplier_fixedpoint_q15, reluish_multiplier_exponent
          output_len
          tensors: input_tensor (int8), output (int8, TFLite ref)
        """
        tensors = {}
        effective_scales = {}
        scales = {}
        generated_params = {}
        aliases = {}

        input_shape = (
            params["batch_size"],
            params["height_1"],
            params["width_1"],
            params["channel_1"],
        )
        input_dtype_str = params["input_data_type"]
        input_dtype_tf = Lib.op_utils.get_tf_dtype(input_dtype_str)

        if input_dtype_tf != tf.int8:
            raise ValueError("This generator currently supports int8 only for HardSwish.")

        # Pick stimulus
        pattern = params.get("pattern", "random_uniform")
        input_q = Op_hard_swish._pattern_int8(input_shape, pattern)
        tensors["input_tensor"] = input_q

        # Interpreter (use reference kernels for determinism)
        interpreter = Interpreter(
            model_path=str(tflite_fname),
            experimental_op_resolver_type=OpResolverType.BUILTIN_REF,
        )
        interpreter.allocate_tensors()

        # Quantization params
        in_det = interpreter.get_input_details()[0]
        out_det = interpreter.get_output_details()[0]

        input_scale, input_zero = in_det["quantization"]
        output_scale, output_zero = out_det["quantization"]

        scales["input_scale"] = float(input_scale)
        scales["input_zero_point"] = int(input_zero)
        scales["output_scale"] = float(output_scale)
        scales["output_zero_point"] = int(output_zero)

        generated_params["input_offset"] = int(input_zero)
        generated_params["output_offset"] = int(output_zero)

        if input_scale == 0.0 or output_scale == 0.0:
            raise ValueError("Invalid (zero) quantization scale(s).")

        # ---------------- TFLM-style multipliers ---------------- #
        # hires_input_scale = (1/128) * s_in
        hires_input_scale = (1.0 / 128.0) * float(input_scale)

        # output_multiplier_real = hires_input_scale / s_out
        out_mul_real = hires_input_scale / float(output_scale)
        out_q15, out_exp = Op_hard_swish._to_q15_exp(out_mul_real)

        # reluish_multiplier_real = hires_input_scale / (3 / 32768)
        reluish_scale = 3.0 / 32768.0
        relu_mul_real = hires_input_scale / reluish_scale
        relu_q15, relu_exp = Op_hard_swish._to_q15_exp(relu_mul_real)

        relu_q3 = int(np.rint(3.0 / input_scale))
        relu_q6 = int(np.rint(6.0 / input_scale))

        # For int8 path, TFLM expects output_multiplier_exponent <= 0
        if out_exp > 0:
            # Very unusual scales; fail early to match TFLM preconditions.
            raise ValueError(f"Unexpected positive output exponent ({out_exp}) for int8 hard-swish.")

        # Needed for compat hard-swish variant
        generated_params["output_multiplier_fp"] = int(out_q15)
        generated_params["output_multiplier_exp"] = int(out_exp)
        generated_params["relu_multiplier_fp"] = int(relu_q15)
        generated_params["relu_multiplier_exp"] = int(relu_exp)

        # Needed for precise hard-swish variant
        M_real = (input_scale * input_scale) / (6.0 * output_scale)
        multiplier, shift = Lib.op_utils.compute_multiplier_shift(M_real)
        generated_params["output_multiplier"] = int(multiplier)
        generated_params["output_shift"] = int(shift)
        generated_params["relu_q3"] = int(relu_q3)
        generated_params["relu_q6"] = int(relu_q6)

        # Keep these for debugging/inspection
        effective_scales["hires_input_scale"] = float(hires_input_scale)
        effective_scales["output_multiplier_real"] = float(out_mul_real)
        effective_scales["reluish_multiplier_real"] = float(relu_mul_real)

        # Run ref TFLite once to capture expected output
        interpreter.set_tensor(in_det["index"], input_q)
        interpreter.invoke()
        out_q = interpreter.get_tensor(out_det["index"]).astype(np.int8)
        tensors["output"] = out_q

        # Output length
        out_shape = out_det["shape"]
        generated_params["output_len"] = int(np.prod(out_shape))

        return Lib.op_utils.Generated_data(
            generated_params,
            tensors,
            scales,
            effective_scales,
            aliases,
        )

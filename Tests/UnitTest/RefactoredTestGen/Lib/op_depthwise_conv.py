# SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import Lib.op_utils
import math

from tensorflow.lite.python.interpreter import Interpreter
from tensorflow.lite.python.interpreter import OpResolverType
import keras
import numpy as np

def generate_data(tflite_fname, params):
    tensors = {}
    effective_scales = {}
    scales = {}
    generated_params = {}
    aliases = {}

    # To be removed
    aliases["output_multiplier"] = "output_mult"
    aliases["bias"] = "biases"
    aliases["output"] = "output_ref"

    if params["tflite_generator"] == "json":
        import tflite_micro # Only tflite_micro interpreter supports int4 convolution
        
        interpreter = tflite_micro.runtime.Interpreter.from_file(
            model_path=str(tflite_fname), arena_size=params["arena_size"],
            intrepreter_config=tflite_micro.runtime.InterpreterConfig.kPreserveAllTensors)

        output_details = interpreter.get_output_details()
        if params["input_h"] == 1:
            # 1D convolution: shape is (batch, width, channels)
            x_output = output_details[0]["shape"][1]  # width
            y_output = 1  # height is always 1 for 1D
        elif params["input_w"] == 1:
            # 1D convolution: shape is (batch, height, channels)  
            x_output = 1  # width is always 1 for 1D
            y_output = output_details[0]["shape"][1]  # height
        else:
            # 2D convolution: shape is (batch, height, width, channels)
            x_output = output_details[0]["shape"][2]  # width
            y_output = output_details[0]["shape"][1]  # height

        input_details = interpreter.get_input_details()
        (scales["input_scale"], scales["input_zero_point"]) = input_details[0]['quantization']
        (scales["output_scale"], scales["output_zero_point"]) = output_details[0]['quantization']

        scales["scaling_factors"] = params["w_scale"]
    else:

        interpreter = Interpreter(str(tflite_fname), experimental_op_resolver_type=OpResolverType.BUILTIN_REF)
        interpreter.allocate_tensors()
        # after allocate_tensors()
        tensor_details = interpreter.get_tensor_details()
        in_ch  = params["in_ch"]
        out_ch = params["out_ch"]
        dm     = out_ch // in_ch
        fy     = params.get("filter_y", 1)
        fx     = params.get("filter_x", 1)

        # pick rank-4, per-channel-quantized weights
        candidates = [
            td for td in tensor_details
            if td["dtype"] in (np.int8, np.int16)
            and len(td["shape"]) == 4
            and td["quantization_parameters"]["scales"].size == out_ch
            and td["quantization_parameters"]["quantized_dimension"] in (3, len(td["shape"])-1)
        ]
        if not candidates:
            raise RuntimeError("Could not locate depthwise weight tensor (rank-4, per-channel).")

        # choose the largest candidate (safeguard if multiple)
        weights_td = max(candidates, key=lambda td: int(np.prod(td["shape"])))

        # scales = per-channel weight scales
        scales["scaling_factors"] = weights_td["quantization_parameters"]["scales"]
        tensors["weights"] = interpreter.get_tensor(weights_td["index"])

        # (optional) bias
        bias_td = next((td for td in tensor_details
                        if td["dtype"] == np.int32 and len(td["shape"]) == 1 and int(td["shape"][0]) == out_ch), None)
        tensors["bias"] = interpreter.get_tensor(bias_td["index"]) if bias_td and params["generate_bias"] else None

        # sanity check: weights size matches DW formula
        expected = (fy * fx * in_ch * dm)
        actual   = tensors["weights"].size
        if actual != expected:
            raise RuntimeError(f"Depthwise weights size mismatch: got {actual}, expected {expected} "
                            f"(fy={fy}, fx={fx}, in_ch={in_ch}, dm={dm}).")

        output_details = interpreter.get_output_details()
        (scales["output_scale"], scales["output_zero_point"]) = output_details[0]['quantization']

        input_details = interpreter.get_input_details()
        (scales["input_scale"], scales["input_zero_point"]) = input_details[0]['quantization']

        if params["input_h"] == 1:
            # 1D convolution: shape is (batch, width, channels)
            x_output = output_details[0]["shape"][1]  # width
            y_output = 1  # height is always 1 for 1D
        elif params["input_w"] == 1:
            # 1D convolution: shape is (batch, height, channels)  
            x_output = 1  # width is always 1 for 1D
            y_output = output_details[0]["shape"][1]  # height
        else:
            # 2D convolution: shape is (batch, height, width, channels)
            x_output = output_details[0]["shape"][2]  # width
            y_output = output_details[0]["shape"][1]  # height
            
    def calculate_padding(x_output, y_output, params):
        x_input = params["input_w"]
        y_input = params["input_h"]

        if params["padding"] == "SAME":
            # Take dilation into account.
            filter_x = (params["filter_x"] - 1) * params["dilation_x"] + 1
            filter_y = (params["filter_y"] - 1) * params["dilation_y"] + 1

            pad_along_width = max((x_output - 1) * params["stride_x"] + filter_x - x_input, 0)
            pad_along_height = max((y_output - 1) * params["stride_y"] + filter_y - y_input, 0)

            pad_top = pad_along_height // 2

            pad_left = pad_along_width // 2
            pad_top_offset = pad_along_height % 2
            pad_left_offset = pad_along_width % 2

            pad_y_with_offset = pad_top + pad_top_offset
            pad_x_with_offset = pad_left + pad_left_offset
            pad_x = pad_left
            pad_y = pad_top
        else:
            pad_x = 0
            pad_y = 0
            pad_y_with_offset = 0
            pad_x_with_offset = 0

        return pad_y_with_offset, pad_x_with_offset, pad_y, pad_x

    pad_y_with_offset, pad_x_with_offset, pad_y, pad_x = calculate_padding(x_output, y_output, params)

    generated_params["input_batches"] = params["batch_size"]
    generated_params["pad_x"] = pad_x
    generated_params["pad_y"] = pad_y
    generated_params["output_h"] = y_output
    generated_params["output_w"] = x_output
    generated_params["dst_size"] = x_output * y_output * params["out_ch"] * params["batch_size"]
    generated_params["input_offset"] = -scales["input_zero_point"]
    generated_params["output_offset"] = scales["output_zero_point"]

    per_channel_multiplier, per_channel_shift = Lib.op_utils.generate_quantize_per_channel_multiplier(params, scales)

    tensors["output_multiplier"] = np.array(per_channel_multiplier)
    tensors["output_shift"] = np.array(per_channel_shift)

    return Lib.op_utils.Generated_data(generated_params, tensors, scales, effective_scales, aliases)

class Op_depthwise_conv(Lib.op_utils.Op_type):

    def get_shapes(params):
        shapes = {}

        # Common default parameters
        params["stride_x"] = 1 if "stride_x" not in params else params["stride_x"]
        params["stride_y"] = 1 if "stride_y" not in params else params["stride_y"]
        params["dilation_x"] = 1 if "dilation_x" not in params else params["dilation_x"]
        params["dilation_y"] = 1 if "dilation_y" not in params else params["dilation_y"]
        params["batch_size"] = 1 if "batch_size" not in params else params["batch_size"]
        params["generate_bias"] = True if "generate_bias" not in params else params["generate_bias"]
        if "out_activation_min" not in params:
            params["out_activation_min"] = Lib.op_utils.get_dtype_min(params["input_data_type"])
        if "out_activation_max" not in params:
            params["out_activation_max"] = Lib.op_utils.get_dtype_max(params["input_data_type"])
        if "bias_min" not in params:
            params["bias_min"] = Lib.op_utils.get_dtype_min("int32_t")
        if "bias_max" not in params:
            params["bias_max"] = Lib.op_utils.get_dtype_max("int32_t")
        if "weights_min" not in params:
            params["weights_min"] = Lib.op_utils.get_dtype_min("int32_t")
        if "weights_max" not in params:
            params["weights_max"] = Lib.op_utils.get_dtype_max("int32_t")

        in_ch = params["in_ch"]
        out_ch = params["out_ch"]
        
        # For depthwise convolution, calculate channel multiplier
        channel_multiplier = out_ch // in_ch
        if out_ch % in_ch != 0:
            raise RuntimeError("ERROR: Output channels {} must be a multiple of input channels {} for depthwise convolution".format(out_ch, in_ch))

        # Handle both 1D and 2D depthwise convolution shapes
        if params["input_h"] == 1:
            # 1D depthwise convolution
            shapes["input_tensor"] = (params["batch_size"], params["input_w"], in_ch)
            shapes["weight_shape"] = [params["filter_x"], in_ch, channel_multiplier]
        elif params["input_w"] == 1:
            # 1D depthwise convolution
            shapes["input_tensor"] = (params["batch_size"], params["input_h"], in_ch)
            shapes["weight_shape"] = [params["filter_y"], in_ch, channel_multiplier]
        else:
            # 2D depthwise convolution
            shapes["input_tensor"] = (params["batch_size"], params["input_h"], params["input_w"], in_ch)
            shapes["weight_shape"] = [params["filter_y"], params["filter_x"], in_ch, channel_multiplier]

        if params["tflite_generator"] == "json":
            params["json_template"] = "depthwise_conv.json" if params["generate_bias"] else "depthwise_conv_null_bias.json"
        else:
            if params["input_h"] == 1:
                shapes["representational_dataset"] = (params["batch_size"], params["input_w"], in_ch)
            elif params["input_w"] == 1:
                shapes["representational_dataset"] = (params["batch_size"], params["input_h"], in_ch)
            else:
                shapes["representational_dataset"] = (params["batch_size"], params["input_h"], params["input_w"], in_ch)

        if params["generate_bias"]:
            shapes["bias_shape"] = [out_ch]
        else:
            shapes["bias_shape"] = []

        return shapes

    def generate_keras_model(shapes, params):

        model = keras.models.Sequential()
        
        # Calculate channel multiplier for depthwise convolution
        channel_multiplier = params["out_ch"] // params["in_ch"]
        
        # Choose between 1D and 2D depthwise convolution and set appropriate input shapes
        if params["input_h"] == 1:
            # 1D depthwise convolution - input shape is (width, channels)
            input_shape = (params["input_w"], params["in_ch"])
            model.add(keras.layers.InputLayer(shape=input_shape, batch_size=params["batch_size"]))
            depthwise_conv_layer = keras.layers.DepthwiseConv1D(
                kernel_size=params["filter_x"],
                strides=params["stride_x"],
                padding=params["padding"],
                depth_multiplier=channel_multiplier,
                dilation_rate=params["dilation_x"],
                use_bias=params["generate_bias"])
            model.add(depthwise_conv_layer)
        elif params["input_w"] == 1:
            # 1D depthwise convolution - input shape is (height, channels)
            input_shape = (params["input_h"], params["in_ch"])
            model.add(keras.layers.InputLayer(shape=input_shape, batch_size=params["batch_size"]))
            depthwise_conv_layer = keras.layers.DepthwiseConv1D(
                kernel_size=params["filter_y"],
                strides=params["stride_y"],
                padding=params["padding"],
                depth_multiplier=channel_multiplier,
                dilation_rate=params["dilation_y"],
                use_bias=params["generate_bias"])
            model.add(depthwise_conv_layer)
        else:
            # 2D depthwise convolution - input shape is (height, width, channels)
            input_shape = (params["input_h"], params["input_w"], params["in_ch"])
            model.add(keras.layers.InputLayer(shape=input_shape, batch_size=params["batch_size"]))
            depthwise_conv_layer = keras.layers.DepthwiseConv2D(
                kernel_size=(params["filter_y"], params["filter_x"]),
                strides=(params["stride_y"], params["stride_x"]),
                padding=params["padding"],
                depth_multiplier=channel_multiplier,
                dilation_rate=(params["dilation_y"], params["dilation_x"]),
                use_bias=params["generate_bias"])
            model.add(depthwise_conv_layer)

        weights = Lib.op_utils.generate_tf_tensor(
            shapes["weight_shape"], params["weights_min"], params["weights_max"], decimals=8)

        if params["generate_bias"]:
            bias = Lib.op_utils.generate_tf_tensor(
                shapes["bias_shape"], params["bias_min"], params["bias_max"])
            depthwise_conv_layer.set_weights([weights, bias])
        else:
            depthwise_conv_layer.set_weights([weights])

        return model

    def generate_data_tflite(tflite_fname, params):
        return generate_data(tflite_fname, params)

    def post_model_update(tflite_path, generated_data, params):

        data = generate_data(tflite_path, params)

        generated_data.params |= data.params
        generated_data.aliases |= data.aliases
        generated_data.tensors |= data.tensors

        return generated_data

    def generate_data_json(shapes, params):


        tensors = {}
        effective_scales = {}
        scales = {}
        generated_params = {}
        aliases = {}

        generated_params["input_batches"] = params["batch_size"]
        generated_params["dst_size"] = params["out_ch"] * params["batch_size"]

        def quantize_float_data(data=None, quantization_bit_range=8, quantization_type="affine", tf_tensor=False):
            if data is None:
                return

            if tf_tensor:
                data = data.numpy()
            data_max = np.amax(data)
            data_min = np.amin(data)

            if quantization_type.lower() == "affine":
                data_min = min(data_min, 0.0)
                data_max = max(data_max, 0.0)

                scale = (data_max - data_min) / (pow(2, quantization_bit_range) - 1)
                zero_point = -(round(data_max * scale)) - pow(2, quantization_bit_range - 1)
                zero_point = max(zero_point, pow(quantization_bit_range - 1) - 1)
                zero_point = min(zero_point, -pow(quantization_bit_range - 1))

            elif quantization_type.lower() == "symmetric":
                absolute_max = max(abs(data_min), abs(data_max))
                scale = absolute_max / (pow(2, quantization_bit_range - 1) - 1)
                zero_point = 0

            else:
                raise RuntimeError("Quantization scheme not supported")

            scale = 0.1 if scale == 0 else scale
            quantized_data = [(x // scale) + zero_point for x in data]
            return np.array(quantized_data), scale, zero_point

        if params["generate_bias"]:
            quant_bias, bias_scale, bias_zp = quantize_float_data(
                np.random.randint(
                    params["bias_min"],
                    params["bias_max"],
                    size=shapes["bias_shape"]),
                quantization_bit_range=8,
                quantization_type="symmetric",
                tf_tensor=(not params["generate_bias"]))

            params["bias_scale"] = [bias_scale] * params["out_ch"]
            params["bias_zp"] = [bias_zp] * params["out_ch"]

            tensors["input_bias"] = quant_bias
        else:
            tensors["input_bias"] = None

        params["w_zp"] = [0] * params["out_ch"]
        params["w_scale"] = np.random.uniform(0.001, 0.01, size=[params["out_ch"]]).tolist()

        params["output_scale"] = np.random.uniform(0.02, 0.06)

        if params["padding"] == "SAME":
            # TODO dilation with padding
            output_x = math.ceil(float(params["input_w"]) / float(params["stride_x"]))
            output_y = math.ceil(float(params["input_h"]) / float(params["stride_y"]))
        else:
            dilation_filter_x = (params["filter_x"] - 1) * (params["dilation_x"] - 1)
            dilation_filter_y = (params["filter_y"] - 1) * (params["dilation_y"] - 1)

            output_x = math.ceil(
                float(params["input_w"] - params["filter_x"] - dilation_filter_x + 1) / float(params["stride_x"]))
            output_y = math.ceil(
                float(params["input_h"] - params["filter_y"] - dilation_filter_y + 1) / float(params["stride_y"]))
        generated_params["output_h"] = output_y
        generated_params["output_w"] = output_x

        generated_params["input_offset"] = -params["input_zp"]
        generated_params["output_offset"] = params["output_zp"]

        aliases["input_bias"] = "biases"
        aliases["input_weights"] = "weights"

        weights = np.random.randint(
            params["weights_min"], params["weights_max"], size=shapes["weight_shape"])


        if params['weights_data_type'] == 'int4_t':
            uneven = weights.size % 2
            if uneven:
                weights = np.append(weights, 0)
            temp = np.reshape(weights, (weights.size // 2, 2)).astype(np.uint8)
            weights = 0xff & ((0xf0 & (temp[:, 1] << 4)) | (temp[:, 0] & 0xf))
        tensors["input_weights"] = weights

        return Lib.op_utils.Generated_data(generated_params, tensors, scales, effective_scales, aliases)
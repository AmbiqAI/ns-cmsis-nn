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
import os
import Lib.op_lstm
import Lib.op_conv
import Lib.op_batch_matmul
import Lib.op_fully_connected
import Lib.op_pooling
import Lib.op_pad
import Lib.op_maximum_minimum
import Lib.op_transpose
import Lib.op_elementwise_mul
import Lib.op_quantize
import Lib.op_dequantize
import tensorflow as tf
import numpy as np
from tensorflow.lite.python.interpreter import Interpreter
from tensorflow.lite.python.interpreter import OpResolverType
import pathlib
import subprocess
import sys
import math
import keras

dtype_map = {
    "float32":   "float",
    "int8_t":    "int8_t",
    "int8":      "int8_t", 
    "int16_t":   "int16_t",
    "int16":     "int16_t"
}

# Optional runtime interpreters
try:
    import tflite_micro
    tflite_micro_imported = True
except ModuleNotFoundError:
    print("WARNING: tflite_micro not installed, skipping tests using this interpreter.")
    tflite_micro_imported = False

try:
    from tflite_runtime.interpreter import Interpreter as TfliteRuntimeInterpreter
    from tflite_runtime.interpreter import OpResolverType as TfliteRuntimeOpResolverType

    tflite_runtime_imported = True
except ModuleNotFoundError:
    print("WARNING: tflite_runtime not installed, skipping tests using this interpreter.")
    tflite_runtime_imported = False


def generate(params, args, fpaths):
    """ Create a test with given parameters """

    # Check if test is valid, skip otherwise
    if (params["interpreter"] == "tflite_runtime") and (not tflite_runtime_imported):
        print("Skipping due to tflite_runtime not being installed...")
        return
    if (params["interpreter"] == "tflite_micro") and (not tflite_micro_imported):
        print("Skipping due to tflite_micro not being installed...")
        return

    op_type = get_op_type(params["op_type"])
    shapes = op_type.get_shapes(params)

    # Create test related fpaths
    fpaths["data_folder"] = pathlib.Path("TestCases") / "TestData" / params["name"]
    fpaths["tflite"] = fpaths["data_folder"] / f"{params['name']}.tflite"
    fpaths["config_data"] = fpaths["data_folder"] / "config_data.h"
    fpaths["test_data"] = fpaths["data_folder"] / "test_data.h"

    # Generate reference data
    if params["tflite_generator"] == "keras":
        keras_model = op_type.generate_keras_model(shapes, params)

        per_tensor_quant_for_dense = False
        try:
            per_tensor_quant_for_dense = not params["per_channel_quant"]
        except KeyError:
            pass

        if "bias_data_type" in params:
            bias_dtype = params["bias_data_type"]
        else:
            bias_dtype = None
        if op_type == Lib.op_dequantize:
            convert_keras_to_tflite(fpaths["tflite"],
                                    keras_model,
                                    quantize=False,
                                    dtype=params["input_data_type"],
                                    bias_dtype=bias_dtype,
                                    shape=shapes,
                                    per_tensor_quant_for_dense=per_tensor_quant_for_dense)
        else:
            convert_keras_to_tflite(fpaths["tflite"],
                                    keras_model,
                                    quantize=True,
                                    dtype=params["input_data_type"],
                                    bias_dtype=bias_dtype,
                                    shape=shapes,
                                    per_tensor_quant_for_dense=per_tensor_quant_for_dense)

        data = op_type.generate_data_tflite(fpaths["tflite"], params)

    elif params["tflite_generator"] == "json":
        data = op_type.generate_data_json(shapes, params)
        json_template_fpath = fpaths["json_template_folder"] / f"{params['json_template']}"
        json_output_fpath = fpaths["data_folder"] / f"{params['name']}.json"
        replacements = {**params, **data.params, **data.scales}
        convert_json_to_tflite(json_template_fpath, json_output_fpath, data.tensors, replacements, args.schema)

    else:
        raise ValueError(f"Invalid tflite generator in {params['name']}")

    data = op_type.post_model_update(fpaths["tflite"], data, params)

    params.update(data.params)

    # Quantize scales
    for name, scale in data.effective_scales.items():
        mult, shift = quantize_scale(scale)
        params[name + "_multiplier"] = mult
        params[name + "_shift"] = shift

    # Get only the input data type (needed if it is a quantized or dequantized type e.g. "float32_to_int8" -> "float32")
    dtype = params["input_data_type"]
    # If the string has a '_to_' segment, strip off everything after '_to_'
    if "_to_" in dtype:
        raw_dtype = dtype.split("_to_", 1)[0]  # take only what comes before "_to_"
        dtype = Lib.op_utils.get_tf_dtype(raw_dtype)
        minval = Lib.op_utils.get_dtype_min(dtype) if "input_min" not in params else params["input_min"]
        maxval = Lib.op_utils.get_dtype_max(dtype) if "input_max" not in params else params["input_max"]
    else:
        minval = Lib.op_utils.get_dtype_min(dtype) if "input_min" not in params else params["input_min"]
        maxval = Lib.op_utils.get_dtype_max(dtype) if "input_max" not in params else params["input_max"]
        dtype = Lib.op_utils.get_tf_dtype(dtype)
    # Run reference model

    # Initialize input tensors
    input_tensors = {}
    for shape_name, shape in shapes.items():
        if "input_tensor" in shape_name:
            if shape_name in data.tensors:
                input_tensors[shape_name] = data.tensors[shape_name]
            else:
                input_tensors[shape_name] = Lib.op_utils.generate_tf_tensor(shape, minval, maxval, decimals=0, datatype=dtype)
                data.tensors[shape_name] = input_tensors[shape_name].numpy()

    if not input_tensors:
        raise ValueError("Op_type must initialize at least one input shape")

    if params["interpreter"] == "tensorflow":
        data.tensors["output"] = invoke_tflite(fpaths["tflite"], input_tensors)
    elif params["interpreter"] == "tflite_runtime":
        data.tensors["output"] = invoke_tflite_runtime(fpaths["tflite"], input_tensors)
    elif params["interpreter"] == "tflite_micro":
        if "arena_size" in params:
            data.tensors["output"] = invoke_tflite_micro(fpaths["tflite"], input_tensors, params["arena_size"])
        else:
            data.tensors["output"] = invoke_tflite_micro(fpaths["tflite"], input_tensors)
    else:
        raise ValueError(f"Invalid interpreter in {params['name']}")

    if "activation_min" in params:
        data.tensors["output"] = np.maximum(data.tensors["output"], params["activation_min"])
    if "activation_max" in params:
        data.tensors["output"] = np.minimum(data.tensors["output"], params["activation_max"])

    # Write data
    header = get_header(params["tflite_generator"], params["interpreter"])

    def include_in_config(key):
        return key not in [
            "suite_name", "name", "input_data_type", "op_type", "input_data_type", "weights_data_type",
            "bias_data_type", "shift_and_mult_data_type", "interpreter", "tflite_generator", "json_template",
            "groups", "generate_bias", "bias_min", "bias_max", "weights_min", "weights_max", "bias_zp", "w_zp",
            "input_zp", "output_zp", "w_scale", "bias_scale", "input_scale", "output_scale", "arena_size"
        ]

    config_params = {key: val for key, val in params.items() if include_in_config(key)}
    write_config(fpaths["config_data"], config_params, params["name"], fpaths["test_data"], header)

    for name, tensor in data.tensors.items():
        fpaths[name] = fpaths["data_folder"] / f"{name}.h"

        # 1) Detect OP type to decide how to pick data type
        if params.get("op_type") in ("quantize", "dequantize"):
            # For quantize/dequantize ops, use your new "X_to_Y" logic
            if name.startswith("input_tensor"):
                # Force input dtype to left side of "float32_to_int16_t" (e.g. "float32")
                raw_dtype = params["input_data_type"]  # e.g. "float32_to_int16_t"
                if "_to_" in raw_dtype:
                    raw_dtype = raw_dtype.split("_to_", 1)[0]  # "float32"
                c_type = map_dtype_to_c_type(raw_dtype)

            elif name == "output":
                # Force output dtype to right side of "float32_to_int16_t" (e.g. "int16_t")
                raw_dtype = params["input_data_type"]
                if "_to_" in raw_dtype:
                    raw_dtype = raw_dtype.split("_to_", 1)[1]  # "int16_t"
                c_type = map_dtype_to_c_type(raw_dtype)

            else:
                # Fallback if there are other input/output names
                c_type = "float"

        else:
            # 2) For all other ops, use the original "get_dtype" logic
            c_type = Lib.op_utils.get_dtype(name, params)

        # 3) Preserve the clipping for output if specified in params
        if (
            name == "output"
            and "out_activation_min" in params
            and "out_activation_max" in params
        ):
            tensor = np.clip(tensor, params["out_activation_min"], params["out_activation_max"])

        # 4) Write the array to a header file
        write_c_array(
            tensor,
            fpaths[name],
            c_type,
            params["name"],
            name,
            fpaths["test_data"],
            header
        )

        if name in data.aliases:
            append_alias_to_c_array_file(fpaths[name], dtype, params["name"], name, data.aliases[name])


def get_op_type(op_type_string):
    if op_type_string == "lstm":
        return Lib.op_lstm.Op_lstm
    elif op_type_string == "conv":
        return Lib.op_conv.Op_conv
    elif op_type_string == "batch_matmul":
        return Lib.op_batch_matmul.Op_batch_matmul
    elif op_type_string == "elementwise_mul":
        return Lib.op_elementwise_mul.Op_elementwise_mul
    elif op_type_string == "fully_connected":
        return Lib.op_fully_connected.Op_fully_connected
    elif op_type_string == "avgpool" or op_type_string == "maxpool":
        return Lib.op_pooling.Op_pooling
    if op_type_string == "pad":
        return Lib.op_pad.Op_pad
    elif op_type_string == "maximum_minimum":
        return Lib.op_maximum_minimum.Op_maximum_minimum
    elif op_type_string == "transpose":
        return Lib.op_transpose.Op_transpose
    elif op_type_string == "quantize":
        return Lib.op_quantize.Op_quantize
    elif op_type_string == "dequantize":
        return Lib.op_dequantize.Op_dequantize
    else:
        raise ValueError(f"Unknown op type '{op_type_string}'")


def convert_keras_to_tflite(
        output_fpath, keras_model, quantize, dtype, bias_dtype, shape, per_tensor_quant_for_dense=False):
    """ Convert a model generated with keras to tflite-format """
    keras_model.compile(loss=keras.losses.categorical_crossentropy,
                        metrics=['accuracy'])
    n_inputs = len(keras_model.inputs)
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    if quantize:

        if shape.get("different_in_shapes") is True:
            def representative_dataset():
                for _ in range(100):
                    data1 = np.random.rand(*shape["representational_dataset"])
                    data2 = np.random.rand(*shape["representational_dataset2"])
                    yield [data1.astype(np.float32), data2.astype(np.float32)]
        else:
            def representative_dataset():
                for _ in range(n_inputs):
                    data = np.random.rand(*shape["representational_dataset"])
                    yield [data.astype(np.float32)]

        converter.representative_dataset = representative_dataset
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter._experimental_disable_per_channel_quantization_for_dense_layers = per_tensor_quant_for_dense
        if dtype == "int8_t":
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = Lib.op_utils.get_tf_dtype(dtype)
            converter.inference_output_type = Lib.op_utils.get_tf_dtype(dtype)
        elif dtype == "float32_to_int8_t":
            converter.inference_input_type = tf.float32
            converter.inference_output_type = tf.int8
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8
            ]

        elif dtype == "float32_to_int16_t":
            converter.inference_input_type = tf.float32
            converter.inference_output_type = tf.int16
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8
            ]
            if bias_dtype == "int32_t":
                converter._experimental_full_integer_quantization_bias_type = tf.int32

        elif dtype == "int8_t_to_int8_t":
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8
            ]
            if bias_dtype == "int32_t":
                converter._experimental_full_integer_quantization_bias_type = tf.int32

        elif dtype == "int16_t_to_int16_t":
            converter.inference_input_type = tf.int16
            converter.inference_output_type = tf.int16
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8
            ]
            if bias_dtype == "int32_t":
                converter._experimental_full_integer_quantization_bias_type = tf.int32
        elif dtype == "int8_t_to_float32":
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.float32
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8
            ]
        elif dtype == "int16_t_to_float32":
            converter.inference_input_type = tf.int16
            converter.inference_output_type = tf.float32
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8
            ]
        else:
            converter.inference_input_type = Lib.op_utils.get_tf_dtype(dtype)
            converter.inference_output_type = Lib.op_utils.get_tf_dtype(dtype)
            if bias_dtype == "int32_t":
                converter._experimental_full_integer_quantization_bias_type = tf.int32
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8
            ]

        tflite_model = converter.convert()

    output_fpath.parent.mkdir(parents=True, exist_ok=True)
    with output_fpath.open("wb") as f:
        f.write(tflite_model)


def invoke_tflite(tflite_path, input_tensor):
    interpreter = Interpreter(str(tflite_path), experimental_op_resolver_type=OpResolverType.BUILTIN_REF)
    interpreter.allocate_tensors()

    for i, val in enumerate(input_tensor.values()):
        input_index = interpreter.get_input_details()[i]["index"]
        interpreter.set_tensor(input_index, val)

    interpreter.invoke()
    output_index = interpreter.get_output_details()[0]["index"]
    data = interpreter.get_tensor(output_index)

    return data.flatten()


def invoke_tflite_runtime(tflite_path, input_tensor):
    interpreter = TfliteRuntimeInterpreter(str(tflite_path),
                                           experimental_op_resolver_type=TfliteRuntimeOpResolverType.BUILTIN_REF)
    interpreter.allocate_tensors()

    for i, val in enumerate(input_tensor.values()):
        input_index = interpreter.get_input_details()[i]["index"]
        interpreter.set_tensor(input_index, val)

    interpreter.invoke()
    output_index = interpreter.get_output_details()[0]["index"]
    data = interpreter.get_tensor(output_index)

    return data.flatten()


def invoke_tflite_micro(tflite_path, input_tensor, arena_size=30000):
    interpreter = tflite_micro.runtime.Interpreter.from_file(model_path=str(tflite_path), arena_size=arena_size)

    for i, val in enumerate(input_tensor.values()):
        interpreter.set_input(val, i)

    interpreter.invoke()
    data = interpreter.get_output(0)

    return data.flatten()


def write_config(config_fpath, params, prefix, test_data_fpath, header):
    config_fpath.parent.mkdir(parents=True, exist_ok=True)
    with config_fpath.open("w+") as f:
        f.write(header)
        f.write("#pragma once\n\n")

        for key, val in params.items():
            if isinstance(val, list):
                f.write("#define " + f"{prefix}_{key} ".upper() + "{")
                for v in val:
                    f.write(f"{v}, ")
                f.write("}\n")
                continue
            if isinstance(val, bool):
                if val:
                    val = "true"
                else:
                    val = "false"

            f.write("#define " + f"{prefix}_{key} ".upper() + f"{val}\n")
    format_output_file(config_fpath)

    with test_data_fpath.open("w") as f:
        f.write(header)
        f.write(f'#include "{config_fpath.name}"\n')


def write_c_array(data, fname, dtype, prefix, tensor_name, test_data_fpath, header):

    # Check that the data looks reasonable
    values, counts = np.unique(data, return_counts=True)
    tf.experimental.numpy.experimental_enable_numpy_behavior()

    size = 0 if data is None else data.size

    if len(values) < size / 2 or max(counts) > size / 2:
        print(f"WARNING: {fname} has repeating values, is this intended?")
    if size and len(data) > 500:
        print(f"WARNING: {fname} has more than 500 values, is this intended?")

    with fname.open("w+") as f:
        f.write(header)
        f.write("#pragma once\n")
        f.write("#include <stdint.h>\n\n")
        if size > 0:
            data_shape = data.shape
            format_width = len(str(data.max())) + 1
            data = data.flatten()

            f.write(f"const {dtype} {prefix}_{tensor_name}[{len(data)}] = \n" + "{")

            for i in range(len(data) - 1):
                if i % data_shape[-1] == 0:
                    f.write("\n")
                f.write(f"{data[i]: {format_width}n}, ")

            if len(data) - 1 % data_shape[-1] == 0:
                f.write("\n")
            f.write(f"{data[len(data) - 1]: {format_width}n}" + "\n};\n")

        else:
            f.write(f"const {dtype} *const {prefix}_{tensor_name} = NULL;\n")

    with test_data_fpath.open("a") as f:
        f.write(f'#include "{fname.name}"\n')

    format_output_file(fname)
    format_output_file(test_data_fpath)


def append_alias_to_c_array_file(fname, dtype, prefix, tensor_name, alias_name):
    with fname.open("a") as f:
        f.write(f"\nconst {dtype} *const {prefix}_{alias_name} = {prefix}_{tensor_name};\n")


def format_output_file(file):
    CLANG_FORMAT = 'clang-format -i'  # For formatting generated headers.
    command_list = CLANG_FORMAT.split(' ')
    command_list.append(file)
    try:
        process = subprocess.run(command_list)
        if process.returncode != 0:
            print(f"ERROR: {command_list = }")
            sys.exit(1)
    except Exception as e:
        raise RuntimeError(f"{e} from: {command_list = }")


def generate_test_from_template(name, test_functions_fpath, template_fpath, unity_fpath):
    template = template_fpath.read_text()
    template = template.replace("template", name)

    with test_functions_fpath.open("a") as f:
        f.write(f'#include "../TestData/{name}/test_data.h"\n\n')
        f.write(template)

    with unity_fpath.open("a") as f:
        f.write("void test_" + name + "(void) { " + name + "(); }\n")


def convert_json_to_tflite(json_template_fpath, json_output_fpath, tensors, params, schema_path):
    """ Convert a model in json-format to tflite-format"""

    # Generate json with values from template
    # This way minimizes string searching/ copying
    json_output_fpath.parent.mkdir(parents=True, exist_ok=True)
    with json_template_fpath.open("r") as template:
        with json_output_fpath.open("w+") as output:
            for line in template:
                line_list = line.replace(",", "").split()
                replaced = False
                for key, val in params.items():
                    if key in line_list:
                        if isinstance(val, bool):
                            if val:
                                val = "true"
                            else:
                                val = "false"
                        # To be able to handle cases like "variable_name" : variable_name
                        # make sure to only replace the last occurence per line
                        new_line = str(val).join(line.rsplit(key, 1))
                        output.write(new_line)
                        replaced = True
                        break

                for key in tensors:
                    if key in line:
                        dtype = Lib.op_utils.get_dtype(key, params)
                        dtype_len = Lib.op_utils.get_dtype_len(dtype)
                        np_dtype = Lib.op_utils.get_np_dtype(dtype)

                        # Tensors are stored byte-wise in schema
                        weights_in_bytes = []
                        for weight in tensors[key].flatten():
                            weights_in_bytes.extend([b for b in int(np_dtype(weight)).to_bytes(dtype_len, 'little')])

                        for byte in weights_in_bytes[:-1]:
                            output.write(f"        {byte},\n")
                        output.write(f"        {weights_in_bytes[-1]}\n")

                        replaced = True
                        break

                if not replaced:
                    output.write(line)

    # Generate tflite from json
    command = f"flatc  -o {json_output_fpath.parent} -c -b {schema_path} {json_output_fpath}"
    command_list = command.split()
    try:
        process = subprocess.run(command_list, env={'PATH': os.getenv('PATH')})
        if process.returncode != 0:
            print(f"ERROR: {command = }")
            sys.exit(1)
    except Exception as e:
        raise RuntimeError(f"{e} from: {command = }. Did you install flatc?")


def quantize_scale(scale):
    significand, shift = math.frexp(scale)
    significand_q31 = round(significand * (1 << 31))
    return significand_q31, shift


def get_header(generator, interpreter):
    header = f"// Generated by {os.path.basename(sys.argv[0])}"
    if generator == "keras":
        header += f" using tensorflow version {tf.__version__} (Keras version {keras.__version__}).\n"
    elif generator == "json":
        command = "flatc  --version"
        command_list = command.split()
        try:
            process = subprocess.Popen(command_list,
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE,
                                       env={'PATH': os.getenv('PATH')})
            flatc_version, err = process.communicate()
            if process.returncode != 0:
                print(f"ERROR: {command = }")
                sys.exit(1)
        except Exception as e:
            raise RuntimeError(f"{e} from: {command = }. Did you install flatc?")
        header += f" using {str(flatc_version)[2:-3]}\n"
    else:
        raise Exception

    if interpreter == "tensorflow":
        version = tf.__version__
        revision = tf.__git_version__
        header += f"// Interpreter from tensorflow version {version} and revision {revision}.\n"
    elif interpreter == "tflite_runtime":
        import tflite_runtime as tfl_runtime

        version = tfl_runtime.__version__
        revision = tfl_runtime.__git_version__
        header += f"// Interpreter from tflite_runtime version {version} and revision {revision}.\n"
    elif interpreter == "tflite_micro":
        version = tflite_micro.__version__
        header += f"// Interpreter from tflite_micro runtime version {version}.\n"
    else:
        raise Exception

    return header


def map_dtype_to_c_type(raw_dtype):
    # Return the correct C type for a given "raw_dtype" string
    if raw_dtype in dtype_map:
        return dtype_map[raw_dtype]
    else:
        raise ValueError(f"Unhandled raw_dtype '{raw_dtype}' in map_dtype_to_c_type()")

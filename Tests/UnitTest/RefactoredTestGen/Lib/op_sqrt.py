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
import tensorflow as tf
import numpy as np
from packaging.version import Version, InvalidVersion

from tensorflow.lite.python.interpreter import Interpreter
from tensorflow.lite.python.interpreter import OpResolverType
from Lib.keras_compat import keras


class Op_sqrt(Lib.op_utils.Op_type):

    @staticmethod
    def _validate_tf_version():
        try:
            tf_version = Version(tf.__version__)
        except InvalidVersion as exc:
            raise RuntimeError(
                f"Unable to parse TensorFlow version '{tf.__version__}'. "
                "SQRT test generation requires tensorflow>=2.21,<2.22."
            ) from exc

        if tf_version < Version("2.21") or tf_version >= Version("2.22"):
            raise RuntimeError(
                "SQRT test generation requires tensorflow>=2.21,<2.22. "
                "Install with: pip install -r Tests/UnitTest/requirements-sqrt-gen.txt"
            )

    def get_shapes(params):
        shapes = {}

        shapes["input_tensor"] = (
            params["batch_size"],
            params["height_1"],
            params["width_1"],
            params["channel_1"]
        )

        shapes["representational_dataset"] = (
            params["batch_size"],
            params["height_1"],
            params["width_1"],
            params["channel_1"]
        )
        # Keep calibration samples in-domain for sqrt to avoid NaN-driven fallback.
        shapes["representational_dataset_min"] = max(0, params.get("input_min", 0))
        shapes["representational_dataset_max"] = max(
            shapes["representational_dataset_min"] + 1,
            params.get("input_max", 127)
        )

        shapes["different_in_shapes"] = False

        return shapes

    def generate_keras_model(shapes, params):
        Op_sqrt._validate_tf_version()
        # Keep Keras model in float domain and rely on TFLite PTQ for int8 IO quantization.
        input_1 = keras.layers.Input(batch_input_shape=shapes["input_tensor"], dtype=tf.float32)
        if hasattr(keras, "ops") and hasattr(keras.ops, "sqrt"):
            layer = keras.ops.sqrt(input_1)
        else:
            layer = keras.layers.Lambda(lambda x: tf.math.sqrt(x))(input_1)
        model = keras.Model([input_1], [layer])
        return model

    def generate_data_tflite(tflite_fname, params):
        tensors = {}
        effective_scales = {}
        scales = {}
        generated_params = {}
        aliases = {}

        input_shape = (
            params["batch_size"],
            params["height_1"],
            params["width_1"],
            params["channel_1"]
        )

        input_dtype_str = params["input_data_type"]
        input_dtype_tf = Lib.op_utils.get_tf_dtype(input_dtype_str)
        if input_dtype_tf != tf.int8:
            raise ValueError("Op_sqrt currently supports int8_t test generation only")

        input_min = params.get("input_min", 0)
        input_max = params.get("input_max", 127)
        random_data = np.random.randint(input_min, input_max + 1, size=input_shape).astype(np.int8)
        tensors["input_tensor"] = random_data

        interpreter = Interpreter(
            model_path=str(tflite_fname),
            experimental_op_resolver_type=OpResolverType.BUILTIN_REF
        )
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]
        input_index = input_details["index"]
        output_index = output_details["index"]
        input_scale, input_zero_point = input_details["quantization"]
        output_scale, output_zero_point = output_details["quantization"]

        generated_params["in_dim"] = [
            params["batch_size"],
            params["height_1"],
            params["width_1"],
            params["channel_1"]
        ]
        generated_params["output_len"] = int(np.prod(output_details["shape"]))
        generated_params["input_scale"] = float(input_scale)
        generated_params["input_zero_point"] = int(input_zero_point)
        generated_params["output_scale"] = float(output_scale)
        generated_params["output_zero_point"] = int(output_zero_point)

        # Run model on random input and capture output.
        interpreter.set_tensor(input_index, random_data)
        interpreter.invoke()
        out_data = interpreter.get_tensor(output_index).astype(np.int8)
        tensors["output"] = out_data

        return Lib.op_utils.Generated_data(
            generated_params,
            tensors,
            scales,
            effective_scales,
            aliases
        )

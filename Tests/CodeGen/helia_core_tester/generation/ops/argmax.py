"""
ArgMax operation implementation.
"""

from typing import Dict, Any
import numpy as np
import tensorflow as tf
from pathlib import Path
from helia_core_tester.generation.ops.base import OperationBase


class OpArgMax(OperationBase):
    """
    ArgMax operation.
    """
    
    def build_keras_model(self) -> tf.keras.Model:
        """Build Keras model for ArgMax operation."""
        input_shape = self.desc['input_shape']
        inputs = tf.keras.Input(shape=input_shape[1:], dtype=tf.float32, name='input')
        
        # Get axis from descriptor (default to last dimension)
        axis = self.desc.get('axis', -1)
        
        # ArgMax operation - returns int64 indices
        x = tf.keras.layers.Lambda(
            lambda x: tf.argmax(x, axis=axis, output_type=tf.int32),
            name='argmax'
        )(inputs)
        
        model = tf.keras.Model(inputs=inputs, outputs=x)
        return model

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        """Convert Keras model to TFLite with quantization."""
        from helia_core_tester.generation.utils.litert_builder import build_arg_op
        activation_dtype = self.desc.get("activation_dtype", "S8")
        if activation_dtype == "S8":
            dtype = "int8"
        elif activation_dtype == "S16":
            dtype = "int16"
        else:
            raise NotImplementedError(f"Unsupported ArgMax dtype: {activation_dtype}")
        model_bytes = build_arg_op(
            "argmax",
            input_shape=self.desc["input_shape"],
            axis=self.desc.get("axis", -1),
            dtype=dtype,
            seed=rep_seed,
        )
        with open(out_path, "wb") as f:
            f.write(model_bytes)
    
    def _select_cmsis_argmax_kernel(self) -> Dict[str, str]:
        """
        Select appropriate CMSIS-NN kernel function for ArgMax operation.
        
        Returns:
            Dictionary with kernel_fn, input_c_type, output_c_type
        """
        activation_dtype = self.desc.get('activation_dtype', 'S8')
        
        if activation_dtype == 'S8':
            return {
                'kernel_fn': 'arm_argmax_s8',
                'input_c_type': 'int8_t',
                'output_c_type': 'int32_t'
            }
        elif activation_dtype == 'S16':
            return {
                'kernel_fn': 'arm_argmax_s16',
                'input_c_type': 'int16_t',
                'output_c_type': 'int32_t'
            }
        else:
            raise NotImplementedError(f"Unsupported ArgMax dtype: {activation_dtype}")
    
    def generate_c_files(self, output_dir: Path) -> None:
        """
        Generate C and H files from templates for ArgMax operation.
        """
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder
        
        name = self.desc['name']
        tflite_path = output_dir / f"{name}.tflite"
        if not tflite_path.exists():
            raise FileNotFoundError(f"TFLite file not found: {tflite_path}")
        
        # Select CMSIS kernel + types
        kernel_info = self._select_cmsis_argmax_kernel()
        
        input_shape = tuple(self.desc["input_shape"])
        
        builder = TemplateContextBuilder()
        
        # Convert shapes to CMSIS dims
        input_dims = builder.nhwc_to_cmsis_dims(input_shape)
        
        # Extract axis from descriptor (default to -1 for last dimension)
        axis = self.desc.get('axis', -1)
        # Convert to 0-based and account for batch dimension
        # For NHWC: 0=N, 1=H, 2=W, 3=C
        if axis < 0:
            axis = len(input_shape) + axis
        # axis is now 0-3 for NHWC
        
        # Generate deterministic integer input data
        rng_state = self.rng.__getstate__()
        self.rng = np.random.default_rng(self.seed)

        if kernel_info["input_c_type"] == "int8_t":
            np_in_dtype = np.int8
            qmin, qmax = -128, 127
        elif kernel_info["input_c_type"] == "int16_t":
            np_in_dtype = np.int16
            qmin, qmax = -32768, 32767
        else:
            raise ValueError(f"Unsupported input_c_type: {kernel_info['input_c_type']}")
        input_q = self.rng.integers(qmin, qmax + 1, size=input_shape, dtype=np_in_dtype)
        self.rng.__setstate__(rng_state)

        # Compute expected output directly
        output_data = np.argmax(input_q, axis=axis).astype(np.int32)
        output_shape = tuple(output_data.shape)
        
        # Format arrays
        input_array_str = builder.format_array_as_c_literal(input_q)
        expected_output_array_str = builder.format_array_as_c_literal(output_data)
        
        # Build template context
        context = {
            'name': name,
            'prefix': name,
            'input_dims': input_dims,
            'axis': axis,
            'input_data_array': input_array_str,
            'expected_output_array': expected_output_array_str,
            'input_dtype': kernel_info["input_c_type"],
            'output_dtype': kernel_info["output_c_type"],
            'kernel_fn': kernel_info["kernel_fn"],
            'output_size': int(np.prod(output_shape)),
        }
        
        # Render templates
        includes_api_dir = output_dir / "includes"
        includes_api_dir.mkdir(parents=True, exist_ok=True)
        
        h_content = self.render_template("argmax/argmax.h.j2", context)
        h_path = includes_api_dir / f"{name}_argmax.h"
        with open(h_path, 'w') as f:
            f.write(h_content)
        
        c_content = self.render_template("argmax/argmax.c.j2", context)
        c_path = output_dir / f"{name}_argmax.c"
        with open(c_path, 'w') as f:
            f.write(c_content)
        
        cmake_context = {
            'name': name,
            'operator': self.desc.get('operator', 'ArgMax'),
            'operator_name': 'argmax'
        }
        cmake_content = self.render_template("common/CMakeLists.txt.j2", cmake_context)
        cmake_path = output_dir / "CMakeLists.txt"
        with open(cmake_path, 'w') as f:
            f.write(cmake_content)
        
        print(f"Generated C/H files and CMakeLists.txt for {name}")

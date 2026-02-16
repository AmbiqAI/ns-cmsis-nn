"""
ReduceMax operation implementation.
"""

from typing import Dict, Any
import numpy as np
import tensorflow as tf
from pathlib import Path
from helia_core_tester.generation.ops.base import OperationBase


class OpReduceMax(OperationBase):
    """
    ReduceMax operation.
    """
    
    def build_keras_model(self) -> tf.keras.Model:
        """Build Keras model for ReduceMax operation."""
        input_shape = self.desc['input_shape']
        inputs = tf.keras.Input(shape=input_shape[1:], dtype=tf.float32, name='input')
        
        # Get axes and keepdims from descriptor
        axes = self.desc.get('axes', [1, 2])  # Default to spatial dimensions
        keepdims = self.desc.get('keepdims', True)
        
        # ReduceMax operation
        x = tf.keras.layers.Lambda(
            lambda x: tf.reduce_max(x, axis=axes, keepdims=keepdims),
            name='reduce_max'
        )(inputs)
        
        model = tf.keras.Model(inputs=inputs, outputs=x)
        return model

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        """Convert Keras model to TFLite with quantization."""
        super().convert_to_tflite(model, out_path, rep_seed)
    
    def _select_cmsis_reduce_max_kernel(self) -> Dict[str, str]:
        """
        Select appropriate CMSIS-NN kernel function for ReduceMax operation.
        
        Returns:
            Dictionary with kernel_fn, input_c_type, output_c_type
        """
        activation_dtype = self.desc.get('activation_dtype', 'S8')
        
        if activation_dtype == 'S8':
            return {
                'kernel_fn': 'arm_reduce_max_s8',
                'input_c_type': 'int8_t',
                'output_c_type': 'int8_t'
            }
        elif activation_dtype == 'S16':
            return {
                'kernel_fn': 'arm_reduce_max_s16',
                'input_c_type': 'int16_t',
                'output_c_type': 'int16_t'
            }
        else:
            raise NotImplementedError(f"Unsupported ReduceMax dtype: {activation_dtype}")
    
    def generate_c_files(self, output_dir: Path) -> None:
        """
        Generate C and H files from templates for ReduceMax operation.
        """
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder
        
        name = self.desc['name']
        tflite_path = output_dir / f"{name}.tflite"
        if not tflite_path.exists():
            raise FileNotFoundError(f"TFLite file not found: {tflite_path}")
        
        # Select CMSIS kernel + types
        kernel_info = self._select_cmsis_reduce_max_kernel()
        
        # Load interpreter
        interpreter = self.load_litert_interpreter(str(tflite_path))
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        input_shape = tuple(input_details[0]['shape'])
        output_shape = tuple(output_details[0]['shape'])
        
        builder = TemplateContextBuilder()
        
        # Convert shapes to CMSIS dims
        input_dims = builder.nhwc_to_cmsis_dims(input_shape)
        output_dims = builder.nhwc_to_cmsis_dims(output_shape)
        
        # Extract axes from descriptor (default to [1, 2] for spatial dimensions)
        axes = self.desc.get('axes', [1, 2])
        if not isinstance(axes, list):
            axes = [axes]
        
        # Calculate axis_dims from axes
        # axis_dims represents which dimensions are being reduced - it's a bitmap/mask
        # For NHWC: 0=N, 1=H, 2=W, 3=C
        # CMSIS-NN expects: axis_dims has 1 in positions being reduced, 0 otherwise
        # Based on CMSIS-NN test: {1, 0, 0, 0} means reduce axis 0 (N dimension)
        axis_dims = [0, 0, 0, 0]  # Default: no reduction
        for axis in axes:
            norm_axis = axis
            if norm_axis < 0:
                norm_axis = len(input_shape) + norm_axis
            if 0 <= norm_axis < 4:
                axis_dims[norm_axis] = 1  # Mark this dimension for reduction
        
        # CMSIS dims are NHWC, so just wrap in dict (no reorder needed for axis mask)
        axis_dims_cmsis = {'n': axis_dims[0], 'h': axis_dims[1], 'w': axis_dims[2], 'c': axis_dims[3]}
        
        # Generate input data and quantize
        rng_state = self.rng.__getstate__()
        self.rng = np.random.default_rng(self.seed)
        
        input_data = self.rng.uniform(-1.0, 1.0, size=input_shape).astype(np.float32)
        
        self.rng.__setstate__(rng_state)
        
        # Extract quantization
        input_qp = input_details[0].get('quantization_parameters', {})
        input_scale = input_qp.get('scales', [1.0])
        input_zp = input_qp.get('zero_points', [0])
        input_scale = float(input_scale[0] if isinstance(input_scale, list) else input_scale)
        input_zp = int(input_zp[0] if isinstance(input_zp, list) else input_zp)
        
        # Quantize inputs
        if kernel_info["input_c_type"] == "int8_t":
            np_in_dtype = np.int8
            qmin, qmax = -128, 127
        elif kernel_info["input_c_type"] == "int16_t":
            np_in_dtype = np.int16
            qmin, qmax = -32768, 32767
        else:
            raise ValueError(f"Unsupported input_c_type: {kernel_info['input_c_type']}")
        
        input_q = np.round(input_data / float(input_scale) + float(input_zp)).astype(np.int32)
        input_q = np.clip(input_q, qmin, qmax).astype(np_in_dtype)
        
        # Run inference
        interpreter.set_tensor(input_details[0]['index'], input_q)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        output_data = np.array(output_data)
        
        # Format arrays
        input_array_str = builder.format_array_as_c_literal(input_q)
        expected_output_array_str = builder.format_array_as_c_literal(output_data)
        
        # Build template context
        context = {
            'name': name,
            'prefix': name,
            'input_dims': input_dims,
            'output_dims': output_dims,
            'axis_dims': axis_dims_cmsis,
            'input_data_array': input_array_str,
            'expected_output_array': expected_output_array_str,
            'input_dtype': kernel_info["input_c_type"],
            'output_dtype': kernel_info["output_c_type"],
            'kernel_fn': kernel_info["kernel_fn"],
        }
        
        # Render templates
        includes_api_dir = output_dir / "includes"
        includes_api_dir.mkdir(parents=True, exist_ok=True)
        
        h_content = self.render_template("reducemax/reducemax.h.j2", context)
        h_path = includes_api_dir / f"{name}_reducemax.h"
        with open(h_path, 'w') as f:
            f.write(h_content)
        
        c_content = self.render_template("reducemax/reducemax.c.j2", context)
        c_path = output_dir / f"{name}_reducemax.c"
        with open(c_path, 'w') as f:
            f.write(c_content)
        
        cmake_context = {
            'name': name,
            'operator': self.desc.get('operator', 'ReduceMax'),
            'operator_name': 'reducemax'
        }
        cmake_content = self.render_template("common/CMakeLists.txt.j2", cmake_context)
        cmake_path = output_dir / "CMakeLists.txt"
        with open(cmake_path, 'w') as f:
            f.write(cmake_content)
        
        print(f"Generated C/H files and CMakeLists.txt for {name}")

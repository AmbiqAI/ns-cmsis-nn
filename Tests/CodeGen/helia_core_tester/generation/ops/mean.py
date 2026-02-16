"""
Mean operation implementation for Helia-Core Tester.
"""

import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Dict
from helia_core_tester.generation.ops.base import OperationBase  


class OpMean(OperationBase):
    """
    Mean operation.
    """
    
    def build_keras_model(self) -> tf.keras.Model:
        """Build Keras model for Mean operation."""
        input_shape = self.desc['input_shape']
        
        inputs = tf.keras.Input(shape=input_shape[1:], dtype=tf.float32, name='input')
        
        # Get axes and keepdims from descriptor
        axes = self.desc.get('axes', [1, 2])  # Default to spatial dimensions
        keepdims = self.desc.get('keepdims', True)
        
        # Mean operation
        output = tf.keras.layers.Lambda(
            lambda x: tf.reduce_mean(x, axis=axes, keepdims=keepdims),
            name='reduce_mean'
        )(inputs)
        
        model = tf.keras.Model(inputs=inputs, outputs=output)
        return model

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        """Convert Keras model to TFLite with quantization."""
        super().convert_to_tflite(model, out_path, rep_seed)
    
    def _select_cmsis_mean_kernel(self) -> Dict[str, str]:
        """
        Select appropriate CMSIS-NN kernel function for Mean operation.
        
        Returns:
            Dictionary with kernel_fn, input_c_type, output_c_type
        """
        activation_dtype = self.desc.get('activation_dtype', 'S8')
        
        if activation_dtype == 'S8':
            return {
                'kernel_fn': 'arm_mean_s8',
                'input_c_type': 'int8_t',
                'output_c_type': 'int8_t'
            }
        elif activation_dtype == 'S16':
            return {
                'kernel_fn': 'arm_mean_s16',
                'input_c_type': 'int16_t',
                'output_c_type': 'int16_t'
            }
        else:
            raise NotImplementedError(f"Unsupported Mean dtype: {activation_dtype}")
    
    def generate_c_files(self, output_dir: Path) -> None:
        """
        Generate C and H files from templates for Mean operation.
        """
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder
        from helia_core_tester.generation.utils.tflite_utils import calculate_multiplier_shift
        
        name = self.desc['name']
        tflite_path = output_dir / f"{name}.tflite"
        if not tflite_path.exists():
            raise FileNotFoundError(f"TFLite file not found: {tflite_path}")
        
        # Select CMSIS kernel + types
        kernel_info = self._select_cmsis_mean_kernel()
        
        # Load LiteRT model for shape and quantization extraction
        from helia_core_tester.generation.utils.litert_utils import get_operator_tensors_from_litert
        model, subgraph = self.load_litert_model(str(tflite_path))
        op_tensors = get_operator_tensors_from_litert(model, subgraph, 0)
        
        # Extract shapes from LiteRT
        input_shape = op_tensors['inputs'][0]['shape']
        output_shape = op_tensors['outputs'][0]['shape']
        
        # Ensure shapes are tuples
        if input_shape is not None:
            input_shape = tuple(input_shape)
        if output_shape is not None:
            output_shape = tuple(output_shape)
        
        builder = TemplateContextBuilder()
        
        # Convert shapes to CMSIS dims
        input_dims = builder.nhwc_to_cmsis_dims(input_shape)
        output_dims = builder.nhwc_to_cmsis_dims(output_shape)
        
        # Extract axes from descriptor (default to [1, 2] for spatial dimensions)
        axes = self.desc.get('axes', [1, 2])
        if not isinstance(axes, list):
            axes = [axes]
        
        # Normalize axes (handle negative indices)
        normalized_axes = []
        for axis in axes:
            if axis < 0:
                axis = len(input_shape) + axis
            if 0 <= axis < len(input_shape):
                normalized_axes.append(axis)
        
        # Calculate axis_dims: bitmap where 1 indicates a reduced dimension
        # For NHWC: 0=N, 1=H, 2=W, 3=C
        # axis_dims should be [n, h, w, c] where 1 means reduced, 0 means not reduced
        axis_dims = [0, 0, 0, 0]  # Initialize: no reduction
        for axis in normalized_axes:
            if 0 <= axis < 4:
                axis_dims[axis] = 1  # Mark as reduced
        
        axis_dims_cmsis = builder.nhwc_to_cmsis_dims(axis_dims)
        
        # Extract quantization from LiteRT
        input_quant = op_tensors['inputs'][0]['quantization']
        output_quant = op_tensors['outputs'][0]['quantization']
        
        input_scale = input_quant.get('scale', 1.0)
        input_zp = input_quant.get('zero_point', 0)
        output_scale = output_quant.get('scale', 1.0)
        output_zp = output_quant.get('zero_point', 0)
        
        # Handle per-channel quantization (convert to scalar)
        if isinstance(input_scale, (list, np.ndarray)):
            input_scale = float(input_scale[0])
        if isinstance(input_zp, (list, np.ndarray)):
            input_zp = int(input_zp[0])
        if isinstance(output_scale, (list, np.ndarray)):
            output_scale = float(output_scale[0])
        if isinstance(output_zp, (list, np.ndarray)):
            output_zp = int(output_zp[0])
        
        input_scale = float(input_scale)
        input_zp = int(input_zp)
        output_scale = float(output_scale)
        output_zp = int(output_zp)
        
        # Calculate reduction size (number of elements being averaged)
        # IMPORTANT: Use the actual TFLite model's input_shape, not the descriptor's shape,
        # because TFLite may have removed or modified the batch dimension during conversion
        reduction_size = 1
        for axis in normalized_axes:
            if 0 <= axis < len(input_shape):
                # Use the actual dimension size from the TFLite model
                reduction_size *= input_shape[axis]
        
        # Convert to Python int for bit_length() method
        reduction_size = int(reduction_size)
        
        # For Mean: base_scale = input_scale / output_scale
        # Then we need to fold 1 / reduction_size into the multiplier/shift
        base_scale = input_scale / output_scale
        
        # Calculate base multiplier and shift
        base_mult, base_shift = calculate_multiplier_shift(base_scale)
        
        # Fold 1 / reduction_size into multiplier/shift
        Q31_MAX = (1 << 31) - 1
        
        # Calculate initial shift: 63 - (count.bit_length() - 1)
        shift = 63 - (reduction_size.bit_length() - 1)
        shift = min(shift, 32)
        shift = min(shift, 31 + base_shift)  # ensure we don't overflow Q31
        
        folded_mult = None
        folded_shift = None
        while shift >= 0:
            folded_mult = ((base_mult << shift) + (reduction_size // 2)) // reduction_size
            if folded_mult <= Q31_MAX:
                folded_shift = base_shift - shift
                break
            shift -= 1  # try smaller shift to prevent overflow
        
        # Use folded values if found, otherwise fall back to base
        if folded_mult is not None and folded_mult > 0:
            out_mult = int(folded_mult)
            out_shift = int(folded_shift)
        else:
            out_mult = base_mult
            out_shift = base_shift
        
        if out_mult == 0 or out_shift < -31:
            effective_scale = base_scale / float(reduction_size)
            out_mult, out_shift = calculate_multiplier_shift(effective_scale)
        
        # Generate input data and quantize
        rng_state = self.rng.__getstate__()
        self.rng = np.random.default_rng(self.seed)
        
        input_data = self.rng.uniform(-1.0, 1.0, size=input_shape).astype(np.float32)
        
        self.rng.__setstate__(rng_state)
        
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
        
        # Run inference using LiteRT interpreter
        interpreter = self.load_litert_interpreter(str(tflite_path))
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
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
            'input_offset': int(input_zp),
            'out_offset': int(output_zp),
            'out_mult': int(out_mult),
            'out_shift': int(out_shift),
            'input_data_array': input_array_str,
            'expected_output_array': expected_output_array_str,
            'input_dtype': kernel_info["input_c_type"],
            'output_dtype': kernel_info["output_c_type"],
            'kernel_fn': kernel_info["kernel_fn"],
        }
        
        # Render templates
        includes_api_dir = output_dir / "includes"
        includes_api_dir.mkdir(parents=True, exist_ok=True)
        
        h_content = self.render_template("mean/mean.h.j2", context)
        h_path = includes_api_dir / f"{name}_mean.h"
        with open(h_path, 'w') as f:
            f.write(h_content)
        
        c_content = self.render_template("mean/mean.c.j2", context)
        c_path = output_dir / f"{name}_mean.c"
        with open(c_path, 'w') as f:
            f.write(c_content)
        
        cmake_context = {
            'name': name,
            'operator': self.desc.get('operator', 'Mean'),
            'operator_name': 'mean'
        }
        cmake_content = self.render_template("common/CMakeLists.txt.j2", cmake_context)
        cmake_path = output_dir / "CMakeLists.txt"
        with open(cmake_path, 'w') as f:
            f.write(cmake_content)
        
        print(f"Generated C/H files and CMakeLists.txt for {name}")

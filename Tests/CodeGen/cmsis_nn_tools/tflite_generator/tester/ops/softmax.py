"""
Softmax operation implementation for Helia-Core Tester.
"""

import math
from typing import Dict, Any
import numpy as np
import tensorflow as tf
from pathlib import Path
from .base import OperationBase  


class OpSoftmax(OperationBase):
    """
    Softmax operation.
    """
    
    def build_keras_model(self) -> tf.keras.Model:
        """Build Keras model for Softmax operation."""
        input_shape = self.desc['input_shape']
        
        inputs = tf.keras.Input(shape=input_shape[1:], dtype=tf.float32, name='input')
        
        # Softmax operation
        output = tf.keras.layers.Softmax()(inputs)
        
        model = tf.keras.Model(inputs=inputs, outputs=output)
        return model

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        """Convert Keras model to TFLite with quantization."""
        
        # Create converter
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Apply quantization based on activation_dtype
        activation_dtype = self.desc.get('activation_dtype', 'S8')
        
        if activation_dtype == 'S8':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.int8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
        elif activation_dtype == 'S16':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
            converter.inference_input_type  = tf.int16
            converter.inference_output_type = tf.int16

        
        # Generate representative dataset
        def representative_data_gen():
            for _ in range(100):
                if 'input_shape' in self.desc:
                    inputs = self.rng.uniform(-1.0, 1.0, size=self.desc['input_shape']).astype(np.float32)
                    yield [inputs]
                elif 'input_1_shape' in self.desc and 'input_2_shape' in self.desc:
                    inputs1 = self.rng.uniform(-1.0, 1.0, size=self.desc['input_1_shape']).astype(np.float32)
                    inputs2 = self.rng.uniform(-1.0, 1.0, size=self.desc['input_2_shape']).astype(np.float32)
                    yield [inputs1, inputs2]
        
        converter.representative_dataset = representative_data_gen
        
        # Convert and save
        tflite_model = converter.convert()
        with open(out_path, 'wb') as f:
            f.write(tflite_model)
    
    def _select_cmsis_softmax_kernel(self) -> Dict[str, str]:
        """
        Select appropriate CMSIS-NN kernel function for Softmax operation.
        
        Returns:
            Dictionary with kernel_fn, input_c_type, output_c_type
        """
        activation_dtype = self.desc.get('activation_dtype', 'S8')
        
        if activation_dtype == 'S8':
            return {
                'kernel_fn': 'arm_softmax_s8',
                'input_c_type': 'int8_t',
                'output_c_type': 'int8_t'
            }
        elif activation_dtype == 'S16':
            return {
                'kernel_fn': 'arm_softmax_s16',
                'input_c_type': 'int16_t',
                'output_c_type': 'int16_t'
            }
        else:
            raise NotImplementedError(f"Unsupported Softmax dtype: {activation_dtype}")
    
    def generate_c_files(self, output_dir: Path) -> None:
        """
        Generate C and H files from templates for Softmax operation.
        """
        from ..utils.template_context import TemplateContextBuilder
        from ..utils.tflite_utils import calculate_multiplier_shift
        
        name = self.desc['name']
        tflite_path = output_dir / f"{name}.tflite"
        if not tflite_path.exists():
            raise FileNotFoundError(f"TFLite file not found: {tflite_path}")
        
        # Select CMSIS kernel + types
        kernel_info = self._select_cmsis_softmax_kernel()
        
        # Load interpreter
        interpreter = self.load_tflite_interpreter(str(tflite_path))
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        input_shape = tuple(input_details[0]['shape'])
        output_shape = tuple(output_details[0]['shape'])
        
        # Extract quantization directly from tensor details
        input_qp = input_details[0].get('quantization_parameters', {})
        output_qp = output_details[0].get('quantization_parameters', {})
        
        input_scale = input_qp.get('scales', [1.0])
        input_zp = input_qp.get('zero_points', [0])
        output_scale = output_qp.get('scales', [1.0])
        output_zp = output_qp.get('zero_points', [0])
        
        # Get first element (per-tensor quantization)
        input_scale = float(input_scale[0] if isinstance(input_scale, list) else input_scale)
        input_zp = int(input_zp[0] if isinstance(input_zp, list) else input_zp)
        output_scale = float(output_scale[0] if isinstance(output_scale, list) else output_scale)
        output_zp = int(output_zp[0] if isinstance(output_zp, list) else output_zp)
        
        builder = TemplateContextBuilder()
        
        # Convert shapes to CMSIS dims
        input_dims = builder.nhwc_to_cmsis_dims(input_shape)
        output_dims = builder.nhwc_to_cmsis_dims(output_shape)
        
        # Calculate multipliers and shifts for softmax
        # - For S8: use preprocess_softmax_scaling: beta * input_scale * (1 << 26)
        # - For S16: use input_scale_beta_rescale = beta * input_scale / (10.0 / 65535.0)
        softmax_input_integer_bits = 5  # scaled_diff_integer_bits, matches ns-cmsis-nn
        beta = 1.0  # softmax beta parameter (typically 1.0)
        
        if kernel_info["input_c_type"] == "int8_t":
            # S8: preprocess_softmax_scaling
            # input_beta_real_multiplier = min(beta * input_scale * (1 << (31 - scaled_diff_integer_bits)), max)
            max_real_multiplier = (1 << 31) - 1
            input_real_multiplier = min(beta * input_scale * (1 << (31 - softmax_input_integer_bits)), max_real_multiplier)
        else:
            # S16: input_scale_beta_rescale
            # input_scale_beta_rescale = beta * input_scale / (10.0 / 65535.0)
            input_scale_beta_rescale = beta * input_scale / (10.0 / 65535.0)
            input_real_multiplier = input_scale_beta_rescale
        
        mult, shift = calculate_multiplier_shift(input_real_multiplier)
        
        # Calculate diff_min for s8 softmax
        # diff_min = -1.0 * calculate_input_radius(input_integer_bits, input_left_shift, total_signed_bits=31)
        # where calculate_input_radius = floor(max_val * (1 << (31 - input_integer_bits)) / (1 << input_left_shift))
        # Note: input_left_shift can be negative, so we handle division properly
        if kernel_info["input_c_type"] == "int8_t":
            # calculate_input_radius equivalent
            max_val = (1 << softmax_input_integer_bits) - 1
            if shift >= 0:
                max_input_rescaled = max_val * (1 << (31 - softmax_input_integer_bits)) / (1 << shift)
            else:
                # When shift is negative, (1 << shift) would be fractional, so we multiply instead
                max_input_rescaled = max_val * (1 << (31 - softmax_input_integer_bits)) * (1 << (-shift))
            diff_min = -int(math.floor(max_input_rescaled))
        else:
            diff_min = 0  # Not used for s16
        # Calculate num_rows and row_size
        # Softmax operates on the last dimension (row_size)
        # num_rows is the product of all dimensions except the last
        if len(input_shape) >= 2:
            num_rows = int(np.prod(input_shape[:-1]))
            row_size = int(input_shape[-1])
        else:
            # 1D case
            num_rows = 1
            row_size = int(input_shape[0])
        
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
            'num_rows': num_rows,
            'row_size': row_size,
            'mult': int(mult),
            'shift': int(shift),
            'diff_min': int(diff_min),
            'input_data_array': input_array_str,
            'expected_output_array': expected_output_array_str,
            'input_dtype': kernel_info["input_c_type"],
            'output_dtype': kernel_info["output_c_type"],
            'kernel_fn': kernel_info["kernel_fn"],
            'is_s16': kernel_info["input_c_type"] == "int16_t",
        }
        
        # Render templates
        includes_api_dir = output_dir / "includes"
        includes_api_dir.mkdir(parents=True, exist_ok=True)
        
        h_content = self.render_template("softmax/softmax.h.j2", context)
        h_path = includes_api_dir / f"{name}_softmax.h"
        with open(h_path, 'w') as f:
            f.write(h_content)
        
        c_content = self.render_template("softmax/softmax.c.j2", context)
        c_path = output_dir / f"{name}_softmax.c"
        with open(c_path, 'w') as f:
            f.write(c_content)
        
        cmake_context = {
            'name': name,
            'operator': self.desc.get('operator', 'Softmax'),
            'operator_name': 'softmax'
        }
        cmake_content = self.render_template("common/CMakeLists.txt.j2", cmake_context)
        cmake_path = output_dir / "CMakeLists.txt"
        with open(cmake_path, 'w') as f:
            f.write(cmake_content)
        
        print(f"Generated C/H files and CMakeLists.txt for {name}")
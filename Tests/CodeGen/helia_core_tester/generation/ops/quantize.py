"""
Quantize operation implementation for Helia-Core Tester.

Following the official CMSIS-NN test generator logic from RefactoredTestGen/Lib/op_quantize.py
"""

from typing import Dict, Any
import numpy as np
import tensorflow as tf
from pathlib import Path
from helia_core_tester.generation.ops.base import OperationBase


class OpQuantize(OperationBase):
    """
    Quantize operation.
    """
    
    def build_keras_model(self) -> tf.keras.Model:
        """Build Keras model for Quantize operation.
        
        Creates a model that can be quantized by TFLite converter.
        Uses an identity operation (Lambda layer) to preserve input values.
        """
        input_shape = self.desc['input_shape']
        
        # Build model with float32 inputs (will be quantized later)
        inputs = tf.keras.Input(shape=input_shape[1:], dtype=tf.float32, name='input')
        
        # Use Lambda layer to create an identity operation
        # This will be quantized by TFLite converter
        x = tf.keras.layers.Lambda(lambda x: x, name='identity')(inputs)
        
        # Apply activation if specified
        activation_str = self.desc.get('activation', 'NONE')
        if activation_str == 'RELU':
            x = tf.keras.layers.ReLU()(x)
        elif activation_str == 'RELU6':
            x = tf.keras.layers.ReLU(max_value=6)(x)
        elif activation_str != 'NONE':
            raise ValueError(f"Unsupported activation: {activation_str}")
        
        model = tf.keras.Model(inputs=inputs, outputs=x)
        return model
        

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        """Convert Keras model to TFLite with quantization."""
        self._convert_with_activation_quantization(
            model,
            out_path,
            input_type=tf.float32,
            rep_seed=rep_seed,
        )
    
    def _select_cmsis_quantize_kernel(self) -> Dict[str, str]:
        """
        Select appropriate CMSIS-NN kernel function for Quantize operation.
        
        Returns:
            Dictionary with kernel_fn, input_c_type, output_c_type
        """
        activation_dtype = self.desc.get('activation_dtype', 'S8')
        
        if activation_dtype == 'S8':
            return {
                'kernel_fn': 'arm_quantize_f32_s8',
                'input_c_type': 'float',
                'output_c_type': 'int8_t'
            }
        elif activation_dtype == 'S16':
            return {
                'kernel_fn': 'arm_quantize_f32_s16',
                'input_c_type': 'float',
                'output_c_type': 'int16_t'
            }
        else:
            raise NotImplementedError(f"Unsupported Quantize dtype: {activation_dtype}")
    
    def generate_c_files(self, output_dir: Path) -> None:
        """
        Generate C and H files from templates for Quantize operation.
        """
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder
        
        name = self.desc['name']
        tflite_path = output_dir / f"{name}.tflite"
        if not tflite_path.exists():
            raise FileNotFoundError(f"TFLite file not found: {tflite_path}")
        
        # Select CMSIS kernel + types
        kernel_info = self._select_cmsis_quantize_kernel()
        
        # Load interpreter
        interpreter = self.load_litert_interpreter(str(tflite_path))
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        input_shape = tuple(input_details[0]['shape'])
        output_shape = tuple(output_details[0]['shape'])
        
        builder = TemplateContextBuilder()
        
        # Extract quantization from output (quantized tensor)
        output_qp = output_details[0].get('quantization_parameters', {})
        output_scale = output_qp.get('scales', [1.0])
        output_zp = output_qp.get('zero_points', [0])
        output_scale = float(output_scale[0] if isinstance(output_scale, list) else output_scale)
        output_zp = int(output_zp[0] if isinstance(output_zp, list) else output_zp)
        
        # Generate input data (float32)
        rng_state = self.rng.__getstate__()
        self.rng = np.random.default_rng(self.seed)
        
        input_data = self.rng.uniform(-1.0, 1.0, size=input_shape).astype(np.float32)
        
        self.rng.__setstate__(rng_state)
        
        # Run inference (input is float32, output is quantized)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        output_data = np.array(output_data)
        
        # Format arrays
        input_array_str = builder.format_array_as_c_literal(input_data)
        expected_output_array_str = builder.format_array_as_c_literal(output_data)
        
        # Calculate size
        input_size = int(np.prod(input_shape))
        
        # Check for activation in descriptor
        activation_str = self.desc.get('activation', 'NONE')
        has_activation = activation_str in ['RELU', 'RELU6']
        
        # Select activation kernel function if needed
        activation_kernel_fn = None
        if has_activation:
            if activation_str == 'RELU':
                if kernel_info["output_c_type"] == "int8_t":
                    activation_kernel_fn = 'arm_relu_q7'
                else:  # int16_t
                    activation_kernel_fn = 'arm_relu_q15'
            elif activation_str == 'RELU6':
                if kernel_info["output_c_type"] == "int8_t":
                    activation_kernel_fn = 'arm_relu6_q7'
                else:  # int16_t
                    # ReLU6 for int16: use manual clamp
                    # Note: For S16, zero_point should be 0, so we quantize 6.0 as: 6.0 / scale
                    activation_kernel_fn = None
        
        # Build template context
        context = {
            'name': name,
            'prefix': name,
            'input_size': input_size,
            'zero_point': int(output_zp),
            'scale': float(output_scale),
            'input_data_array': input_array_str,
            'expected_output_array': expected_output_array_str,
            'input_dtype': kernel_info["input_c_type"],
            'output_dtype': kernel_info["output_c_type"],
            'kernel_fn': kernel_info["kernel_fn"],
            'has_activation': has_activation,
            'activation_kernel_fn': activation_kernel_fn,
            'activation_type': activation_str if has_activation else 'NONE',
        }
        
        # Render templates
        includes_api_dir = output_dir / "includes"
        includes_api_dir.mkdir(parents=True, exist_ok=True)
        
        h_content = self.render_template("quantize/quantize.h.j2", context)
        h_path = includes_api_dir / f"{name}_quantize.h"
        with open(h_path, 'w') as f:
            f.write(h_content)
        
        c_content = self.render_template("quantize/quantize.c.j2", context)
        c_path = output_dir / f"{name}_quantize.c"
        with open(c_path, 'w') as f:
            f.write(c_content)
        
        cmake_context = {
            'name': name,
            'operator': self.desc.get('operator', 'Quantize'),
            'operator_name': 'quantize'
        }
        cmake_content = self.render_template("common/CMakeLists.txt.j2", cmake_context)
        cmake_path = output_dir / "CMakeLists.txt"
        with open(cmake_path, 'w') as f:
            f.write(cmake_content)
        
        print(f"Generated C/H files and CMakeLists.txt for {name}")

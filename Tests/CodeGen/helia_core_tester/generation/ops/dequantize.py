"""
Dequantize operation implementation.
"""

from typing import Dict, Any
import numpy as np
import tensorflow as tf
from pathlib import Path
from helia_core_tester.generation.ops.base import OperationBase


class OpDequantize(OperationBase):
    """
    Dequantize operation.
    """
    
    def build_keras_model(self) -> tf.keras.Model:
        """Build Keras model for Dequantize operation."""
        input_shape = self.desc['input_shape']
        inputs = tf.keras.Input(shape=input_shape[1:], dtype=tf.float32, name='input')
        x = tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32))(inputs)
        
        # Apply activation if specified (check descriptor or infer from name)
        activation_str = self.desc.get('activation', 'NONE')
        # Also check if name suggests activation (for backward compatibility)
        if activation_str == 'NONE' and 'relu' in self.desc.get('name', '').lower():
            if 'relu6' in self.desc.get('name', '').lower():
                activation_str = 'RELU6'
            else:
                activation_str = 'RELU'
        
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
        # Dequantize produces float32 output
        self._convert_with_activation_quantization(
            model,
            out_path,
            output_type=tf.float32,
            rep_seed=rep_seed,
        )
    
    def _select_cmsis_dequantize_kernel(self) -> Dict[str, str]:
        """
        Select appropriate CMSIS-NN kernel function for Dequantize operation.
        
        Returns:
            Dictionary with kernel_fn, input_c_type, output_c_type
        """
        activation_dtype = self.desc.get('activation_dtype', 'S8')
        
        if activation_dtype == 'S8':
            return {
                'kernel_fn': 'arm_dequantize_s8_f32',
                'input_c_type': 'int8_t',
                'output_c_type': 'float'
            }
        elif activation_dtype == 'S16':
            return {
                'kernel_fn': 'arm_dequantize_s16_f32',
                'input_c_type': 'int16_t',
                'output_c_type': 'float'
            }
        else:
            raise NotImplementedError(f"Unsupported Dequantize dtype: {activation_dtype}")
    
    def generate_c_files(self, output_dir: Path) -> None:
        """
        Generate C and H files from templates for Dequantize operation.
        """
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder
        
        name = self.desc['name']
        tflite_path = output_dir / f"{name}.tflite"
        if not tflite_path.exists():
            raise FileNotFoundError(f"TFLite file not found: {tflite_path}")
        
        # Select CMSIS kernel + types
        kernel_info = self._select_cmsis_dequantize_kernel()
        
        # Load interpreter
        interpreter = self.load_litert_interpreter(str(tflite_path))
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        input_shape = tuple(input_details[0]['shape'])
        output_shape = tuple(output_details[0]['shape'])
        
        builder = TemplateContextBuilder()
        
        # Extract quantization from input (quantized tensor)
        input_qp = input_details[0].get('quantization_parameters', {})
        input_scale = input_qp.get('scales', [1.0])
        input_zp = input_qp.get('zero_points', [0])
        
        # Handle array/list scales (take first element for per-tensor assumption)
        if isinstance(input_scale, (list, np.ndarray)):
            if len(input_scale) > 1:
                print(f"Warning: Input has per-channel quantization with {len(input_scale)} scales. Using first scale.")
                input_scale = input_scale[0]
            elif len(input_scale) == 1:
                input_scale = input_scale[0]
            else:
                # Empty array, use default
                input_scale = 1.0
            
        if isinstance(input_zp, (list, np.ndarray)):
            if len(input_zp) > 1:
                print(f"Warning: Input has per-channel quantization with {len(input_zp)} zero points. Using first zero point.")
                input_zp = input_zp[0]
            elif len(input_zp) == 1:
                input_zp = input_zp[0]
            else:
                input_zp = 0
            
        input_scale = float(input_scale)
        input_zp = int(input_zp)
        
        # Generate input data and quantize
        rng_state = self.rng.__getstate__()
        self.rng = np.random.default_rng(self.seed)
        
        # For S16, generate input data that better utilizes the quantization range
        # Calculate the effective float range based on scale and zero_point
        if kernel_info["input_c_type"] == "int8_t":
            np_in_dtype = np.int8
            qmin, qmax = -128, 127
            # For S8, use standard range
            input_data_float = self.rng.uniform(-1.0, 1.0, size=input_shape).astype(np.float32)
        elif kernel_info["input_c_type"] == "int16_t":
            np_in_dtype = np.int16
            qmin, qmax = -32768, 32767
            # For S16, generate data that will quantize to a good range of int16 values
            # Calculate float range that maps to a reasonable subset of int16 range
            # Use approximately 80% of the int16 range to avoid edge cases
            range_fraction = 0.8
            effective_qmin = int(qmin + (1.0 - range_fraction) * (qmax - qmin) / 2)
            effective_qmax = int(qmax - (1.0 - range_fraction) * (qmax - qmin) / 2)
            float_min = (effective_qmin - input_zp) * input_scale
            float_max = (effective_qmax - input_zp) * input_scale
            # Ensure we have a valid range
            if float_max > float_min:
                input_data_float = self.rng.uniform(float_min, float_max, size=input_shape).astype(np.float32)
            else:
                # Fallback to standard range if calculated range is invalid
                input_data_float = self.rng.uniform(-1.0, 1.0, size=input_shape).astype(np.float32)
        else:
            raise ValueError(f"Unsupported input_c_type: {kernel_info['input_c_type']}")
        
        self.rng.__setstate__(rng_state)
        
        # Quantize inputs: quantized = round(float_value / scale + zero_point)
        input_q = np.round(input_data_float / float(input_scale) + float(input_zp)).astype(np.int32)
        input_q = np.clip(input_q, qmin, qmax).astype(np_in_dtype)
        
        # Check actual input dtype from TFLite model
        input_dtype = input_details[0]['dtype']
        # If model expects float32 input, we need to use float32 (some dequantize models are built this way)
        if input_dtype == np.float32:
            # Use float32 input directly
            input_tensor = input_data_float
        else:
            # Use quantized input
            input_tensor = input_q
        
        # Run inference (input type depends on model, output should be float32)
        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        output_data = np.array(output_data)
        
        # Check output dtype - if it's quantized, we need to dequantize it
        output_dtype = output_details[0]['dtype']
        if output_dtype != np.float32:
            # Output is still quantized, need to dequantize manually
            output_qp = output_details[0].get('quantization_parameters', {})
            output_scale = output_qp.get('scales', [1.0])
            output_zp = output_qp.get('zero_points', [0])
            
            # Handle array/list scales
            if isinstance(output_scale, (list, np.ndarray)):
                if len(output_scale) > 1:
                    print(f"Warning: Output has per-channel quantization with {len(output_scale)} scales. Using first scale.")
                    output_scale = output_scale[0]
                elif len(output_scale) == 1:
                    output_scale = output_scale[0]
                else:
                    output_scale = 1.0
            
            if isinstance(output_zp, (list, np.ndarray)):
                if len(output_zp) > 1:
                    print(f"Warning: Output has per-channel quantization with {len(output_zp)} zero points. Using first zero point.")
                    output_zp = output_zp[0]
                elif len(output_zp) == 1:
                    output_zp = output_zp[0]
                else:
                    output_zp = 0
            
            output_scale = float(output_scale)
            output_zp = int(output_zp)
            
            # Dequantize: output = (quantized_value - zero_point) * scale
            output_data = (output_data.astype(np.float32) - float(output_zp)) * output_scale
        else:
            # Already float32, just ensure it's the right type
            output_data = output_data.astype(np.float32)
        
        # Format arrays
        input_array_str = builder.format_array_as_c_literal(input_q)
        expected_output_array_str = builder.format_array_as_c_literal(output_data)
        
        # Calculate size
        input_size = int(np.prod(input_shape))
        
        # Check for activation (from descriptor or infer from name)
        # The TFLite model includes any activation op we added during build.
        # The CMSIS-NN arm_dequantize_s16_f32 only does dequantization, so we need to
        # apply ReLU in C code to match TFLite output if the model has activation.
        activation_str = self.desc.get('activation', 'NONE')
        if activation_str == 'NONE' and 'relu' in name.lower():
            if 'relu6' in name.lower():
                activation_str = 'RELU6'
            else:
                activation_str = 'RELU'
        
        has_activation = activation_str in ['RELU', 'RELU6']
        
        # Build template context
        context = {
            'name': name,
            'prefix': name,
            'input_size': input_size,
            'zero_point': int(input_zp),
            'scale': float(input_scale),
            'input_data_array': input_array_str,
            'expected_output_array': expected_output_array_str,
            'input_dtype': kernel_info["input_c_type"],
            'output_dtype': kernel_info["output_c_type"],
            'kernel_fn': kernel_info["kernel_fn"],
            'has_activation': has_activation,
            'activation_type': activation_str if has_activation else 'NONE',
        }
        
        # Render templates
        includes_api_dir = output_dir / "includes"
        includes_api_dir.mkdir(parents=True, exist_ok=True)
        
        h_content = self.render_template("dequantize/dequantize.h.j2", context)
        h_path = includes_api_dir / f"{name}_dequantize.h"
        with open(h_path, 'w') as f:
            f.write(h_content)
        
        c_content = self.render_template("dequantize/dequantize.c.j2", context)
        c_path = output_dir / f"{name}_dequantize.c"
        with open(c_path, 'w') as f:
            f.write(c_content)
        
        cmake_context = {
            'name': name,
            'operator': self.desc.get('operator', 'Dequantize'),
            'operator_name': 'dequantize'
        }
        cmake_content = self.render_template("common/CMakeLists.txt.j2", cmake_context)
        cmake_path = output_dir / "CMakeLists.txt"
        with open(cmake_path, 'w') as f:
            f.write(cmake_content)
        
        print(f"Generated C/H files and CMakeLists.txt for {name}")

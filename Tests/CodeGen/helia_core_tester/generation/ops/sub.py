"""
Subtract operation implementation.
"""

from typing import Dict, Any
import numpy as np
import tensorflow as tf
from pathlib import Path
from helia_core_tester.generation.ops.base import OperationBase


class OpSub(OperationBase):
    """
    Subtract operation.
    """
    
    def build_keras_model(self) -> tf.keras.Model:
        """Build Keras model for Subtract operation."""
        input_1_shape = self.desc['input_1_shape']
        input_2_shape = self.desc['input_2_shape']
        
        # Build model with float32 inputs (will be quantized later)
        input1 = tf.keras.Input(shape=input_1_shape[1:], dtype=tf.float32, name='input1')
        input2 = tf.keras.Input(shape=input_2_shape[1:], dtype=tf.float32, name='input2')
        
        # Subtract operation
        x = tf.keras.layers.Subtract()([input1, input2])
            
        model = tf.keras.Model(inputs=[input1, input2], outputs=x)
        return model

    def _select_cmsis_sub_kernel(self) -> Dict[str, str]:
        """
        Select appropriate CMSIS-NN kernel function for Sub operation.
        
        Returns:
            Dictionary with kernel_fn, input_c_type, output_c_type
        """
        activation_dtype = self.desc.get('activation_dtype', 'S8')
        
        if activation_dtype == 'S8':
            return {
                'kernel_fn': 'arm_sub_s8',
                'input_c_type': 'int8_t',
                'output_c_type': 'int8_t'
            }
        elif activation_dtype == 'S16':
            return {
                'kernel_fn': 'arm_sub_s16',
                'input_c_type': 'int16_t',
                'output_c_type': 'int16_t'
            }
        else:
            raise NotImplementedError(f"Unsupported Sub dtype: {activation_dtype}")
    
    def generate_c_files(self, output_dir: Path) -> None:
        """
        Generate C and H files from templates for Sub operation.
        """
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder
        from helia_core_tester.generation.utils.tflite_utils import calculate_multiplier_shift
        
        name = self.desc['name']
        tflite_path = output_dir / f"{name}.tflite"
        if not tflite_path.exists():
            raise FileNotFoundError(f"TFLite file not found: {tflite_path}")
        
        # Select CMSIS kernel + types
        kernel_info = self._select_cmsis_sub_kernel()
        
        # Load LiteRT model for shape and quantization extraction
        from helia_core_tester.generation.utils.litert_utils import get_operator_tensors_from_litert
        model, subgraph = self.load_litert_model(str(tflite_path))
        op_tensors = get_operator_tensors_from_litert(model, subgraph, 0)
        
        # Extract shapes from LiteRT (multi-input operator)
        input1_shape = op_tensors['inputs'][0]['shape']
        input2_shape = op_tensors['inputs'][1]['shape'] if len(op_tensors['inputs']) > 1 else input1_shape
        output_shape = op_tensors['outputs'][0]['shape']
        
        # Ensure shapes are tuples
        if input1_shape is not None:
            input1_shape = tuple(input1_shape)
        if input2_shape is not None:
            input2_shape = tuple(input2_shape)
        if output_shape is not None:
            output_shape = tuple(output_shape)
        
        # Extract quantization from LiteRT
        input1_quant = op_tensors['inputs'][0]['quantization']
        input2_quant = op_tensors['inputs'][1]['quantization'] if len(op_tensors['inputs']) > 1 else input1_quant
        output_quant = op_tensors['outputs'][0]['quantization']
        
        input1_scale = input1_quant.get('scale', 1.0)
        input1_zp = input1_quant.get('zero_point', 0)
        input2_scale = input2_quant.get('scale', 1.0)
        input2_zp = input2_quant.get('zero_point', 0)
        output_scale = output_quant.get('scale', 1.0)
        output_zp = output_quant.get('zero_point', 0)
        
        # Handle per-channel quantization (convert to scalar)
        if isinstance(input1_scale, (list, np.ndarray)):
            input1_scale = float(input1_scale[0])
        if isinstance(input1_zp, (list, np.ndarray)):
            input1_zp = int(input1_zp[0])
        if isinstance(input2_scale, (list, np.ndarray)):
            input2_scale = float(input2_scale[0])
        if isinstance(input2_zp, (list, np.ndarray)):
            input2_zp = int(input2_zp[0])
        if isinstance(output_scale, (list, np.ndarray)):
            output_scale = float(output_scale[0])
        if isinstance(output_zp, (list, np.ndarray)):
            output_zp = int(output_zp[0])
        
        input1_scale = float(input1_scale)
        input1_zp = int(input1_zp)
        input2_scale = float(input2_scale)
        input2_zp = int(input2_zp)
        output_scale = float(output_scale)
        output_zp = int(output_zp)
        
        builder = TemplateContextBuilder()
        
        # Convert shapes to CMSIS dims
        input1_dims = builder.nhwc_to_cmsis_dims(input1_shape)
        input2_dims = builder.nhwc_to_cmsis_dims(input2_shape)
        output_dims = builder.nhwc_to_cmsis_dims(output_shape)
        
        # For elementwise sub, effective scale for each input is: input_scale / output_scale
        effective_scale1 = float(input1_scale) / float(output_scale)
        effective_scale2 = float(input2_scale) / float(output_scale)
        
        # Calculate multipliers and shifts for each input
        mult1, shift1 = calculate_multiplier_shift(effective_scale1)
        mult2, shift2 = calculate_multiplier_shift(effective_scale2)
        
        # Output multiplier and shift
        output_mult, output_shift = calculate_multiplier_shift(1.0)
        
        # Left shift: typically 0 for sub operations
        left_shift = 0
        
        # Activation min/max based on dtype
        activation_dtype = self.desc.get('activation_dtype', 'S8')
        if activation_dtype == 'S8':
            activation_min = -128
            activation_max = 127
        elif activation_dtype == 'S16':
            activation_min = -32768
            activation_max = 32767
        else:
            activation_min = -128
            activation_max = 127
        
        # Generate input data and quantize
        rng_state = self.rng.__getstate__()
        self.rng = np.random.default_rng(self.seed)
        
        input1_data = self.rng.uniform(-1.0, 1.0, size=input1_shape).astype(np.float32)
        input2_data = self.rng.uniform(-1.0, 1.0, size=input2_shape).astype(np.float32)
        
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
        
        input1_q = np.round(input1_data / float(input1_scale) + float(input1_zp)).astype(np.int32)
        input1_q = np.clip(input1_q, qmin, qmax).astype(np_in_dtype)
        
        input2_q = np.round(input2_data / float(input2_scale) + float(input2_zp)).astype(np.int32)
        input2_q = np.clip(input2_q, qmin, qmax).astype(np_in_dtype)
        
        # Run inference using LiteRT interpreter
        interpreter = self.load_litert_interpreter(str(tflite_path))
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        interpreter.set_tensor(input_details[0]['index'], input1_q)
        interpreter.set_tensor(input_details[1]['index'], input2_q)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        output_data = np.array(output_data)
        
        # Format arrays
        input1_array_str = builder.format_array_as_c_literal(input1_q)
        input2_array_str = builder.format_array_as_c_literal(input2_q)
        expected_output_array_str = builder.format_array_as_c_literal(output_data)
        
        # Build template context
        context = {
            'name': name,
            'prefix': name,
            'input1_dims': input1_dims,
            'input2_dims': input2_dims,
            'output_dims': output_dims,
            'input1_offset': int(input1_zp),
            'input1_mult': int(mult1),
            'input1_shift': int(shift1),
            'input2_offset': int(input2_zp),
            'input2_mult': int(mult2),
            'input2_shift': int(shift2),
            'left_shift': int(left_shift),
            'out_offset': int(output_zp),
            'out_mult': int(output_mult),
            'out_shift': int(output_shift),
            'out_activation_min': int(activation_min),
            'out_activation_max': int(activation_max),
            'input1_data_array': input1_array_str,
            'input2_data_array': input2_array_str,
            'expected_output_array': expected_output_array_str,
            'input_dtype': kernel_info["input_c_type"],
            'output_dtype': kernel_info["output_c_type"],
            'kernel_fn': kernel_info["kernel_fn"],
        }
        
        # Render templates
        includes_api_dir = output_dir / "includes"
        includes_api_dir.mkdir(parents=True, exist_ok=True)
        
        h_content = self.render_template("sub/sub.h.j2", context)
        h_path = includes_api_dir / f"{name}_sub.h"
        with open(h_path, 'w') as f:
            f.write(h_content)
        
        c_content = self.render_template("sub/sub.c.j2", context)
        c_path = output_dir / f"{name}_sub.c"
        with open(c_path, 'w') as f:
            f.write(c_content)
        
        cmake_context = {
            'name': name,
            'operator': self.desc.get('operator', 'Sub'),
            'operator_name': 'sub'
        }
        cmake_content = self.render_template("common/CMakeLists.txt.j2", cmake_context)
        cmake_path = output_dir / "CMakeLists.txt"
        with open(cmake_path, 'w') as f:
            f.write(cmake_content)
        
        print(f"Generated C/H files and CMakeLists.txt for {name}")


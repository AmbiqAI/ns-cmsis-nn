"""
Multiply (elementwise) operation implementation for Helia-Core Tester.
"""

from typing import Dict, Any
import numpy as np
import tensorflow as tf
from pathlib import Path
from helia_core_tester.generation.ops.base import OperationBase 


class OpMul(OperationBase):
    """
    Mul operation.
    """
    
    def build_keras_model(self) -> tf.keras.Model:
        """Build Keras model for Multiply operation."""
        input_1_shape = self.desc['input_1_shape']
        input_2_shape = self.desc['input_2_shape']
        
        input1 = tf.keras.Input(shape=input_1_shape[1:], dtype=tf.float32, name='input1')
        input2 = tf.keras.Input(shape=input_2_shape[1:], dtype=tf.float32, name='input2')
        
        # Multiply operation
        output = tf.keras.layers.Multiply()([input1, input2])
        
        model = tf.keras.Model(inputs=[input1, input2], outputs=output)
        return model

    def _select_cmsis_mul_kernel(self) -> Dict[str, str]:
        """
        Select appropriate CMSIS-NN kernel function for Mul operation.
        
        Returns:
            Dictionary with kernel_fn, input_c_type, output_c_type
        """
        activation_dtype = self.desc.get('activation_dtype', 'S8')
        
        if activation_dtype == 'S8':
            return {
                'kernel_fn': 'arm_elementwise_mul_s8',
                'input_c_type': 'int8_t',
                'output_c_type': 'int8_t'
            }
        elif activation_dtype == 'S16':
            return {
                'kernel_fn': 'arm_elementwise_mul_s16',
                'input_c_type': 'int16_t',
                'output_c_type': 'int16_t'
            }
        else:
            raise NotImplementedError(f"Unsupported Mul dtype: {activation_dtype}")
    
    def generate_c_files(self, output_dir: Path) -> None:
        """
        Generate C and H files from templates for Mul operation.
        """
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder
        from helia_core_tester.generation.utils.tflite_utils import (
    calculate_multiplier_shift,
    scalar_scale_zp,
    activation_bounds,
)
        
        name = self.desc['name']
        tflite_path = output_dir / f"{name}.tflite"
        if not tflite_path.exists():
            raise FileNotFoundError(f"TFLite file not found: {tflite_path}")
        
        # Select CMSIS kernel + types
        kernel_info = self._select_cmsis_mul_kernel()
        
        # Load LiteRT model for shape and quantization extraction
        from helia_core_tester.generation.utils.litert_utils import get_operator_tensors_from_litert
        model, subgraph = self.load_litert_model(str(tflite_path))
        op_tensors = get_operator_tensors_from_litert(model, subgraph, 0)
        
        input1_shape = self._ensure_shape_tuple(op_tensors['inputs'][0]['shape'])
        input2_shape = self._ensure_shape_tuple(
            op_tensors['inputs'][1]['shape'] if len(op_tensors['inputs']) > 1 else op_tensors['inputs'][0]['shape']
        )
        output_shape = self._ensure_shape_tuple(op_tensors['outputs'][0]['shape'])
        
        input1_quant = op_tensors['inputs'][0]['quantization']
        input2_quant = op_tensors['inputs'][1]['quantization'] if len(op_tensors['inputs']) > 1 else input1_quant
        output_quant = op_tensors['outputs'][0]['quantization']
        
        input1_scale, input1_zp = scalar_scale_zp(input1_quant)
        input2_scale, input2_zp = scalar_scale_zp(input2_quant)
        output_scale, output_zp = scalar_scale_zp(output_quant)
        
        builder = TemplateContextBuilder()
        input1_dims = builder.nhwc_to_cmsis_dims(input1_shape)
        input2_dims = builder.nhwc_to_cmsis_dims(input2_shape)
        output_dims = builder.nhwc_to_cmsis_dims(output_shape)
        
        effective_scale = (float(input1_scale) * float(input2_scale)) / float(output_scale)
        output_mult, output_shift = calculate_multiplier_shift(effective_scale)
        
        activation_dtype = self.desc.get('activation_dtype', 'S8')
        activation_min, activation_max = activation_bounds(activation_dtype)
        
        # Generate input data and quantize
        rng_state = self.rng.__getstate__()
        self.rng = np.random.default_rng(self.seed)
        
        input1_data = self.rng.uniform(-1.0, 1.0, size=input1_shape).astype(np.float32)
        input2_data = self.rng.uniform(-1.0, 1.0, size=input2_shape).astype(np.float32)
        
        self.rng.__setstate__(rng_state)
        
        qmin, qmax = activation_bounds(activation_dtype)
        np_in_dtype = np.int16 if activation_dtype == "S16" else np.int8
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
        
        # Calculate block size (total number of elements)
        block_size = int(np.prod(output_shape))
        
        # Build template context
        context = {
            'name': name,
            'prefix': name,
            'input1_dims': input1_dims,
            'input2_dims': input2_dims,
            'output_dims': output_dims,
            # For CMSIS-NN, input offsets should be negated (like other operations)
            # Output offset is used as-is (not negated)
            'input1_offset': -int(input1_zp),
            'input2_offset': -int(input2_zp),
            'out_offset': int(output_zp),
            'out_mult': int(output_mult),
            'out_shift': int(output_shift),
            'out_activation_min': int(activation_min),
            'out_activation_max': int(activation_max),
            'block_size': int(block_size),
            'input1_data_array': input1_array_str,
            'input2_data_array': input2_array_str,
            'expected_output_array': expected_output_array_str,
            'input_dtype': kernel_info["input_c_type"],
            'output_dtype': kernel_info["output_c_type"],
            'kernel_fn': kernel_info["kernel_fn"],
        }
        
        cmake_context = {
            'name': name,
            'operator': self.desc.get('operator', 'Mul'),
            'operator_name': 'mul',
        }
        self._write_op_outputs(output_dir, "mul", "mul/mul.h.j2", "mul/mul.c.j2", context, cmake_context)
        print(f"Generated C/H files and CMakeLists.txt for {name}")
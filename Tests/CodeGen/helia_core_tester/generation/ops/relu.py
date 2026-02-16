"""
ReLU operation implementation for Helia-Core Tester.
"""

from typing import Dict, Any
import numpy as np
import tensorflow as tf
from pathlib import Path
from helia_core_tester.generation.ops.base import OperationBase  


class OpRelu(OperationBase):
    """
    Relu operation.
    """
    
    def build_keras_model(self) -> tf.keras.Model:
        """Build Keras model for ReLU operation."""
        input_shape = self.desc['input_shape']
        
        inputs = tf.keras.Input(shape=input_shape[1:], dtype=tf.float32, name='input')
        
        # ReLU operation
        output = tf.keras.layers.ReLU()(inputs)
        
        model = tf.keras.Model(inputs=inputs, outputs=output)
        return model

    def _select_cmsis_relu_kernel(self) -> Dict[str, str]:
        """
        Select appropriate CMSIS-NN kernel function for Relu operation.
        
        Returns:
            Dictionary with kernel_fn, input_c_type, output_c_type
        """
        activation_dtype = self.desc.get('activation_dtype', 'S8')
        
        if activation_dtype == 'S8':
            return {
                'kernel_fn': 'arm_relu_s8',
                'input_c_type': 'int8_t',
                'output_c_type': 'int8_t'
            }
        elif activation_dtype == 'S16':
            return {
                'kernel_fn': 'arm_relu_s16',
                'input_c_type': 'int16_t',
                'output_c_type': 'int16_t'
            }
        else:
            raise NotImplementedError(f"Unsupported Relu dtype: {activation_dtype}")
    
    def generate_c_files(self, output_dir: Path) -> None:
        """
        Generate C and H files from templates for Relu operation.
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
        kernel_info = self._select_cmsis_relu_kernel()
        
        # Load LiteRT model for shape and quantization extraction
        from helia_core_tester.generation.utils.litert_utils import get_operator_tensors_from_litert
        model, subgraph = self.load_litert_model(str(tflite_path))
        op_tensors = get_operator_tensors_from_litert(model, subgraph, 0)
        
        input_shape = self._ensure_shape_tuple(op_tensors['inputs'][0]['shape'])
        output_shape = self._ensure_shape_tuple(op_tensors['outputs'][0]['shape'])
        
        input_quant = op_tensors['inputs'][0]['quantization']
        output_quant = op_tensors['outputs'][0]['quantization']
        input_scale, input_zp = scalar_scale_zp(input_quant)
        output_scale, output_zp = scalar_scale_zp(output_quant)
        
        builder = TemplateContextBuilder()
        input_dims = builder.nhwc_to_cmsis_dims(input_shape)
        output_dims = builder.nhwc_to_cmsis_dims(output_shape)
        
        # For ReLU, effective scale is: input_scale / output_scale
        # This converts the input to the output scale
        effective_scale = float(input_scale) / float(output_scale)
        
        # Calculate output multiplier and shift
        output_mult, output_shift = calculate_multiplier_shift(effective_scale)
        
        # Generate input data and quantize
        rng_state = self.rng.__getstate__()
        self.rng = np.random.default_rng(self.seed)
        
        input_data = self.rng.uniform(-1.0, 1.0, size=input_shape).astype(np.float32)
        
        self.rng.__setstate__(rng_state)
        
        activation_dtype = self.desc.get('activation_dtype', 'S8')
        qmin, qmax = activation_bounds(activation_dtype)
        np_in_dtype = np.int16 if activation_dtype == "S16" else np.int8
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
        
        # Calculate output size (total number of elements)
        output_size = int(np.prod(output_shape))
        
        # Build template context
        context = {
            'name': name,
            'prefix': name,
            'input_dims': input_dims,
            'output_dims': output_dims,
            'input_offset': int(input_zp),
            'output_offset': int(output_zp),
            'output_mult': int(output_mult),
            'output_shift': int(output_shift),
            'output_size': int(output_size),
            'input_data_array': input_array_str,
            'expected_output_array': expected_output_array_str,
            'input_dtype': kernel_info["input_c_type"],
            'output_dtype': kernel_info["output_c_type"],
            'kernel_fn': kernel_info["kernel_fn"],
        }
        
        cmake_context = {
            'name': name,
            'operator': self.desc.get('operator', 'Relu'),
            'operator_name': 'relu',
        }
        self._write_op_outputs(output_dir, "relu", "relu/relu.h.j2", "relu/relu.c.j2", context, cmake_context)
        print(f"Generated C/H files and CMakeLists.txt for {name}")
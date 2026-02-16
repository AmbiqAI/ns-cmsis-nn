"""
Pooling operation implementation.
"""

from typing import Dict, Any
from pathlib import Path
import numpy as np
import tensorflow as tf
from helia_core_tester.generation.ops.base import OperationBase


class OpPooling(OperationBase):
    """
    Pooling operation.
    """
    
    def build_keras_model(self) -> tf.keras.Model:
        """Build Keras model for Pooling operation."""
        input_shape = self.desc['input_shape']
        
        # Build model with float32 inputs (will be quantized later)
        inputs = tf.keras.Input(shape=input_shape[1:], dtype=tf.float32, name='input')
        
        # Determine pooling type
        pooling_type = self.desc.get('pooling_type', 'AVERAGE')
        
        # Normalize padding to lowercase
        padding = self.desc.get('padding', 'valid')
        if isinstance(padding, str):
            padding = padding.lower()
        
        if pooling_type == 'AVERAGE':
            x = tf.keras.layers.AveragePooling2D(
                pool_size=self.desc.get('pool_size', [2, 2]),
                strides=self.desc.get('strides', [2, 2]),
                padding=padding
            )(inputs)
        elif pooling_type == 'MAX':
            x = tf.keras.layers.MaxPooling2D(
                pool_size=self.desc.get('pool_size', [2, 2]),
                strides=self.desc.get('strides', [2, 2]),
                padding=padding
            )(inputs)
        else:
            raise ValueError(f"Unsupported pooling_type: {pooling_type}")
            
        model = tf.keras.Model(inputs=inputs, outputs=x)
        return model

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        """Convert Keras model to TFLite with quantization."""
        super().convert_to_tflite(model, out_path, rep_seed)
    
    def _select_cmsis_pooling_kernel(self) -> Dict[str, str]:
        """
        Select appropriate CMSIS-NN pooling kernel function.
        
        Returns:
            Dictionary with kernel function name, C types, and buffer size function
        """
        pooling_type = self.desc.get('pooling_type', 'AVERAGE').upper()
        activation_dtype = self.desc.get('activation_dtype', 'S8').upper()
        
        if pooling_type == 'MAX':
            if activation_dtype == 'S8':
                return {
                    'kernel_fn': 'arm_max_pool_s8',
                    'kernel_get_buffer_size_fn': None,  # Max pooling doesn't need buffer
                    'input_c_type': 'int8_t',
                    'output_c_type': 'int8_t',
                }
            elif activation_dtype == 'S16':
                return {
                    'kernel_fn': 'arm_max_pool_s16',
                    'kernel_get_buffer_size_fn': None,
                    'input_c_type': 'int16_t',
                    'output_c_type': 'int16_t',
                }
            else:
                raise NotImplementedError(f"Unsupported MaxPool dtype: {activation_dtype}")
        elif pooling_type == 'AVERAGE':
            if activation_dtype == 'S8':
                return {
                    'kernel_fn': 'arm_avgpool_s8',
                    'kernel_get_buffer_size_fn': 'arm_avgpool_s8_get_buffer_size',
                    'input_c_type': 'int8_t',
                    'output_c_type': 'int8_t',
                }
            elif activation_dtype == 'S16':
                return {
                    'kernel_fn': 'arm_avgpool_s16',
                    'kernel_get_buffer_size_fn': 'arm_avgpool_s16_get_buffer_size',
                    'input_c_type': 'int16_t',
                    'output_c_type': 'int16_t',
                }
            else:
                raise NotImplementedError(f"Unsupported AvgPool dtype: {activation_dtype}")
        else:
            raise ValueError(f"Unsupported pooling_type: {pooling_type}")
    
    def generate_c_files(self, output_dir: Path) -> None:
        """
        Generate C and H files from templates for Pooling operation.
        """
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder
        
        name = self.desc['name']
        tflite_path = output_dir / f"{name}.tflite"
        if not tflite_path.exists():
            raise FileNotFoundError(f"TFLite file not found: {tflite_path}")
        
        # Select CMSIS kernel + types
        kernel_info = self._select_cmsis_pooling_kernel()
        pooling_type = self.desc.get('pooling_type', 'AVERAGE').upper()
        
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
        
        # Extract quantization from LiteRT
        quant_params = {
            'input': op_tensors['inputs'][0]['quantization'],
            'output': op_tensors['outputs'][0]['quantization']
        }
        
        builder = TemplateContextBuilder()
        
        # Convert shapes to CMSIS dims
        input_dims = builder.nhwc_to_cmsis_dims(input_shape)
        output_dims = builder.nhwc_to_cmsis_dims(output_shape)
        
        # Filter dims for pooling: pool_size from descriptor
        pool_size = self.desc.get('pool_size', [2, 2])
        if isinstance(pool_size, (int, float)):
            pool_h = pool_w = int(pool_size)
        else:
            pool_h = int(pool_size[0])
            pool_w = int(pool_size[1])
        
        filter_dims = {
            'n': 1,
            'h': pool_h,
            'w': pool_w,
            'c': 1
        }
        
        # Build pool parameters
        pool_params = builder.build_pool_params(
            self.desc,
            input_shape,
            (pool_h, pool_w),
            output_shape,
            quant_params['output']
        )
        
        # Generate input data and quantize to the interpreter's real input dtype
        # IMPORTANT: Reset RNG to seed to ensure input data matches what was used
        # during TFLite conversion (representative dataset generation may have advanced RNG)
        rng_state = self.rng.__getstate__()
        self.rng = np.random.default_rng(self.seed)
        input_data = self.generate_input_data()
        self.rng.__setstate__(rng_state)
        
        input_scale = quant_params['input'].get('scale', 1.0)
        input_zp = quant_params['input'].get('zero_point', 0)
        if isinstance(input_scale, (list, np.ndarray)):
            input_scale = input_scale[0]
        if isinstance(input_zp, (list, np.ndarray)):
            input_zp = input_zp[0]
        
        if kernel_info["input_c_type"] == "int8_t":
            qmin, qmax = -128, 127
            np_in_dtype = np.int8
        elif kernel_info["input_c_type"] == "int16_t":
            qmin, qmax = -32768, 32767
            np_in_dtype = np.int16
        else:
            raise ValueError(f"Unsupported input_c_type: {kernel_info['input_c_type']}")
        
        input_q = np.round(input_data / float(input_scale) + float(input_zp)).astype(np.int32)
        input_q = np.clip(input_q, qmin, qmax).astype(np_in_dtype)
        
        # Run inference (dtype must match interpreter input)
        # Use LiteRT interpreter for inference
        output_data = self.run_inference(str(tflite_path), input_q)
        
        # Format input and output arrays
        input_data_array_str = builder.format_array_as_c_literal(input_q)
        expected_output_array_str = builder.format_array_as_c_literal(output_data)
        
        # Calculate buffer size max
        activation_dtype = self.desc.get('activation_dtype', 'S8')
        buffer_size_max = builder.calculate_pooling_buffer_size_max(
            input_dims,
            output_dims,
            pooling_type=pooling_type,
            output_dtype=activation_dtype
        )
        
        # Build template context
        context = {
            'name': name,
            'prefix': name,
            'input_dims': input_dims,
            'filter_dims': filter_dims,
            'output_dims': output_dims,
            'pool_params': pool_params,
            'input_data_array': input_data_array_str,
            'expected_output_array': expected_output_array_str,
            'input_dtype': kernel_info["input_c_type"],
            'output_dtype': kernel_info["output_c_type"],
            'kernel_fn': kernel_info["kernel_fn"],
            'kernel_get_buffer_size_fn': kernel_info["kernel_get_buffer_size_fn"],
            'buffer_size_max': buffer_size_max,
            'pooling_type': pooling_type,
        }
        
        # Render templates
        includes_api_dir = output_dir / "includes"
        includes_api_dir.mkdir(parents=True, exist_ok=True)
        
        h_content = self.render_template("pooling/pooling.h.j2", context)
        h_path = includes_api_dir / f"{name}_pooling.h"
        with open(h_path, 'w') as f:
            f.write(h_content)
        
        c_content = self.render_template("pooling/pooling.c.j2", context)
        c_path = output_dir / f"{name}_pooling.c"
        with open(c_path, 'w') as f:
            f.write(c_content)
        
        cmake_context = {
            'name': name,
            'operator': self.desc.get('operator', 'Pooling'),
            'operator_name': 'pooling'
        }
        cmake_content = self.render_template("common/CMakeLists.txt.j2", cmake_context)
        cmake_path = output_dir / "CMakeLists.txt"
        with open(cmake_path, 'w') as f:
            f.write(cmake_content)
        
        print(f"Generated C/H files and CMakeLists.txt for {name}")

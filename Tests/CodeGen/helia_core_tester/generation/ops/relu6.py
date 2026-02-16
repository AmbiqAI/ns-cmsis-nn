"""
ReLU6 operation implementation for Helia-Core Tester.
"""

from typing import Dict, Any
import numpy as np
import tensorflow as tf
from pathlib import Path
from helia_core_tester.generation.ops.base import OperationBase  


class OpRelu6(OperationBase):
    """
    Relu6 operation.
    """
    
    def build_keras_model(self) -> tf.keras.Model:
        """Build Keras model for ReLU6 operation."""
        input_shape = self.desc['input_shape']
        
        inputs = tf.keras.Input(shape=input_shape[1:], dtype=tf.float32, name='input')
        
        # ReLU6 operation
        output = tf.keras.layers.ReLU(max_value=6.0)(inputs)
        
        model = tf.keras.Model(inputs=inputs, outputs=output)
        return model

    def _select_cmsis_relu6_kernel(self) -> Dict[str, str]:
        """
        Select appropriate CMSIS-NN kernel function for Relu6 operation.
        
        Returns:
            Dictionary with kernel_fn, input_c_type, output_c_type
        """
        activation_dtype = self.desc.get('activation_dtype', 'S8')
        
        if activation_dtype == 'S8':
            return {
                'kernel_fn': 'arm_relu_generic_s8',
                'input_c_type': 'int8_t',
                'output_c_type': 'int8_t'
            }
        elif activation_dtype == 'S16':
            return {
                'kernel_fn': 'arm_relu_generic_s16',
                'input_c_type': 'int16_t',
                'output_c_type': 'int16_t'
            }
        else:
            raise NotImplementedError(f"Unsupported Relu6 dtype: {activation_dtype}")
    
    def generate_c_files(self, output_dir: Path) -> None:
        """
        Generate C and H files from templates for Relu6 operation.
        """
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder
        from helia_core_tester.generation.utils.tflite_utils import calculate_multiplier_shift
        
        name = self.desc['name']
        tflite_path = output_dir / f"{name}.tflite"
        if not tflite_path.exists():
            raise FileNotFoundError(f"TFLite file not found: {tflite_path}")
        
        # Select CMSIS kernel + types
        kernel_info = self._select_cmsis_relu6_kernel()
        
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
        
        builder = TemplateContextBuilder()
        
        # Convert shapes to CMSIS dims
        input_dims = builder.nhwc_to_cmsis_dims(input_shape)
        output_dims = builder.nhwc_to_cmsis_dims(output_shape)
        
        # For ReLU6, effective scale is: input_scale / output_scale
        effective_scale = float(input_scale) / float(output_scale)
        
        # Calculate output multiplier and shift
        output_mult, output_shift = calculate_multiplier_shift(effective_scale)
        
        # ReLU6 clamps to [0, 6.0] in float space
        # Quantize these bounds: act_min = quantize(0.0), act_max = quantize(6.0)
        act_min_float = 0.0
        act_max_float = 6.0
        
        act_min = int(np.round(act_min_float / float(output_scale) + float(output_zp)))
        act_max = int(np.round(act_max_float / float(output_scale) + float(output_zp)))
        
        # Clamp to valid range for the data type
        activation_dtype = self.desc.get('activation_dtype', 'S8')
        if activation_dtype == 'S8':
            act_min = max(-128, min(127, act_min))
            act_max = max(-128, min(127, act_max))
        elif activation_dtype == 'S16':
            act_min = max(-32768, min(32767, act_min))
            act_max = max(-32768, min(32767, act_max))
        
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
            'act_min': int(act_min),
            'act_max': int(act_max),
            'output_size': int(output_size),
            'input_data_array': input_array_str,
            'expected_output_array': expected_output_array_str,
            'input_dtype': kernel_info["input_c_type"],
            'output_dtype': kernel_info["output_c_type"],
            'kernel_fn': kernel_info["kernel_fn"],
        }
        
        # Render templates
        includes_api_dir = output_dir / "includes"
        includes_api_dir.mkdir(parents=True, exist_ok=True)
        
        h_content = self.render_template("relu6/relu6.h.j2", context)
        h_path = includes_api_dir / f"{name}_relu6.h"
        with open(h_path, 'w') as f:
            f.write(h_content)
        
        c_content = self.render_template("relu6/relu6.c.j2", context)
        c_path = output_dir / f"{name}_relu6.c"
        with open(c_path, 'w') as f:
            f.write(c_content)
        
        cmake_context = {
            'name': name,
            'operator': self.desc.get('operator', 'Relu6'),
            'operator_name': 'relu6'
        }
        cmake_content = self.render_template("common/CMakeLists.txt.j2", cmake_context)
        cmake_path = output_dir / "CMakeLists.txt"
        with open(cmake_path, 'w') as f:
            f.write(cmake_content)
        
        print(f"Generated C/H files and CMakeLists.txt for {name}")
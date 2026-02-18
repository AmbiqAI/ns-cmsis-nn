"""
Transpose operation implementation.
"""

from typing import Dict, Any
import numpy as np
import tensorflow as tf
from pathlib import Path
from helia_core_tester.generation.ops.base import OperationBase


class OpTranspose(OperationBase):
    """
    Transpose operation.
    """
    
    def build_keras_model(self) -> tf.keras.Model:
        """Build Keras model for Transpose operation."""
        input_shape = self.desc['input_shape']
        inputs = tf.keras.Input(shape=input_shape[1:], dtype=tf.float32, name='input')
        # For 4D input (batch, height, width, channels), transpose height and width
        # This is a common transpose pattern for image data
        x = tf.keras.layers.Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1, 3]))(inputs)
        model = tf.keras.Model(inputs=inputs, outputs=x)
        return model

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        """Convert Keras model to TFLite with quantization."""
        super().convert_to_tflite(model, out_path, rep_seed)
    
    def _select_cmsis_transpose_kernel(self) -> Dict[str, str]:
        """
        Select appropriate CMSIS-NN kernel function for Transpose operation.
        
        Returns:
            Dictionary with kernel_fn, input_c_type, output_c_type
        """
        activation_dtype = self.desc.get('activation_dtype', 'S8')
        
        if activation_dtype == 'S8':
            return {
                'kernel_fn': 'arm_transpose_s8',
                'input_c_type': 'int8_t',
                'output_c_type': 'int8_t'
            }
        elif activation_dtype == 'S16':
            return {
                'kernel_fn': 'arm_transpose_s16',
                'input_c_type': 'int16_t',
                'output_c_type': 'int16_t'
            }
        else:
            raise NotImplementedError(f"Unsupported Transpose dtype: {activation_dtype}")
    
    def generate_c_files(self, output_dir: Path) -> None:
        """
        Generate C and H files from templates for Transpose operation.
        """
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder
        
        name = self.desc['name']
        tflite_path = output_dir / f"{name}.tflite"
        if not tflite_path.exists():
            raise FileNotFoundError(f"TFLite file not found: {tflite_path}")
        
        # Select CMSIS kernel + types
        kernel_info = self._select_cmsis_transpose_kernel()
        
        # Load LiteRT model for shape extraction
        from helia_core_tester.generation.utils.litert_utils import get_operator_tensors_from_litert
        model, subgraph = self.load_litert_model(str(tflite_path))
        op_tensors = get_operator_tensors_from_litert(model, subgraph, 0)
        
        # Extract shapes from LiteRT.
        # Prefer subgraph I/O tensors to avoid picking the permutation tensor.
        input_shape = None
        output_shape = None
        subgraph_input_indices = set(subgraph.inputs or [])
        subgraph_output_indices = set(subgraph.outputs or [])

        for input_tensor_info in op_tensors['inputs']:
            tensor_idx = input_tensor_info.get('index', -1)
            tensor_shape = input_tensor_info.get('shape')
            if tensor_idx in subgraph_input_indices:
                input_shape = tensor_shape
                break

        if input_shape is None and op_tensors['inputs']:
            input_shape = op_tensors['inputs'][0]['shape']

        for output_tensor_info in op_tensors['outputs']:
            tensor_idx = output_tensor_info.get('index', -1)
            tensor_shape = output_tensor_info.get('shape')
            if tensor_idx in subgraph_output_indices:
                output_shape = tensor_shape
                break

        if output_shape is None and op_tensors['outputs']:
            output_shape = op_tensors['outputs'][0]['shape']
        
        # Ensure shapes are tuples
        if input_shape is not None:
            input_shape = tuple(input_shape)
        if output_shape is not None:
            output_shape = tuple(output_shape)
        
        # Load LiteRT interpreter for extracting permutation tensor
        interpreter = self.load_litert_interpreter(str(tflite_path))
        
        builder = TemplateContextBuilder()
        
        # Convert shapes to CMSIS dims
        input_dims = builder.nhwc_to_cmsis_dims(input_shape)
        output_dims = builder.nhwc_to_cmsis_dims(output_shape)
        
        # Extract permutation from TFLite model
        # For 4D NHWC, default permutation is [0, 2, 1, 3] (swap H and W)
        # Check if there's a permutation tensor in the model
        permutation = [0, 2, 1, 3]  # Default: swap H and W
        tensor_details = interpreter.get_tensor_details()
        for tensor in tensor_details:
            if tensor['name'] and 'perm' in tensor['name'].lower():
                perm_data = interpreter.get_tensor(tensor['index'])
                if perm_data is not None and len(perm_data) == 4:
                    permutation = [int(x) for x in perm_data]
                break
        
        num_dims = len(input_shape)
        
        # Generate input data and quantize
        rng_state = self.rng.__getstate__()
        self.rng = np.random.default_rng(self.seed)
        
        input_data = self.rng.uniform(-1.0, 1.0, size=input_shape).astype(np.float32)
        
        self.rng.__setstate__(rng_state)
        
        # Extract quantization from LiteRT (match the selected input tensor)
        input_quant = op_tensors['inputs'][0]['quantization']
        for input_tensor_info in op_tensors['inputs']:
            tensor_idx = input_tensor_info.get('index', -1)
            if tensor_idx in subgraph_input_indices:
                input_quant = input_tensor_info['quantization']
                break
        input_scale = input_quant.get('scale', 1.0)
        input_zp = input_quant.get('zero_point', 0)
        
        # Handle per-channel quantization (convert to scalar)
        if isinstance(input_scale, (list, np.ndarray)):
            input_scale = float(input_scale[0]) if len(input_scale) > 0 else 1.0
        if isinstance(input_zp, (list, np.ndarray)):
            input_zp = int(input_zp[0]) if len(input_zp) > 0 else 0
        
        input_scale = float(input_scale)
        input_zp = int(input_zp)
        
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
        
        # Run inference using interpreter (LiteRT is read-only, so we still need interpreter for execution)
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        interpreter.set_tensor(input_details[0]['index'], input_q)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        output_data = np.array(output_data)
        
        # Format arrays
        input_array_str = builder.format_array_as_c_literal(input_q)
        expected_output_array_str = builder.format_array_as_c_literal(output_data)
        permutation_array_str = builder.format_array_as_c_literal(np.array(permutation, dtype=np.uint32))
        
        # Build template context
        context = {
            'name': name,
            'prefix': name,
            'input_dims': input_dims,
            'output_dims': output_dims,
            'num_dims': num_dims,
            'permutation_array': permutation_array_str,
            'input_data_array': input_array_str,
            'expected_output_array': expected_output_array_str,
            'input_dtype': kernel_info["input_c_type"],
            'output_dtype': kernel_info["output_c_type"],
            'kernel_fn': kernel_info["kernel_fn"],
        }
        
        # Render templates
        includes_api_dir = output_dir / "includes"
        includes_api_dir.mkdir(parents=True, exist_ok=True)
        
        h_content = self.render_template("transpose/transpose.h.j2", context)
        h_path = includes_api_dir / f"{name}_transpose.h"
        with open(h_path, 'w') as f:
            f.write(h_content)
        
        c_content = self.render_template("transpose/transpose.c.j2", context)
        c_path = output_dir / f"{name}_transpose.c"
        with open(c_path, 'w') as f:
            f.write(c_content)
        
        cmake_context = {
            'name': name,
            'operator': self.desc.get('operator', 'Transpose'),
            'operator_name': 'transpose'
        }
        cmake_content = self.render_template("common/CMakeLists.txt.j2", cmake_context)
        cmake_path = output_dir / "CMakeLists.txt"
        with open(cmake_path, 'w') as f:
            f.write(cmake_content)
        
        print(f"Generated C/H files and CMakeLists.txt for {name}")

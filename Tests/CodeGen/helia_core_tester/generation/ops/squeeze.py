"""
Squeeze operation implementation.
"""

from typing import Dict, Any
import numpy as np
import tensorflow as tf
from pathlib import Path
from helia_core_tester.generation.ops.base import OperationBase


class OpSqueeze(OperationBase):
    """
    Squeeze operation - removes dimensions of size 1.
    """
    
    def build_keras_model(self) -> tf.keras.Model:
        """Build Keras model for Squeeze operation."""
        input_shape = self.desc['input_shape']
        axes = self.desc.get('axes', None)
        
        inputs = tf.keras.Input(shape=input_shape[1:], dtype=tf.float32, name='input')
        
        # Squeeze operation
        # Calculate output shape by removing dimensions of size 1
        output_shape_list = []
        if axes is not None:
            # Squeeze specific axes
            axes_adjusted = [a - 1 if a > 0 else a for a in axes]
            for i, dim in enumerate(input_shape[1:]):
                if i not in axes_adjusted or dim != 1:
                    output_shape_list.append(dim)
        else:
            # Squeeze all dimensions of size 1
            for dim in input_shape[1:]:
                if dim != 1:
                    output_shape_list.append(dim)
        
        output_shape = tuple(output_shape_list) if output_shape_list else (1,)
        
        if axes is not None:
            # Adjust axes to account for batch dimension removal
            axes_adjusted = [a - 1 if a > 0 else a for a in axes]
            x = tf.keras.layers.Lambda(
                lambda x: tf.squeeze(x, axis=axes_adjusted),
                output_shape=output_shape,
                name='squeeze'
            )(inputs)
        else:
            # Squeeze all dimensions of size 1
            x = tf.keras.layers.Lambda(
                lambda x: tf.squeeze(x),
                output_shape=output_shape,
                name='squeeze'
            )(inputs)
        
        model = tf.keras.Model(inputs=inputs, outputs=x)
        return model

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        """Convert Keras model to TFLite with quantization."""
        super().convert_to_tflite(model, out_path, rep_seed)
    
    def _select_cmsis_squeeze_kernel(self) -> Dict[str, str]:
        """
        Select appropriate CMSIS-NN kernel function for Squeeze operation.
        
        Squeeze is similar to Reshape - it's just a memory copy with shape change.
        We use arm_reshape_s8 which is essentially a memcpy.
        
        Returns:
            Dictionary with kernel_fn, input_c_type, output_c_type
        """
        activation_dtype = self.desc.get('activation_dtype', 'S8')
        
        if activation_dtype == 'S8':
            return {
                'kernel_fn': 'arm_reshape_s8',
                'input_c_type': 'int8_t',
                'output_c_type': 'int8_t'
            }
        else:
            raise NotImplementedError(f"Unsupported Squeeze dtype: {activation_dtype} (only S8 supported)")
    
    def generate_c_files(self, output_dir: Path) -> None:
        """
        Generate C and H files from templates for Squeeze operation.
        
        Squeeze removes dimensions of size 1. Since it's just a shape/view change
        and the data layout doesn't change, we can use arm_reshape_s8 (which is
        essentially a memcpy) similar to how Reshape works.
        """
        from pathlib import Path
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder
        
        name = self.desc['name']
        tflite_path = output_dir / f"{name}.tflite"
        if not tflite_path.exists():
            raise FileNotFoundError(f"TFLite file not found: {tflite_path}")
        
        # Select CMSIS kernel + types
        kernel_info = self._select_cmsis_squeeze_kernel()
        
        # Load LiteRT model for shape extraction
        from helia_core_tester.generation.utils.litert_utils import get_operator_tensors_from_litert
        model, subgraph = self.load_litert_model(str(tflite_path))
        op_tensors = get_operator_tensors_from_litert(model, subgraph, 0)
        
        # Extract shapes from LiteRT.
        # Prefer subgraph I/O tensors to avoid selecting a shape/axes tensor.
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
        
        builder = TemplateContextBuilder()
        
        # Convert shapes to CMSIS dims
        input_dims = builder.nhwc_to_cmsis_dims(input_shape)
        output_dims = builder.nhwc_to_cmsis_dims(output_shape)
        
        # Calculate total size (should be same for input and output)
        total_size = int(np.prod(input_shape))
        
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
            'total_size': total_size,
            'input_data_array': input_array_str,
            'expected_output_array': expected_output_array_str,
            'input_dtype': kernel_info["input_c_type"],
            'output_dtype': kernel_info["output_c_type"],
            'kernel_fn': kernel_info["kernel_fn"],
        }
        
        # Render templates
        includes_api_dir = output_dir / "includes"
        includes_api_dir.mkdir(parents=True, exist_ok=True)
        
        h_content = self.render_template("squeeze/squeeze.h.j2", context)
        h_path = includes_api_dir / f"{name}_squeeze.h"
        with open(h_path, 'w') as f:
            f.write(h_content)
        
        c_content = self.render_template("squeeze/squeeze.c.j2", context)
        c_path = output_dir / f"{name}_squeeze.c"
        with open(c_path, 'w') as f:
            f.write(c_content)
        
        cmake_context = {
            'name': name,
            'operator': self.desc.get('operator', 'Squeeze'),
            'operator_name': 'squeeze'
        }
        cmake_content = self.render_template("common/CMakeLists.txt.j2", cmake_context)
        cmake_path = output_dir / "CMakeLists.txt"
        with open(cmake_path, 'w') as f:
            f.write(cmake_content)
        
        print(f"Generated C/H files and CMakeLists.txt for {name}")

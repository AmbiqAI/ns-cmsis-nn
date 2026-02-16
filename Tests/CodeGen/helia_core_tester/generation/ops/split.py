"""
Split operation implementation.
"""

from typing import Dict, Any
import numpy as np
import tensorflow as tf
from pathlib import Path
from helia_core_tester.generation.ops.base import OperationBase


class OpSplit(OperationBase):
    """
    Split operation - splits a tensor into multiple tensors.
    """
    
    def build_keras_model(self) -> tf.keras.Model:
        """Build Keras model for Split operation."""
        input_shape = self.desc['input_shape']
        axis = self.desc.get('axis', -1)
        num_splits = self.desc.get('num_splits')
        size_splits = self.desc.get('size_splits', None)
        
        inputs = tf.keras.Input(shape=input_shape[1:], dtype=tf.float32, name='input')
        
        # Adjust axis to account for batch dimension removal
        if axis >= 0:
            axis_adjusted = axis - 1 if axis > 0 else axis
        else:
            axis_adjusted = axis
        
        # Split operation
        if size_splits is not None:
            # SplitV - split with specified sizes
            x = tf.split(inputs, size_splits, axis=axis_adjusted)
        elif num_splits is not None:
            # Split - split into equal parts
            x = tf.split(inputs, num_splits, axis=axis_adjusted)
        else:
            raise ValueError("Split operation requires either 'num_splits' or 'size_splits'")
        
        # For TFLite, we need to return a single output, so we'll concatenate back
        # In practice, TFLite Split returns multiple outputs, but Keras Model needs single output
        # We'll use the first split as output (this is a limitation we'll need to handle)
        output = x[0] if len(x) > 0 else inputs
        
        model = tf.keras.Model(inputs=inputs, outputs=output)
        return model

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        """Convert Keras model to TFLite with quantization."""
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        activation_dtype = self.desc.get('activation_dtype', 'S8')
        
        if activation_dtype == 'S8':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.int8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
        elif activation_dtype == 'S16':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8
            ]
            converter.inference_input_type = tf.int16
            converter.inference_output_type = tf.int16
        
        def representative_data_gen():
            for _ in range(100):
                if 'input_shape' in self.desc:
                    inputs = self.rng.uniform(-1.0, 1.0, size=self.desc['input_shape']).astype(np.float32)
                    yield [inputs]
        
        converter.representative_dataset = representative_data_gen
        
        tflite_model = converter.convert()
        with open(out_path, 'wb') as f:
            f.write(tflite_model)
    
    def _select_cmsis_split_kernel(self) -> Dict[str, str]:
        """
        Select appropriate CMSIS-NN kernel function for Split operation.
        
        Returns:
            Dictionary with kernel_fn, input_c_type, output_c_type
        """
        activation_dtype = self.desc.get('activation_dtype', 'S8')
        
        if activation_dtype == 'S8':
            return {
                'kernel_fn': 'arm_split_s8',
                'input_c_type': 'int8_t',
                'output_c_type': 'int8_t'
            }
        elif activation_dtype == 'S16':
            return {
                'kernel_fn': 'arm_split_s16',
                'input_c_type': 'int16_t',
                'output_c_type': 'int16_t'
            }
        else:
            raise NotImplementedError(f"Unsupported Split dtype: {activation_dtype}")
    
    def generate_c_files(self, output_dir: Path) -> None:
        """
        Generate C and H files from templates for Split operation.
        """
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder
        
        name = self.desc['name']
        tflite_path = output_dir / f"{name}.tflite"
        if not tflite_path.exists():
            raise FileNotFoundError(f"TFLite file not found: {tflite_path}")
        
        # Select CMSIS kernel + types
        kernel_info = self._select_cmsis_split_kernel()
        
        # Load LiteRT model for shape extraction
        from helia_core_tester.generation.utils.litert_utils import get_operator_tensors_from_litert
        model, subgraph = self.load_litert_model(str(tflite_path))
        op_tensors = get_operator_tensors_from_litert(model, subgraph, 0)
        
        # Extract shapes from LiteRT
        input_shape = op_tensors['inputs'][0]['shape']
        # For now, handle only the first output (Split has multiple outputs)
        output_shape = op_tensors['outputs'][0]['shape'] if op_tensors['outputs'] else None
        
        # Ensure shapes are tuples
        if input_shape is not None:
            input_shape = tuple(input_shape)
        if output_shape is not None:
            output_shape = tuple(output_shape)
        
        builder = TemplateContextBuilder()
        
        # Convert shapes to CMSIS dims
        input_dims = builder.nhwc_to_cmsis_dims(input_shape)
        output_dims = builder.nhwc_to_cmsis_dims(output_shape)
        
        # Extract axis and num_splits/size_splits from descriptor
        axis = self.desc.get('axis', -1)
        num_splits = self.desc.get('num_splits', None)
        size_splits = self.desc.get('size_splits', None)
        
        # Convert axis to 0-based
        if axis < 0:
            axis = len(input_shape) + axis
        
        # Calculate split_dims
        if size_splits is not None:
            split_dims = size_splits
            num_splits = len(size_splits)
        elif num_splits is not None:
            # Equal splits
            axis_size = input_shape[axis]
            split_size = axis_size // num_splits
            split_dims = [split_size] * num_splits
        else:
            # Default: split into 2 equal parts
            num_splits = 2
            axis_size = input_shape[axis]
            split_size = axis_size // num_splits
            split_dims = [split_size] * num_splits
        
        # Generate input data and quantize
        rng_state = self.rng.__getstate__()
        self.rng = np.random.default_rng(self.seed)
        
        input_data = self.rng.uniform(-1.0, 1.0, size=input_shape).astype(np.float32)
        
        self.rng.__setstate__(rng_state)
        
        # Extract quantization
        input_qp = input_details[0].get('quantization_parameters', {})
        input_scale = input_qp.get('scales', [1.0])
        input_zp = input_qp.get('zero_points', [0])
        input_scale = float(input_scale[0] if isinstance(input_scale, list) else input_scale)
        input_zp = int(input_zp[0] if isinstance(input_zp, list) else input_zp)
        
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
        # Get first output (Split has multiple outputs)
        output_data = interpreter.get_tensor(output_details[0]['index'])
        output_data = np.array(output_data)
        
        # Format arrays
        input_array_str = builder.format_array_as_c_literal(input_q)
        expected_output_array_str = builder.format_array_as_c_literal(output_data)
        split_dims_array_str = builder.format_array_as_c_literal(np.array(split_dims, dtype=np.int32))
        input_shape_array_str = builder.format_array_as_c_literal(np.array(input_shape, dtype=np.int32))
        
        # Build template context
        context = {
            'name': name,
            'prefix': name,
            'input_dims': input_dims,
            'output_dims': output_dims,
            'input_dims_count': len(input_shape),
            'axis': axis,
            'num_splits': num_splits,
            'split_dims_array': split_dims_array_str,
            'input_shape_array': input_shape_array_str,
            'input_data_array': input_array_str,
            'expected_output_array': expected_output_array_str,
            'input_dtype': kernel_info["input_c_type"],
            'output_dtype': kernel_info["output_c_type"],
            'kernel_fn': kernel_info["kernel_fn"],
        }
        
        # Render templates
        includes_api_dir = output_dir / "includes"
        includes_api_dir.mkdir(parents=True, exist_ok=True)
        
        h_content = self.render_template("split/split.h.j2", context)
        h_path = includes_api_dir / f"{name}_split.h"
        with open(h_path, 'w') as f:
            f.write(h_content)
        
        c_content = self.render_template("split/split.c.j2", context)
        c_path = output_dir / f"{name}_split.c"
        with open(c_path, 'w') as f:
            f.write(c_content)
        
        cmake_context = {
            'name': name,
            'operator': self.desc.get('operator', 'Split'),
            'operator_name': 'split'
        }
        cmake_content = self.render_template("common/CMakeLists.txt.j2", cmake_context)
        cmake_path = output_dir / "CMakeLists.txt"
        with open(cmake_path, 'w') as f:
            f.write(cmake_content)
        
        print(f"Generated C/H files and CMakeLists.txt for {name}")


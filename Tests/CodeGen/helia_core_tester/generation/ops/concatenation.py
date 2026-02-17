"""
Concatenation operation implementation.
"""

from typing import Dict, Any, List
import numpy as np
import tensorflow as tf
from pathlib import Path
from helia_core_tester.generation.ops.base import OperationBase


class OpConcatenation(OperationBase):
    """
    Concatenation operation - concatenates tensors along an axis.
    """
    
    def build_keras_model(self) -> tf.keras.Model:
        """Build Keras model for Concatenation operation."""
        # Concatenation requires multiple inputs
        input_shapes = []
        is_const_variant = False
        
        if 'input_1_shape' in self.desc:
            # Multiple named inputs
            i = 1
            while f'input_{i}_shape' in self.desc:
                input_shapes.append(self.desc[f'input_{i}_shape'])
                i += 1
        else:
            # Single input shape - check if it's a const variant
            if 'const' in self.desc.get('name', '').lower():
                is_const_variant = True
                # For const variants, we have one variable input and one constant input
                input_shapes = [self.desc.get('input_shape')]
            else:
                # Single input shape (shouldn't happen for concatenation, but handle gracefully)
                input_shapes = [self.desc.get('input_shape')]
        
        axis = self.desc.get('axis', -1)
        
        # Axis in descriptor is NHWC including batch; Keras Concatenate expects the same
        axis_adjusted = axis
        
        # Create input layers
        inputs = []
        const_tensors = []
        
        if is_const_variant:
            # Create one variable input
            var_shape = input_shapes[0]
            var_input = tf.keras.Input(shape=var_shape[1:], dtype=tf.float32, name='input1')
            inputs.append(var_input)
            
            # Create constant input based on variant type
            name = self.desc.get('name', '').lower()
            if 'scalar' in name:
                # Const scalar: create a scalar constant and broadcast it
                # For axis=-1 (channel), const shape should be [1, 1, 1, 1] or similar
                const_value = tf.constant(0.5, dtype=tf.float32)  # Use a non-zero value
                # Broadcast to match concatenation requirements
                # For channel axis (-1), we need shape [1, H, W, 1]
                const_tensor = tf.keras.layers.Lambda(
                    lambda x: tf.broadcast_to(const_value, [tf.shape(x)[0], var_shape[1], var_shape[2], 1]),
                    name='const_scalar'
                )(var_input)
            elif 'vector' in name:
                # Const vector: create a vector constant
                # For axis=2 (width), const shape should be [1, H, 1, C] or similar
                const_value = tf.constant(0.5, dtype=tf.float32)
                # For width axis (2), we need shape [1, H, 1, C]
                const_tensor = tf.keras.layers.Lambda(
                    lambda x: tf.broadcast_to(const_value, [tf.shape(x)[0], var_shape[1], 1, var_shape[3]]),
                    name='const_vector'
                )(var_input)
            else:
                # Default: treat as scalar
                const_value = tf.constant(0.5, dtype=tf.float32)
                const_tensor = tf.keras.layers.Lambda(
                    lambda x: tf.broadcast_to(const_value, [tf.shape(x)[0], var_shape[1], var_shape[2], 1]),
                    name='const'
                )(var_input)
            
            const_tensors.append(const_tensor)
        else:
            # Regular multiple inputs
            for i, shape in enumerate(input_shapes):
                inp = tf.keras.Input(shape=shape[1:], dtype=tf.float32, name=f'input{i+1}')
                inputs.append(inp)
        
        # Combine variable inputs and constant tensors
        all_inputs = inputs + const_tensors
        
        # Concatenation operation
        x = tf.keras.layers.Concatenate(axis=axis_adjusted)(all_inputs)
        
        model = tf.keras.Model(inputs=inputs, outputs=x)
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
                inputs_list = []
                i = 1
                while f'input_{i}_shape' in self.desc:
                    shape = self.desc[f'input_{i}_shape']
                    inputs_list.append(self.rng.uniform(-1.0, 1.0, size=shape).astype(np.float32))
                    i += 1
                if not inputs_list and 'input_shape' in self.desc:
                    # For const variants, only provide the variable input
                    # The constant is embedded in the model
                    inputs_list.append(self.rng.uniform(-1.0, 1.0, size=self.desc['input_shape']).astype(np.float32))
                yield inputs_list
        
        converter.representative_dataset = representative_data_gen
        
        tflite_model = converter.convert()
        with open(out_path, 'wb') as f:
            f.write(tflite_model)
    
    def _select_cmsis_concatenation_kernel(self) -> Dict[str, str]:
        """
        Select appropriate CMSIS-NN kernel function for Concatenation operation.
        
        Returns:
            Dictionary with kernel_fn, input_c_type, output_c_type
        """
        activation_dtype = self.desc.get('activation_dtype', 'S8')
        
        if activation_dtype == 'S8':
            return {
                'kernel_fn': 'arm_concatenation_s8',
                'input_c_type': 'int8_t',
                'output_c_type': 'int8_t'
            }
        elif activation_dtype == 'S16':
            return {
                'kernel_fn': 'arm_concatenation_s16',
                'input_c_type': 'int16_t',
                'output_c_type': 'int16_t'
            }
        else:
            raise NotImplementedError(f"Unsupported Concatenation dtype: {activation_dtype}")
    
    def generate_c_files(self, output_dir: Path) -> None:
        """
        Generate C and H files from templates for Concatenation operation.
        """
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder
        
        name = self.desc['name']
        tflite_path = output_dir / f"{name}.tflite"
        if not tflite_path.exists():
            raise FileNotFoundError(f"TFLite file not found: {tflite_path}")
        
        # Select CMSIS kernel + types
        kernel_info = self._select_cmsis_concatenation_kernel()
        
        # Load LiteRT model for shape and quantization extraction
        from helia_core_tester.generation.utils.litert_utils import get_operator_tensors_from_litert, get_tensor_data_from_litert
        model, subgraph = self.load_litert_model(str(tflite_path))
        op_tensors = get_operator_tensors_from_litert(model, subgraph, 0)
        
        # Extract shapes from LiteRT (multi-input operator)
        op_inputs = list(op_tensors['inputs'])
        num_inputs = len(op_inputs)
        input_shapes = [tuple(tensor['shape']) if tensor['shape'] is not None else None for tensor in op_inputs]
        output_shape = op_tensors['outputs'][0]['shape']
        
        # Ensure output shape is tuple (may be overridden below)
        if output_shape is not None:
            output_shape = tuple(output_shape)
        
        # Check if this is a const variant (need to check with interpreter for this logic)
        interpreter = self.load_litert_interpreter(str(tflite_path))
        input_details = interpreter.get_input_details()
        is_const_variant = 'const' in name.lower() and len(input_details) == 1

        # If const variant and operator inputs include const data, reorder so variable inputs come first
        if is_const_variant and len(op_inputs) > len(input_details):
            variable_inputs = [t for t in op_inputs if t.get('data') is None]
            const_inputs = [t for t in op_inputs if t.get('data') is not None]
            if len(variable_inputs) + len(const_inputs) == len(op_inputs):
                op_inputs = variable_inputs + const_inputs
                input_shapes = [tuple(tensor['shape']) if tensor['shape'] is not None else None for tensor in op_inputs]
                num_inputs = len(op_inputs)
        
        builder = TemplateContextBuilder()
        
        # Extract axis from descriptor (default to -1 for last dimension)
        axis = self.desc.get('axis', -1)
        # Convert negative axis based on actual rank (NHWC includes batch)
        input_rank = len(input_shapes[0]) if input_shapes and input_shapes[0] is not None else (len(output_shape) if output_shape else 0)
        if axis < 0:
            axis = input_rank + axis  # -1 -> last dimension
        
        # For const variants, we need to create a const input shape
        const_shape = None
        if is_const_variant and len(input_shapes) == 1:
            var_shape = input_shapes[0]
            # Calculate const input shape based on axis
            const_shape = list(var_shape)
            if 'scalar' in name.lower():
                # Const scalar: for axis=-1 (channel), shape is [1, H, W, 1]
                if axis == 3:  # Channel axis
                    const_shape = [1, var_shape[1], var_shape[2], 1]
                else:
                    const_shape[axis] = 1
            elif 'vector' in name.lower():
                # Const vector: for axis=2 (width), shape is [1, H, 1, C]
                if axis == 2:  # Width axis
                    const_shape = [1, var_shape[1], 1, var_shape[3]]
                else:
                    const_shape[axis] = 1
            else:
                # Default: same as scalar
                const_shape[axis] = 1
            
            input_shapes.append(tuple(const_shape))
            num_inputs = 2
            const_shape = tuple(const_shape)

        # Recompute output shape from inputs to avoid relying on potentially wrong op output metadata
        if input_shapes and all(shape is not None for shape in input_shapes):
            output_shape = list(input_shapes[0])
            output_shape[axis] = sum(int(shape[axis]) for shape in input_shapes)
            output_shape = tuple(output_shape)

        # Convert shapes to CMSIS dims
        output_dims = builder.nhwc_to_cmsis_dims(output_shape)
        output_rank = len(output_shape)
        
        # Calculate input_concat_dims (dimension along axis for each input)
        input_concat_dims = []
        for input_shape in input_shapes:
            if axis < len(input_shape):
                input_concat_dims.append(int(input_shape[axis]))
            else:
                input_concat_dims.append(1)
        
        # Generate input data and quantize
        rng_state = self.rng.__getstate__()
        self.rng = np.random.default_rng(self.seed)
        
        input_data_list = []
        input_q_list = []
        input_array_strs = []
        
        # Extract quantization (assume all inputs have same quantization)
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
        
        # Generate variable input(s)
        for i, input_shape in enumerate(input_shapes[:len(input_details)]):
            input_data = self.rng.uniform(-1.0, 1.0, size=input_shape).astype(np.float32)
            input_q = np.round(input_data / float(input_scale) + float(input_zp)).astype(np.int32)
            input_q = np.clip(input_q, qmin, qmax).astype(np_in_dtype)
            input_data_list.append(input_data)
            input_q_list.append(input_q)
            input_array_strs.append(builder.format_array_as_c_literal(input_q))
        
        # Generate const input if needed
        const_input_q = None
        const_from_model = False
        if is_const_variant:
            # Prefer using constant tensor data from the model if available
            for tensor in op_inputs:
                if tensor.get('data') is not None:
                    const_input_q = np.array(tensor['data'], copy=True)
                    const_from_model = True
                    break
            # If not found in op inputs, search all tensors for a matching const buffer
            if const_input_q is None and const_shape is not None:
                for tensor in subgraph.tensors:
                    if tensor.shape is None:
                        continue
                    shape = tuple(tensor.shape.tolist())
                    if shape != const_shape:
                        continue
                    data = get_tensor_data_from_litert(tensor, model)
                    if data is not None:
                        const_input_q = np.array(data, copy=True)
                        const_from_model = True
                        break
            if const_input_q is None:
                const_shape = input_shapes[-1]
                # Use a fixed value (0.5 quantized) for const input
                const_value_float = 0.5
                const_value_q = int(np.round(const_value_float / float(input_scale) + float(input_zp)))
                const_value_q = max(qmin, min(qmax, const_value_q))
                const_input_q = np.full(const_shape, const_value_q, dtype=np_in_dtype)
            else:
                const_input_q = const_input_q.astype(np_in_dtype)
            input_q_list.append(const_input_q)
            input_array_strs.append(builder.format_array_as_c_literal(const_input_q))
        
        self.rng.__setstate__(rng_state)
        
        # Run inference using LiteRT interpreter
        interpreter = self.load_litert_interpreter(str(tflite_path))
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # For const variants, TFLite model only has one input (the variable input)
        # The const is embedded in the model, so we only set the variable input
        for i, input_q in enumerate(input_q_list[:len(input_details)]):
            interpreter.set_tensor(input_details[i]['index'], input_q)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        output_data = np.array(output_data)

        # If const variant and we couldn't find a true const buffer, derive const input from output
        if is_const_variant and const_input_q is not None and not const_from_model:
            # If const was synthesized (e.g., 0.5), back-calculate from output by slicing
            # along concat axis using input_concat_dims.
            try:
                split_indices = np.cumsum(input_concat_dims[:-1])
                split_tensors = np.split(output_data, split_indices, axis=axis)
                derived_const = split_tensors[1]
                const_input_q = np.array(derived_const, copy=True).astype(np_in_dtype)
                input_q_list[-1] = const_input_q
                input_array_strs[-1] = builder.format_array_as_c_literal(const_input_q)
            except Exception:
                pass
        
        # Format arrays
        expected_output_array_str = builder.format_array_as_c_literal(output_data)
        input_concat_dims_array_str = builder.format_array_as_c_literal(np.array(input_concat_dims, dtype=np.int32))
        output_shape_array_str = builder.format_array_as_c_literal(np.array(output_shape, dtype=np.int32))
        
        # Build template context
        context = {
            'name': name,
            'prefix': name,
            'output_dims': output_dims,
            'output_rank': output_rank,
            'num_inputs': num_inputs,
            'axis': axis,
            'input_concat_dims_array': input_concat_dims_array_str,
            'output_shape_array': output_shape_array_str,
            'input_data_arrays': input_array_strs,
            'expected_output_array': expected_output_array_str,
            'input_dtype': kernel_info["input_c_type"],
            'output_dtype': kernel_info["output_c_type"],
            'kernel_fn': kernel_info["kernel_fn"],
        }
        
        # Render templates
        includes_api_dir = output_dir / "includes"
        includes_api_dir.mkdir(parents=True, exist_ok=True)
        
        h_content = self.render_template("concatenation/concatenation.h.j2", context)
        h_path = includes_api_dir / f"{name}_concatenation.h"
        with open(h_path, 'w') as f:
            f.write(h_content)
        
        c_content = self.render_template("concatenation/concatenation.c.j2", context)
        c_path = output_dir / f"{name}_concatenation.c"
        with open(c_path, 'w') as f:
            f.write(c_content)
        
        cmake_context = {
            'name': name,
            'operator': self.desc.get('operator', 'Concatenation'),
            'operator_name': 'concatenation'
        }
        cmake_content = self.render_template("common/CMakeLists.txt.j2", cmake_context)
        cmake_path = output_dir / "CMakeLists.txt"
        with open(cmake_path, 'w') as f:
            f.write(cmake_content)
        
        print(f"Generated C/H files and CMakeLists.txt for {name}")

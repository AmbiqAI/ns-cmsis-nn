"""
PReLU operation implementation.
"""

from typing import Dict, Any, Iterable
import numpy as np
import tensorflow as tf
from pathlib import Path
from helia_core_tester.generation.ops.base import OperationBase


class OpPReLU(OperationBase):
    """
    PReLU (Parametric ReLU) operation.
    """
    
    def _prepare_alpha_values(self, input_shape: tuple, alpha_values: Iterable[float] | None = None) -> np.ndarray:
        """Prepare alpha values for PReLU layer."""
        value_shape = input_shape[1:]  # Remove batch dimension
        if not value_shape:
            raise ValueError("Input shape must include at least one non-batch dimension for PReLU.")
        num_values = int(np.prod(value_shape))
        
        if alpha_values is None:
            # Default: linear spacing from 0.05 to 0.25
            data = np.linspace(0.05, 0.25, num=num_values, dtype=np.float32)
        else:
            data = np.asarray(alpha_values, dtype=np.float32)
            # If single scalar, expand to match input shape
            if data.size == 1:
                data = np.full(num_values, data[0], dtype=np.float32)
            elif data.size != num_values:
                raise ValueError(
                    f"alpha_values has {data.size} entries, but expected {num_values} "
                    f"to match input shape {value_shape}."
                )
        return data.reshape(value_shape)
    
    def build_keras_model(self) -> tf.keras.Model:
        """Build Keras model for PReLU operation."""
        input_shape = self.desc['input_shape']
        
        # Build model with float32 inputs (will be quantized later)
        inputs = tf.keras.Input(shape=input_shape[1:], dtype=tf.float32, name='input')
        
        # Get alpha values from descriptor
        # Priority: 1) top-level 'alpha' field, 2) hint.extras.alpha_values, 3) default
        alpha_values = None
        
        # Check for top-level 'alpha' field (scalar value)
        if 'alpha' in self.desc:
            alpha_scalar = self.desc['alpha']
            if isinstance(alpha_scalar, (int, float)):
                # Scalar alpha: use this value for all elements
                # Will be expanded in _prepare_alpha_values
                alpha_values = [float(alpha_scalar)]
            elif isinstance(alpha_scalar, list):
                # List of alpha values
                alpha_values = alpha_scalar
        
        # Fallback to hint.extras.alpha_values if not found at top level
        if alpha_values is None and 'hint' in self.desc:
            extras = self.desc.get('hint', {}).get('extras', {})
            if 'alpha_values' in extras:
                # Flatten the nested list if present
                alpha_list = extras['alpha_values']
                if isinstance(alpha_list, list) and len(alpha_list) > 0:
                    if isinstance(alpha_list[0], list):
                        # Flatten nested list
                        alpha_values = [item for sublist in alpha_list for item in sublist]
                    else:
                        alpha_values = alpha_list
        
        # If alpha_values is None, _prepare_alpha_values will use default
        alpha_array = self._prepare_alpha_values(tuple(input_shape), alpha_values)
        
        # PReLU operation with per-element alpha
        prelu_layer = tf.keras.layers.PReLU(
            alpha_initializer=tf.keras.initializers.Constant(alpha_array),
            shared_axes=None,  # Per-element alpha
            name='prelu'
        )
        output = prelu_layer(inputs)
            
        model = tf.keras.Model(inputs=[inputs], outputs=output)
        return model

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        """Convert Keras model to TFLite with quantization."""
        # Create converter
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Apply quantization based on activation_dtype
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

        
        # Generate representative dataset
        def representative_data_gen():
            rng = np.random.default_rng(rep_seed)
            for _ in range(100):
                if 'input_shape' in self.desc:
                    inputs = rng.uniform(-8.0, 8.0, size=self.desc['input_shape']).astype(np.float32)
                    yield [inputs]
                elif 'input_1_shape' in self.desc and 'input_2_shape' in self.desc:
                    inputs1 = rng.uniform(-8.0, 8.0, size=self.desc['input_1_shape']).astype(np.float32)
                    inputs2 = rng.uniform(-8.0, 8.0, size=self.desc['input_2_shape']).astype(np.float32)
                    yield [inputs1, inputs2]
        
        converter.representative_dataset = representative_data_gen
        
        # Convert and save
        tflite_model = converter.convert()
        with open(out_path, 'wb') as f:
            f.write(tflite_model)
    
    def _select_cmsis_prelu_kernel(self) -> Dict[str, str]:
        """
        Select appropriate CMSIS-NN kernel function for PReLU operation.
        
        Returns:
            Dictionary with kernel_fn, input_c_type, output_c_type
        """
        activation_dtype = self.desc.get('activation_dtype', 'S8')
        
        if activation_dtype == 'S8':
            return {
                'kernel_fn': 'arm_prelu_s8',
                'input_c_type': 'int8_t',
                'output_c_type': 'int8_t'
            }
        else:
            raise NotImplementedError(f"Unsupported PReLU dtype: {activation_dtype} (only S8 supported)")
    
    def generate_c_files(self, output_dir: Path) -> None:
        """
        Generate C and H files from templates for PReLU operation.
        """
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder
        from helia_core_tester.generation.utils.tflite_utils import calculate_multiplier_shift
        from helia_core_tester.generation.utils.litert_utils import (
            load_litert_model, extract_weights_biases_from_litert, get_tensor_data_from_litert
        )
        
        name = self.desc['name']
        tflite_path = output_dir / f"{name}.tflite"
        if not tflite_path.exists():
            raise FileNotFoundError(f"TFLite file not found: {tflite_path}")
        
        # Select CMSIS kernel + types
        kernel_info = self._select_cmsis_prelu_kernel()
        
        # Load LiteRT model for tensor extraction
        from helia_core_tester.generation.utils.litert_utils import get_operator_tensors_from_litert
        
        model, subgraph = load_litert_model(str(tflite_path))
        if len(subgraph.operators) == 0:
            raise ValueError("No operators found in model")
        
        op_tensors = get_operator_tensors_from_litert(model, subgraph, 0)
        
        # Extract shapes from LiteRT
        if not op_tensors['inputs']:
            raise ValueError("No input tensors found")
        if not op_tensors['outputs']:
            raise ValueError("No output tensors found")
        
        input_shape = op_tensors['inputs'][0]['shape']
        output_shape = op_tensors['outputs'][0]['shape']
        
        # Extract quantization from LiteRT
        input_quant = op_tensors['inputs'][0]['quantization']
        output_quant = op_tensors['outputs'][0]['quantization']
        
        input_scale = input_quant.get('scale', 1.0) if isinstance(input_quant, dict) else 1.0
        input_zp = input_quant.get('zero_point', 0) if isinstance(input_quant, dict) else 0
        output_scale = output_quant.get('scale', 1.0) if isinstance(output_quant, dict) else 1.0
        output_zp = output_quant.get('zero_point', 0) if isinstance(output_quant, dict) else 0
        
        # Get first element (per-tensor quantization)
        input_scale = float(input_scale[0] if isinstance(input_scale, (list, np.ndarray)) else input_scale)
        input_zp = int(input_zp[0] if isinstance(input_zp, (list, np.ndarray)) else input_zp)
        output_scale = float(output_scale[0] if isinstance(output_scale, (list, np.ndarray)) else output_scale)
        output_zp = int(output_zp[0] if isinstance(output_zp, (list, np.ndarray)) else output_zp)
        
        # Extract alpha weights using LiteRT
        # For PReLU, alpha is typically the second input (index 1)
        # Alpha can be 1D (vector), 2D, 3D, etc., so we need to check operator inputs directly
        op = subgraph.operators[0]
        alpha_weights = None
        alpha_quant = input_quant  # Default to input quantization
        
        # Check if alpha is in operator inputs (typically at index 1)
        if len(op.inputs) > 1:
            alpha_tensor_idx = op.inputs[1]
            if alpha_tensor_idx >= 0 and alpha_tensor_idx < len(subgraph.tensors):
                alpha_tensor = subgraph.tensors[alpha_tensor_idx]
                alpha_weights = get_tensor_data_from_litert(alpha_tensor, model)
                if alpha_weights is not None:
                    # Get alpha quantization from the tensor
                    from helia_core_tester.generation.utils.litert_utils import get_tensor_quantization_from_litert
                    alpha_quant = get_tensor_quantization_from_litert(alpha_tensor)
        
        # Fallback: try extract_weights_biases_from_litert
        if alpha_weights is None:
            weights_biases = extract_weights_biases_from_litert(model, subgraph, 0)
            alpha_weights = weights_biases.get('weights')
            if alpha_weights is None:
                # Alpha might be 1D and classified as bias by generic extractor
                alpha_weights = weights_biases.get('biases')
        
        if alpha_weights is None:
            raise ValueError("PReLU requires alpha weights but none found in TFLite model")
        
        # Extract alpha quantization parameters (already extracted above)
        if isinstance(alpha_quant, dict):
            alpha_scale = alpha_quant.get('scale', input_scale)
            alpha_zp = alpha_quant.get('zero_point', input_zp)
        else:
            alpha_scale = input_scale
            alpha_zp = input_zp
        
        # Get first element (per-tensor quantization)
        alpha_scale = float(alpha_scale[0] if isinstance(alpha_scale, (list, np.ndarray)) else alpha_scale)
        alpha_zp = int(alpha_zp[0] if isinstance(alpha_zp, (list, np.ndarray)) else alpha_zp)
        
        builder = TemplateContextBuilder()
        
        # Convert shapes to CMSIS dims
        input_dims = builder.nhwc_to_cmsis_dims(input_shape)
        output_dims = builder.nhwc_to_cmsis_dims(output_shape)
        alpha_shape = alpha_weights.shape
        
        # For PReLU, alpha dimensions should match the input's non-singleton dimensions
        if len(alpha_shape) == 1 and len(input_shape) >= 2:
            # Alpha is 1D, input is 2D+: match alpha to input's layout
            # If input was converted as width-based (w=shape[1], c=1), alpha should match
            if input_dims['w'] > 1 and input_dims['c'] == 1:
                # Input uses width dimension, so alpha should too
                alpha_dims = {
                    'n': 1,
                    'h': 1,
                    'w': int(alpha_shape[0]),
                    'c': 1
                }
            elif input_dims['c'] > 1 and input_dims['w'] == 1:
                # Input uses channel dimension, so alpha should too
                alpha_dims = {
                    'n': 1,
                    'h': 1,
                    'w': 1,
                    'c': int(alpha_shape[0])
                }
            else:
                # Default: match input dimensions
                alpha_dims = builder.nhwc_to_cmsis_dims(alpha_shape)
        else:
            # Use standard conversion
            alpha_dims = builder.nhwc_to_cmsis_dims(alpha_shape)
        
        output_multiplier_identity = float(input_scale) / float(output_scale)
        output_multiplier_alpha = (float(alpha_scale) * float(input_scale)) / float(output_scale)
        
        # Calculate multipliers and shifts (equivalent to AirFixedPointScale.from_real_multiplier)
        mult_identity, shift_identity = calculate_multiplier_shift(output_multiplier_identity)
        mult_alpha, shift_alpha = calculate_multiplier_shift(output_multiplier_alpha)
        
        # Quantize alpha weights
        # Check if alpha_weights are already quantized (int8/int16) or float
        if alpha_weights.dtype in [np.int8, np.int16, np.uint8]:
            # Alpha weights are already quantized, use them directly
            alpha_q = alpha_weights.astype(np.int8) if alpha_weights.dtype == np.uint8 else alpha_weights
        else:
            # Alpha weights are float, need to quantize them
            alpha_q = np.round(alpha_weights / float(alpha_scale) + float(alpha_zp)).astype(np.int32)
            alpha_q = np.clip(alpha_q, -128, 127).astype(np.int8)
        
        # Generate input data and quantize
        rng_state = self.rng.__getstate__()
        self.rng = np.random.default_rng(self.seed)
        
        input_data = self.rng.uniform(-1.0, 1.0, size=input_shape).astype(np.float32)
        
        self.rng.__setstate__(rng_state)
        
        # Quantize inputs
        if kernel_info["input_c_type"] == "int8_t":
            np_in_dtype = np.int8
            qmin, qmax = -128, 127
        else:
            raise ValueError(f"Unsupported input_c_type: {kernel_info['input_c_type']}")
        
        input_q = np.round(input_data / float(input_scale) + float(input_zp)).astype(np.int32)
        input_q = np.clip(input_q, qmin, qmax).astype(np_in_dtype)
        
        # Load LiteRT interpreter for inference
        interpreter = self.load_litert_interpreter(str(tflite_path))
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
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
        alpha_array_str = builder.format_array_as_c_literal(alpha_q)
        expected_output_array_str = builder.format_array_as_c_literal(output_data)
        
        # Build template context
        context = {
            'name': name,
            'prefix': name,
            'input_dims': input_dims,
            'alpha_dims': alpha_dims,
            'output_dims': output_dims,
            'input_offset': -int(input_zp),  # Negated for CMSIS-NN
            'alpha_offset': -int(alpha_zp),  # Negated for CMSIS-NN
            'output_offset': int(output_zp),  # Not negated
            'output_mult_alpha': int(mult_alpha),
            'output_shift_alpha': int(shift_alpha),
            'output_mult_identity': int(mult_identity),
            'output_shift_identity': int(shift_identity),
            'input_data_array': input_array_str,
            'alpha_array': alpha_array_str,
            'expected_output_array': expected_output_array_str,
            'input_dtype': kernel_info["input_c_type"],
            'output_dtype': kernel_info["output_c_type"],
            'kernel_fn': kernel_info["kernel_fn"],
        }
        
        # Render templates
        includes_api_dir = output_dir / "includes"
        includes_api_dir.mkdir(parents=True, exist_ok=True)
        
        h_content = self.render_template("prelu/prelu.h.j2", context)
        h_path = includes_api_dir / f"{name}_prelu.h"
        with open(h_path, 'w') as f:
            f.write(h_content)
        
        c_content = self.render_template("prelu/prelu.c.j2", context)
        c_path = output_dir / f"{name}_prelu.c"
        with open(c_path, 'w') as f:
            f.write(c_content)
        
        cmake_context = {
            'name': name,
            'operator': self.desc.get('operator', 'PReLU'),
            'operator_name': 'prelu'
        }
        cmake_content = self.render_template("common/CMakeLists.txt.j2", cmake_context)
        cmake_path = output_dir / "CMakeLists.txt"
        with open(cmake_path, 'w') as f:
            f.write(cmake_content)
        
        print(f"Generated C/H files and CMakeLists.txt for {name}")


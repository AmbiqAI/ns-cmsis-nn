"""
FullyConnected operation implementation with dtype-aware quantization.
"""

from typing import Dict, Any
import numpy as np
import tensorflow as tf
from .base import OperationBase
import keras
from pathlib import Path

class OpFullyConnected(OperationBase):
    """
    FullyConnected operation.
    """
    
    def build_keras_model(self) -> tf.keras.Model:
        """Build Keras model for FullyConnected operation."""
        input_shape = self.desc['input_shape']
        filter_shape = self.desc['filter_shape']
        
        # Handle both 2D [batch, features] and 4D [batch, h, w, c] input shapes
        if len(input_shape) == 2:
            # 2D input: [batch, features]
            input_features = input_shape[1]
            batch_size = input_shape[0]
            needs_flatten = False
        else:
            # 4D input: [batch, h, w, c]
            # Calculate total features: h * w * c
            input_features = input_shape[1] * input_shape[2] * input_shape[3]
            batch_size = input_shape[0]
            needs_flatten = True
        
        # Extract output units from filter_shape
        # filter_shape can be [output_units, input_features] or [output_units, 1, 1, input_features]
        if len(filter_shape) == 2:
            output_units = filter_shape[0]
        else:
            output_units = filter_shape[0]
        
        # Get activation and use_bias from descriptor
        activation_str = self.desc.get('activation', 'NONE')
        use_bias = self.desc.get('use_bias', True)
        
        # Build model with input layer matching descriptor shape
        if needs_flatten:
            # 4D input: [batch, h, w, c]
            inputs = keras.layers.Input(shape=input_shape[1:], batch_size=batch_size, name='input')
            # Flatten to [batch, h*w*c]
            x = keras.layers.Flatten()(inputs)
        else:
            # 2D input: [batch, features]
            inputs = keras.layers.Input(shape=(input_features,), batch_size=batch_size, name='input')
            x = inputs
        
        # Dense layer without activation (we'll apply activation separately if needed)
        x = keras.layers.Dense(output_units, activation=None, use_bias=use_bias, name='dense')(x)
        
        # Apply activation if specified
        if activation_str == 'RELU':
            x = keras.layers.ReLU()(x)
        elif activation_str == 'RELU6':
            x = keras.layers.ReLU(max_value=6)(x)
        elif activation_str == 'TANH':
            x = keras.layers.Activation('tanh')(x)
        elif activation_str == 'SIGMOID':
            x = keras.layers.Activation('sigmoid')(x)
        elif activation_str != 'NONE':
            raise ValueError(f"Unsupported activation: {activation_str}")
        
        model = keras.models.Model(inputs=inputs, outputs=x)
        return model

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        """Convert Keras model to TFLite with quantization."""
        import tensorflow as tf
        import numpy as np
        
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
            converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
            converter.inference_input_type = tf.int16
            converter.inference_output_type = tf.int16
        
        # Generate representative dataset
        def representative_data_gen():
            for _ in range(100):
                if 'input_shape' in self.desc:
                    inputs = self.rng.uniform(-1.0, 1.0, size=self.desc['input_shape']).astype(np.float32)
                    yield [inputs]
                elif 'input_1_shape' in self.desc and 'input_2_shape' in self.desc:
                    inputs1 = self.rng.uniform(-1.0, 1.0, size=self.desc['input_1_shape']).astype(np.float32)
                    inputs2 = self.rng.uniform(-1.0, 1.0, size=self.desc['input_2_shape']).astype(np.float32)
                    yield [inputs1, inputs2]
        
        converter.representative_dataset = representative_data_gen
        
        # Convert and save
        tflite_model = converter.convert()
        with open(out_path, 'wb') as f:
            f.write(tflite_model)
    
    def _select_cmsis_fc_kernel(self) -> Dict[str, str]:
        """
        Select appropriate CMSIS-NN fully connected kernel function.
        
        Returns:
            Dictionary with kernel function name, C types, and buffer size function
        """
        activation_dtype = self.desc.get('activation_dtype', 'S8').upper()
        weight_dtype = self.desc.get('weight_dtype', 'S8').upper()
        
        if activation_dtype == 'S8' and weight_dtype == 'S8':
            return {
                'kernel_fn': 'arm_fully_connected_wrapper_s8',
                'kernel_get_buffer_size_fn': 'arm_fully_connected_s8_get_buffer_size',
                'input_c_type': 'int8_t',
                'output_c_type': 'int8_t',
                'weight_c_type': 'int8_t',
                'bias_c_type': 'int32_t'
            }
        elif activation_dtype == 'S16' and weight_dtype == 'S8':
            return {
                'kernel_fn': 'arm_fully_connected_wrapper_s16',
                'kernel_get_buffer_size_fn': 'arm_fully_connected_s16_get_buffer_size',
                'input_c_type': 'int16_t',
                'output_c_type': 'int16_t',
                'weight_c_type': 'int8_t',
                'bias_c_type': 'int64_t'
            }
        else:
            raise NotImplementedError(f"Unsupported FullyConnected dtype combo: {activation_dtype} x {weight_dtype}")
    
    def generate_c_files(self, output_dir: Path) -> None:
        """
        Generate C and H files from templates for FullyConnected operation.
        """
        from pathlib import Path
        from ..utils.template_context import TemplateContextBuilder
        
        name = self.desc['name']
        tflite_path = output_dir / f"{name}.tflite"
        if not tflite_path.exists():
            raise FileNotFoundError(f"TFLite file not found: {tflite_path}")
        
        # Select CMSIS kernel + types
        kernel_info = self._select_cmsis_fc_kernel()
        
        # Load interpreter
        interpreter = self.load_tflite_interpreter(str(tflite_path))
        
        # Tensor shapes
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        input_shape = tuple(input_details[0]['shape'])
        output_shape = tuple(output_details[0]['shape'])
        
        # Extract quantization parameters from interpreter (more reliable than LiteRT)
        # LiteRT sometimes returns incorrect scales (e.g., 1.0 instead of actual scale)
        input_qp = input_details[0].get('quantization_parameters', {})
        output_qp = output_details[0].get('quantization_parameters', {})
        
        input_quant = {
            'scale': input_qp.get('scales', [1.0]),
            'zero_point': input_qp.get('zero_points', [0]),
            'per_channel': len(input_qp.get('scales', [])) > 1
        }
        output_quant = {
            'scale': output_qp.get('scales', [1.0]),
            'zero_point': output_qp.get('zero_points', [0]),
            'per_channel': len(output_qp.get('scales', [])) > 1
        }
        
        # Extract weights and biases (still use LiteRT for weights, but get weight quantization from LiteRT)
        weights_biases = self.extract_weights_biases(str(tflite_path))
        weights = weights_biases.get('weights')
        biases = weights_biases.get('biases')
        
        # Get weight quantization from LiteRT
        from ..utils.litert_utils import (
            load_litert_model, get_operator_tensors_from_litert,
            get_tensor_data_from_litert, get_tensor_quantization_from_litert,
            get_tensor_shape_from_litert
        )
        model, subgraph = load_litert_model(str(tflite_path))
        op_tensors = get_operator_tensors_from_litert(model, subgraph, 0)
        
        weight_quant = None
        if weights is not None:
            # Search all tensors to find the one matching our weights
            op = subgraph.operators[0]
            input_indices = set(subgraph.inputs)
            output_indices = set(subgraph.outputs)
            
            for tensor_idx, tensor in enumerate(subgraph.tensors):
                # Skip if it's an input or output
                if tensor_idx in input_indices or tensor_idx in output_indices:
                    continue
                
                tensor_data = get_tensor_data_from_litert(tensor, model)
                tensor_shape = get_tensor_shape_from_litert(tensor)
                
                # Check if this is the weight tensor (matches shape and data)
                if (tensor_data is not None and tensor_shape is not None and 
                    len(tensor_shape) > 1 and tensor_data.shape == weights.shape and
                    np.array_equal(tensor_data, weights)):
                    weight_quant = get_tensor_quantization_from_litert(tensor)
                    break
            
            # Fallback: check operator inputs for weight tensor
            if weight_quant is None:
                for input_tensor_info in op_tensors['inputs']:
                    if input_tensor_info['data'] is not None and len(input_tensor_info['shape']) > 1:
                        weight_quant = input_tensor_info.get('quantization')
                        break
        
        # For weight_sum computation, if weight_quant is None, use output_quant
        # (weights typically share output scale/ZP in TFLite when not explicitly quantized)
        weight_quant_for_params = weight_quant if weight_quant is not None else output_quant
        
        quant_params = {
            'input': input_quant,
            'output': output_quant,
            'weight': weight_quant_for_params or {'scale': 1.0, 'zero_point': 0, 'per_channel': False}
        }
        
        # Weight tensor for TFLite FullyConnected is [output_units, input_features]
        # CMSIS expects filter_dims: n=input_features, c=output_units, h=1, w=1
        if weights is not None:
            filter_shape = tuple(weights.shape)
            # Handle 1D weights (shouldn't happen, but handle gracefully)
            if len(filter_shape) == 1:
                # If we got a 1D array, it might be incorrectly extracted
                # Try to infer from descriptor or output shape
                if len(output_shape) == 2:
                    output_units = output_shape[1]
                    input_features = filter_shape[0] // output_units if filter_shape[0] % output_units == 0 else filter_shape[0]
                    if filter_shape[0] == output_units * input_features:
                        # Reshape to 2D
                        weights = weights.reshape(output_units, input_features)
                        filter_shape = tuple(weights.shape)
                    else:
                        raise ValueError(f"Cannot infer 2D shape from 1D weights shape {filter_shape} and output shape {output_shape}")
                else:
                    raise ValueError(f"Unsupported filter shape: {filter_shape} (1D) and output shape: {output_shape}")
            
            if len(filter_shape) == 2:
                # TFLite format: [output_units, input_features]
                filter_dims = {
                    'n': int(filter_shape[1]),  # input_features (col_dim)
                    'h': 1,
                    'w': 1,
                    'c': int(filter_shape[0])   # output_units (row_dim)
                }
            else:
                raise ValueError(f"Unsupported filter shape: {filter_shape}")
            # Ensure weights are int8 in generated C
            if weights.dtype != np.int8:
                weights = weights.astype(np.int8)
        else:
            # Fallback: descriptor format [output_units, input_features]
            fs = tuple(self.desc['filter_shape'])
            if len(fs) == 2:
                filter_dims = {
                    'n': int(fs[1]),  # input_features
                    'h': 1,
                    'w': 1,
                    'c': int(fs[0])   # output_units
                }
            else:
                raise ValueError(f"Unsupported filter_shape in descriptor: {fs}")
        
        builder = TemplateContextBuilder()
        # For fully connected, input is flattened: [batch, features] or [batch, h, w, c]
        # CMSIS expects input_dims: n=batch, h=1, w=1, c=features
        if len(input_shape) == 2:
            input_dims = {
                'n': int(input_shape[0]),
                'h': 1,
                'w': 1,
                'c': int(input_shape[1])
            }
        elif len(input_shape) == 4:
            # Flatten: [batch, h, w, c] -> features = h * w * c
            input_dims = {
                'n': int(input_shape[0]),
                'h': 1,
                'w': 1,
                'c': int(input_shape[1] * input_shape[2] * input_shape[3])
            }
        else:
            input_dims = builder.nhwc_to_cmsis_dims(input_shape)
        
        # Output is [batch, output_units]
        if len(output_shape) == 2:
            output_dims = {
                'n': int(output_shape[0]),
                'h': 1,
                'w': 1,
                'c': int(output_shape[1])
            }
        else:
            output_dims = builder.nhwc_to_cmsis_dims(output_shape)
        
        # Build FC parameters
        fc_params = builder.build_fc_params(
            self.desc,
            quant_params['input'],
            quant_params.get('weights', quant_params.get('weight', quant_params['output'])),
            quant_params['output']
        )
        
        # Build quantization parameters
        # CRITICAL: The effective scale for multiplier/shift is NOT output_scale directly!
        # It should be: effective_scale = (input_scale * weight_scale) / output_scale
        input_quant = quant_params['input']
        output_quant = quant_params['output']
        weight_quant = quant_params.get('weights', quant_params.get('weight', output_quant))
        
        input_scale = input_quant.get('scale', 1.0)
        if isinstance(input_scale, (list, np.ndarray)):
            input_scale = float(input_scale[0])
        else:
            input_scale = float(input_scale)
        
        output_scale = output_quant.get('scale', 1.0)
        if isinstance(output_scale, (list, np.ndarray)):
            output_scale = float(output_scale[0])
        else:
            output_scale = float(output_scale)
        
        weight_scale = weight_quant.get('scale', 1.0)
        per_channel = bool(weight_quant.get('per_channel', False))
        
        # Calculate effective scales: (input_scale * weight_scale) / output_scale
        # For S16 per-channel, we need to reduce multipliers from Q31 to Q15 format
        from ..utils.tflite_utils import reduce_multiplier_q31_to_q15, calculate_per_channel_multiplier_shift
        
        activation_dtype = self.desc.get('activation_dtype', 'S8')
        is_s16 = (activation_dtype == 'S16' or kernel_info["input_c_type"] == "int16_t")
        
        if per_channel and isinstance(weight_scale, np.ndarray):
            # Per-channel: effective_scale[i] = (input_scale * weight_scale[i]) / output_scale
            effective_scales = (input_scale * weight_scale) / output_scale
            
            multipliers, shifts = calculate_per_channel_multiplier_shift(
                effective_scales, 
                reduce_to_q15=False  # Kernel does the reduction internally
            )
            
            # Create quant_params dict with pre-calculated multipliers/shifts
            quant_params_dict = {
                'multiplier': multipliers,
                'shift': shifts,
                'multiplier_array': ', '.join(map(str, multipliers)),
                'shift_array': ', '.join(map(str, shifts)),
                'per_channel': True
            }
        else:
            # Per-tensor: effective_scale = (input_scale * weight_scale) / output_scale
            if isinstance(weight_scale, (list, np.ndarray)):
                weight_scale = float(weight_scale[0])
            else:
                weight_scale = float(weight_scale)
            effective_scale = (input_scale * weight_scale) / output_scale
            # Ensure effective_scale is a Python float, not numpy scalar
            effective_scale = float(effective_scale)
            effective_quant = {
                'scale': effective_scale,
                'zero_point': output_quant.get('zero_point', 0),
                'per_channel': False
            }
            quant_params_dict = builder.build_quant_params(effective_quant, per_channel=False)
            
            # For S16 per-tensor, also reduce multiplier from Q31 to Q15
            if is_s16:
                mult = quant_params_dict.get('multiplier', 0)
                quant_params_dict['multiplier'] = reduce_multiplier_q31_to_q15(mult)
        
        quant_params_dict['per_channel'] = per_channel
        
        # Compute weight_sum for S8 fully connected
        weight_sum_array_str = ""
        has_weight_sum = False
        if kernel_info["input_c_type"] == "int8_t" and weights is not None:
            # Import vector_sum_s8 from dwconv (same function)
            from .dwconv import vector_sum_s8
            
            # For fully connected, weights are [output_units, input_features]
            # vector_rows = output_units, vector_cols = input_features
            weights_data = weights
            if len(weights_data.shape) == 2:
                vector_rows = weights_data.shape[0]  # output_units
                vector_cols = weights_data.shape[1]  # input_features
                kernel_matrix = weights_data.reshape(vector_rows, vector_cols)
            else:
                raise ValueError(f"Unsupported fully connected filter layout for weight sum: {weights_data.shape}")
            
            # Get input and weight zero points
            input_zp = quant_params['input'].get('zero_point', 0)
            if isinstance(input_zp, (list, np.ndarray)):
                input_zp = int(input_zp[0])
            else:
                input_zp = int(input_zp)
            
            # For weight_sum computation, use the actual weight tensor's zero point
            # If weight_quant was found, use it; otherwise use output_quant
            # For per-channel weights, use the first zero point 
            weight_quant_for_sum = weight_quant if weight_quant is not None else output_quant
            weight_zp = weight_quant_for_sum.get('zero_point', 0)
            if isinstance(weight_zp, (list, np.ndarray)):
                weight_zp = int(weight_zp[0])  
            else:
                weight_zp = int(weight_zp)
            
            # Compute weight_sum using vector_sum_s8    
            bias_data_for_sum = None
            if biases is not None and biases.size > 0:
                if biases.dtype != np.int32:
                    bias_data_for_sum = biases.astype(np.int32)
                else:
                    bias_data_for_sum = biases
            
            weight_sum = vector_sum_s8(
                vector_data=kernel_matrix,
                vector_cols=vector_cols,
                vector_rows=vector_rows,
                lhs_offset=-input_zp,  # input_offset = -input_zero_point
                rhs_offset=-weight_zp,  # weight_offset = -weight_zero_point
                bias_data=bias_data_for_sum,
            ).astype(np.int32)
            
            weight_sum_array_str = builder.format_array_as_c_literal(weight_sum)
            has_weight_sum = True
            
            # If weight_sum is precomputed, biases are effectively "consumed" into it
            # So we set has_biases to False for the kernel call 
            if bias_data_for_sum is not None:
                has_biases = False
                biases = None
        
        # Format weights and biases as C arrays
        weights_array_str = builder.format_array_as_c_literal(weights)
        if not has_weight_sum:  # Only include biases if weight_sum is not precomputed
            has_biases = biases is not None and biases.size > 0
            if has_biases:
                # Convert biases to appropriate type for CMSIS-NN
                if kernel_info["bias_c_type"] == "int64_t":
                    if biases.dtype != np.int64:
                        biases = biases.astype(np.int64)
                else:  # int32_t
                    if biases.dtype != np.int32:
                        biases = biases.astype(np.int32)
                biases_array_str = builder.format_array_as_c_literal(biases)
            else:
                biases_array_str = ""
        else:
            # Weight_sum includes bias, so no separate bias array
            has_biases = False
            biases_array_str = ""
        
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
        output_data = self.run_inference(interpreter, input_q)
        
        # Format input and output arrays
        input_data_array_str = builder.format_array_as_c_literal(input_q)
        # Output dtype should match the kernel's output type
        if kernel_info["output_c_type"] == "int16_t":
            expected_output_array_str = builder.format_array_as_c_literal(output_data.astype(np.int16))
        else:
            expected_output_array_str = builder.format_array_as_c_literal(output_data.astype(np.int8))
        
        # Calculate buffer size max
        # For per-channel S16, we need to use per_channel buffer size function
        # For per-tensor S16, buffer size is 0
        # For S8, buffer size depends on weight_sum or regular buffer
        activation_dtype = self.desc.get('activation_dtype', 'S8')
        is_s16 = (activation_dtype == 'S16' or kernel_info["input_c_type"] == "int16_t")
        
        if is_s16 and quant_params_dict.get('per_channel', False):
            # Per-channel S16 requires output_ch * sizeof(int32_t) buffer
            buffer_size_max = output_dims['c'] * 4  # sizeof(int32_t) = 4
        else:
            buffer_size_max = builder.calculate_fc_buffer_size_max(
                filter_dims,
                output_dtype=activation_dtype
            )
        
        # Weight_sum is precomputed and embedded in header
        # No runtime buffer needed since it's a constant array
        
        # Build template context
        context = {
            'name': name,
            'prefix': name,
            'input_dims': input_dims,
            'filter_dims': filter_dims,
            'output_dims': output_dims,
            'fc_params': fc_params,
            'quant_params': quant_params_dict,
            'weights_array': weights_array_str,
            'biases_array': biases_array_str,
            'has_biases': has_biases,
            'input_data_array': input_data_array_str,
            'expected_output_array': expected_output_array_str,
            'input_dtype': kernel_info["input_c_type"],
            'output_dtype': kernel_info["output_c_type"],
            'bias_dtype': kernel_info["bias_c_type"],
            'kernel_fn': kernel_info["kernel_fn"],
            'kernel_get_buffer_size_fn': kernel_info["kernel_get_buffer_size_fn"],
            'buffer_size_max': buffer_size_max,
            'weight_sum_array': weight_sum_array_str,
            'has_weight_sum': has_weight_sum,
        }
        
        # Render templates
        includes_api_dir = output_dir / "includes"
        includes_api_dir.mkdir(parents=True, exist_ok=True)
        
        h_content = self.render_template("fully_connected/fully_connected.h.j2", context)
        h_path = includes_api_dir / f"{name}_fully_connected.h"
        with open(h_path, 'w') as f:
            f.write(h_content)
        
        c_content = self.render_template("fully_connected/fully_connected.c.j2", context)
        c_path = output_dir / f"{name}_fully_connected.c"
        with open(c_path, 'w') as f:
            f.write(c_content)
        
        cmake_context = {
            'name': name,
            'operator': self.desc.get('operator', 'FullyConnected'),
            'operator_name': 'fully_connected'
        }
        cmake_content = self.render_template("common/CMakeLists.txt.j2", cmake_context)
        cmake_path = output_dir / "CMakeLists.txt"
        with open(cmake_path, 'w') as f:
            f.write(cmake_content)
        
        print(f"Generated C/H files and CMakeLists.txt for {name}")
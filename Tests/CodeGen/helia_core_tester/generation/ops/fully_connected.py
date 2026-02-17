"""
FullyConnected operation implementation with dtype-aware quantization.
"""

from typing import Dict, Any, Optional
import numpy as np
from helia_core_tester.generation.ops.base import OperationBase
import keras
from pathlib import Path


class OpFullyConnected(OperationBase):
    """
    FullyConnected operation.
    """
    
    def build_keras_model(self):
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
    
    def _get_zero_point(self, quant_dict: Dict[str, Any]) -> int:
        """Get zero point from quantization dictionary."""
        if quant_dict is None:
            return 0
        zp = quant_dict.get('zero_point', 0)
        if isinstance(zp, (list, np.ndarray)):
            return int(zp[0]) if len(zp) > 0 else 0
        return int(zp)
    
    def _compute_activation_range(self, output_quant: Dict[str, Any], output_dtype: np.dtype) -> tuple[int, int]:
        """Compute activation min/max based on output quantization and dtype."""
        activation_str = self.desc.get('activation', 'NONE')
        output_zp = self._get_zero_point(output_quant)
        output_scale = output_quant.get('scale', 1.0)
        if isinstance(output_scale, (list, np.ndarray)):
            output_scale = float(output_scale[0])
        
        if output_dtype == np.int16:
            default_min, default_max = -32768, 32767
        else:  # int8
            default_min, default_max = -128, 127
        
        if activation_str == 'RELU':
            activation_min = max(0, default_min)
            activation_max = default_max
        elif activation_str == 'RELU6':
            # RELU6: clamp to [0, 6] in float, then quantize
            relu6_max_float = 6.0
            relu6_max_quantized = int(np.round(relu6_max_float / output_scale + output_zp))
            activation_min = max(0, default_min)
            activation_max = min(relu6_max_quantized, default_max)
        else:  # NONE, TANH, SIGMOID, etc.
            activation_min = default_min
            activation_max = default_max
        
        # Override with descriptor values if present
        if 'activation_min' in self.desc:
            activation_min = int(self.desc['activation_min'])
        if 'activation_max' in self.desc:
            activation_max = int(self.desc['activation_max'])
        
        return activation_min, activation_max
    
    def _compute_weight_sum_size(self, weights: Optional[np.ndarray], output_dtype: np.dtype) -> int:
        """Compute the size of the weight sum tensor."""
        if weights is None:
            return 0
        if output_dtype == np.int8 and self._supports_weight_sum():
            # Weight sum size = output_units (number of rows in weight matrix)
            return weights.shape[0] if len(weights.shape) == 2 else 0
        return 0
    
    def _supports_weight_sum(self) -> bool:
        """Check if platform supports weight sum optimization."""
        # For CMSIS-NN, weight sum is supported for S8 fully connected
        # This is a simplified check - in a real implementation, this would check platform capabilities
        return True
    
    def _should_precompute_weight_sum(self, weights: Optional[np.ndarray], output_dtype: np.dtype) -> bool:
        """Determine if weight sum should be precomputed."""
        return (
            output_dtype == np.int8
            and self._supports_weight_sum()
            and weights is not None
            and weights.size > 0
        )
    
    def _compute_fixed_point_multipliers(
        self,
        input_scale: float,
        weight_scales: np.ndarray,
        output_scale: float
    ) -> list[Dict[str, Any]]:
        """
        Compute fixed-point multipliers and shifts for each output channel.
        
        Returns:
            List of dictionaries with 'multiplier' and 'shift' keys
        """
        from helia_core_tester.generation.utils.tflite_utils import calculate_per_channel_multiplier_shift
        
        # Compute effective scales: (input_scale * weight_scale) / output_scale
        if isinstance(weight_scales, np.ndarray):
            effective_scales = (input_scale * weight_scales) / output_scale
        else:
            effective_scales = np.array([(input_scale * weight_scales) / output_scale])
        
        multipliers, shifts = calculate_per_channel_multiplier_shift(
            effective_scales,
            reduce_to_q15=False  # Kernel handles reduction internally
        )
        
        return [
            {'multiplier': int(m), 'shift': int(s)}
            for m, s in zip(multipliers, shifts)
        ]

    def _find_fc_operator_index(self, model, subgraph, weights: Optional[np.ndarray]) -> int:
        """
        Find the operator index corresponding to the FullyConnected op.
        Prefer an operator whose inputs include the exact weights tensor.
        """
        from helia_core_tester.generation.utils.litert_utils import get_operator_tensors_from_litert

        if weights is not None:
            for op_index in range(len(subgraph.operators)):
                op_tensors = get_operator_tensors_from_litert(model, subgraph, op_index)
                for tensor in op_tensors.get('inputs', []):
                    data = tensor.get('data')
                    if data is not None and data.shape == weights.shape and np.array_equal(data, weights):
                        return op_index

        # Fallback: pick the first operator that looks like FC (weights as second input)
        for op_index in range(len(subgraph.operators)):
            op_tensors = get_operator_tensors_from_litert(model, subgraph, op_index)
            if len(op_tensors.get('inputs', [])) >= 2:
                weight_candidate = op_tensors['inputs'][1].get('data')
                if weight_candidate is not None and weight_candidate.ndim >= 2 and weight_candidate.size > 4:
                    return op_index

        return 0
    
    def generate_c_files(self, output_dir: Path) -> None:
        """
        Generate C and H files from templates for FullyConnected operation.
        """
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder
        
        name = self.desc['name']
        tflite_path = output_dir / f"{name}.tflite"
        if not tflite_path.exists():
            raise FileNotFoundError(f"TFLite file not found: {tflite_path}")
        
        # Select CMSIS kernel + types
        kernel_info = self._select_cmsis_fc_kernel()
        
        # Load LiteRT model for shape and quantization extraction
        from helia_core_tester.generation.utils.litert_utils import get_operator_tensors_from_litert
        model, subgraph = self.load_litert_model(str(tflite_path))
        
        # Extract weights and biases (initial pass)
        weights_biases = self.extract_weights_biases(str(tflite_path))
        weights = weights_biases.get('weights')
        biases = weights_biases.get('biases')

        # Find the FullyConnected operator index and fetch its tensors
        fc_op_index = self._find_fc_operator_index(model, subgraph, weights)
        op_tensors = get_operator_tensors_from_litert(model, subgraph, fc_op_index)

        # Extract shapes from LiteRT (use FC op tensors)
        input_shape = op_tensors['inputs'][0]['shape']
        output_shape = op_tensors['outputs'][0]['shape']

        # Prefer weights/biases directly from the FC op inputs when available
        if len(op_tensors.get('inputs', [])) > 1 and op_tensors['inputs'][1].get('data') is not None:
            weights = op_tensors['inputs'][1]['data']
        if len(op_tensors.get('inputs', [])) > 2 and op_tensors['inputs'][2].get('data') is not None:
            biases = op_tensors['inputs'][2]['data']

        # Ensure shapes are tuples
        if input_shape is not None:
            input_shape = tuple(input_shape)
        if output_shape is not None:
            output_shape = tuple(output_shape)
        
        # Extract quantization parameters from LiteRT (use FC op tensors)
        input_quant_litert = op_tensors['inputs'][0]['quantization']
        output_quant_litert = op_tensors['outputs'][0]['quantization']
        
        input_quant = {
            'scale': input_quant_litert.get('scale', 1.0),
            'zero_point': input_quant_litert.get('zero_point', 0),
            'per_channel': input_quant_litert.get('per_channel', False)
        }
        output_quant = {
            'scale': output_quant_litert.get('scale', 1.0),
            'zero_point': output_quant_litert.get('zero_point', 0),
            'per_channel': output_quant_litert.get('per_channel', False)
        }
        
        # Get weight quantization from LiteRT
        from helia_core_tester.generation.utils.litert_utils import (
            get_tensor_data_from_litert, get_tensor_quantization_from_litert,
            get_tensor_shape_from_litert
        )
        
        weight_quant = None
        if weights is not None:
            # Prefer weight quantization from FC op input[1] when available
            if len(op_tensors.get('inputs', [])) > 1 and op_tensors['inputs'][1].get('data') is not None:
                weight_quant = op_tensors['inputs'][1].get('quantization')

        if weight_quant is None and weights is not None:
            # Search all tensors to find the one matching our weights
            input_indices = set(subgraph.inputs)
            output_indices = set(subgraph.outputs)
            
            for tensor_idx, tensor in enumerate(subgraph.tensors):
                if tensor_idx in input_indices or tensor_idx in output_indices:
                    continue
                
                tensor_data = get_tensor_data_from_litert(tensor, model)
                tensor_shape = get_tensor_shape_from_litert(tensor)
                
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
        
        # Prepare weight quantization dict
        if weight_quant is None:
            weight_quant = output_quant  # Fallback to output quantization
        
        weight_quant_dict = {
            'scale': weight_quant.get('scale', 1.0),
            'zero_point': weight_quant.get('zero_point', 0),
            'per_channel': weight_quant.get('per_channel', False)
        }
        
        # Validate weights shape
        if weights is not None:
            filter_shape = tuple(weights.shape)
            if len(filter_shape) == 1:
                # Try to infer 2D shape
                if len(output_shape) == 2:
                    output_units = output_shape[1]
                    input_features = filter_shape[0] // output_units if filter_shape[0] % output_units == 0 else filter_shape[0]
                    if filter_shape[0] == output_units * input_features:
                        weights = weights.reshape(output_units, input_features)
                        filter_shape = tuple(weights.shape)
                    else:
                        raise ValueError(f"Cannot infer 2D shape from 1D weights shape {filter_shape}")
                else:
                    raise ValueError(f"Unsupported filter shape: {filter_shape} (1D)")
            
            if len(filter_shape) != 2:
                raise ValueError(f"Unsupported filter shape: {filter_shape}")
            
            # Ensure weights are int8
            if weights.dtype != np.int8:
                weights = weights.astype(np.int8)
            
            # TFLite format: [output_units, input_features]
            filter_dims = {
                'n': int(filter_shape[1]),  # input_features (col_dim)
                'h': 1,
                'w': 1,
                'c': int(filter_shape[0])   # output_units (row_dim)
            }
        else:
            # Fallback: descriptor format
            fs = tuple(self.desc['filter_shape'])
            if len(fs) != 2:
                raise ValueError(f"Unsupported filter_shape in descriptor: {fs}")
            filter_dims = {
                'n': int(fs[1]),  # input_features
                'h': 1,
                'w': 1,
                'c': int(fs[0])   # output_units
            }

        # Fix up biases: ensure we have a 1D bias matching output units
        output_units = int(filter_dims['c'])
        if biases is not None:
            if not hasattr(biases, "shape") or biases.ndim != 1 or biases.shape[0] != output_units:
                biases = None

        if biases is None:
            # Try to read bias directly from FC op input[2]
            op = subgraph.operators[fc_op_index]
            if len(op.inputs) > 2:
                bias_tensor_idx = op.inputs[2]
                if 0 <= bias_tensor_idx < len(subgraph.tensors):
                    bias_tensor = subgraph.tensors[bias_tensor_idx]
                    bias_data = get_tensor_data_from_litert(bias_tensor, model)
                    if bias_data is not None and bias_data.ndim == 1 and bias_data.shape[0] == output_units:
                        biases = bias_data

        if biases is None:
            # Fallback: search all tensors for a matching 1D bias vector
            input_indices = set(subgraph.inputs)
            output_indices = set(subgraph.outputs)
            for tensor_idx, tensor in enumerate(subgraph.tensors):
                if tensor_idx in input_indices or tensor_idx in output_indices:
                    continue
                bias_data = get_tensor_data_from_litert(tensor, model)
                tensor_shape = get_tensor_shape_from_litert(tensor)
                if bias_data is not None and tensor_shape is not None:
                    if len(tensor_shape) == 1 and tensor_shape[0] == output_units:
                        biases = bias_data
                        break
        
        builder = TemplateContextBuilder()
        
        # Prefer descriptor batch size when model shapes report 1 for dynamic batch
        desc_input_shape = self.desc.get('input_shape', None)
        desc_batch = None
        if isinstance(desc_input_shape, (list, tuple)) and len(desc_input_shape) > 0:
            desc_batch = int(desc_input_shape[0])

        batch_size = int(input_shape[0]) if input_shape is not None and len(input_shape) > 0 else None
        if desc_batch is not None and desc_batch > 1 and batch_size != desc_batch:
            batch_size = desc_batch

        # Compute input dimensions
        if len(input_shape) == 2:
            input_dims = {
                'n': int(batch_size if batch_size is not None else input_shape[0]),
                'h': 1,
                'w': 1,
                'c': int(input_shape[1])
            }
        elif len(input_shape) == 4:
            # Flatten: [batch, h, w, c] -> features = h * w * c
            input_dims = {
                'n': int(batch_size if batch_size is not None else input_shape[0]),
                'h': 1,
                'w': 1,
                'c': int(input_shape[1] * input_shape[2] * input_shape[3])
            }
        else:
            input_dims = builder.nhwc_to_cmsis_dims(input_shape)
        
        # Compute output dimensions - use weights shape to get correct output_units
        if weights is not None and len(weights.shape) == 2:
            correct_output_units = int(weights.shape[0])
            if batch_size is None:
                batch_size = int(output_shape[0]) if len(output_shape) >= 1 else int(input_shape[0])
            
            output_dims = {
                'n': batch_size,
                'h': 1,
                'w': 1,
                'c': correct_output_units
            }
            if len(output_shape) == 2 and output_shape[1] != correct_output_units:
                print(f"Warning: LiteRT output_shape[1] ({output_shape[1]}) != weights.shape[0] ({correct_output_units}). Using weights shape.")
        elif len(output_shape) == 2:
            output_dims = {
                'n': int(output_shape[0]),
                'h': 1,
                'w': 1,
                'c': int(output_shape[1])
            }
        else:
            output_dims = builder.nhwc_to_cmsis_dims(output_shape)
        
        # Get scales as arrays for per-channel computation
        input_scale = input_quant['scale']
        if isinstance(input_scale, (list, np.ndarray)):
            input_scale = float(input_scale[0])
        else:
            input_scale = float(input_scale)
        
        weight_scale = weight_quant_dict['scale']
        output_scale = output_quant['scale']
        per_channel = bool(weight_quant_dict.get('per_channel', False))
        
        # Convert scales to numpy arrays for per-channel computation
        if per_channel and isinstance(weight_scale, (list, np.ndarray)):
            weight_scales = np.array(weight_scale, dtype=np.float64)
        else:
            if isinstance(weight_scale, (list, np.ndarray)):
                weight_scales = np.array([float(weight_scale[0])], dtype=np.float64)
            else:
                weight_scales = np.array([float(weight_scale)], dtype=np.float64)
        
        if isinstance(output_scale, (list, np.ndarray)):
            output_scale = float(output_scale[0])
        else:
            output_scale = float(output_scale)
        
        # Compute fixed-point multipliers and shifts
        outputs_fp = self._compute_fixed_point_multipliers(
            input_scale,
            weight_scales,
            output_scale
        )
        
        # Build quantization parameters dict
        if per_channel and len(outputs_fp) > 1:
            multipliers = [fp['multiplier'] for fp in outputs_fp]
            shifts = [fp['shift'] for fp in outputs_fp]
            quant_params_dict = {
                'multiplier': multipliers,
                'shift': shifts,
                'multiplier_array': ', '.join(map(str, multipliers)),
                'shift_array': ', '.join(map(str, shifts)),
                'per_channel': True
            }
        else:
            # Per-tensor
            if len(outputs_fp) > 0:
                multiplier = outputs_fp[0]['multiplier']
                shift = outputs_fp[0]['shift']
            else:
                # Fallback calculation
                effective_scale = (input_scale * float(weight_scales[0])) / output_scale
                from helia_core_tester.generation.utils.tflite_utils import calculate_per_channel_multiplier_shift
                mults, shfts = calculate_per_channel_multiplier_shift(
                    np.array([effective_scale]),
                    reduce_to_q15=False
                )
                multiplier = int(mults[0])
                shift = int(shfts[0])
                
            quant_params_dict = {
                'multiplier': multiplier,
                'shift': shift,
                'per_channel': False
            }
        
        # Compute activation range
        output_dtype = np.int16 if kernel_info["output_c_type"] == "int16_t" else np.int8
        activation_min, activation_max = self._compute_activation_range(output_quant, output_dtype)
        
        # Build FC parameters
        input_zp = self._get_zero_point(input_quant)
        weight_zp = self._get_zero_point(weight_quant_dict)
        output_zp = self._get_zero_point(output_quant)
        
        activation_dtype = self.desc.get('activation_dtype', 'S8')
        if activation_dtype == 'S16':
            # S16 uses symmetric quantization (zero points are 0)
            fc_params = {
                'input_offset': 0,
                'filter_offset': 0,
                'output_offset': 0,
                'activation_min': activation_min,
                'activation_max': activation_max,
            }
        else:
            # S8 uses zero points as offsets
            fc_params = {
                'input_offset': int(-input_zp),
                'filter_offset': int(-weight_zp),
                'output_offset': int(output_zp),
                'activation_min': activation_min,
                'activation_max': activation_max,
            }
        
        # Compute weight sum if needed
        weight_sum = None
        has_weight_sum = False
        if self._should_precompute_weight_sum(weights, output_dtype):
            from .dwconv import vector_sum_s8
            
            vector_rows = weights.shape[0]  # output_units
            vector_cols = weights.shape[1]  # input_features
            
            lhs_offset = -input_zp
            rhs_offset = -weight_zp
            
            bias_data = None
            if biases is not None and biases.size > 0:
                if biases.dtype != np.int32:
                    bias_data = biases.astype(np.int32)
                else:
                    bias_data = biases
            
            weight_sum = vector_sum_s8(
                vector_data=weights,
                vector_cols=vector_cols,
                vector_rows=vector_rows,
                lhs_offset=lhs_offset,
                rhs_offset=rhs_offset,
                bias_data=bias_data,
            ).astype(np.int32)
            
            has_weight_sum = True
            
            # If weight_sum is precomputed, biases are consumed into it
            if bias_data is not None:
                biases = None
        
        # Format arrays
        weights_array_str = builder.format_array_as_c_literal(weights) if weights is not None else ""
        
        has_biases = False
        biases_array_str = ""
        if not has_weight_sum:
            has_biases = biases is not None and biases.size > 0
            if has_biases:
                # Convert biases to appropriate type
                if kernel_info["bias_c_type"] == "int64_t":
                    if biases.dtype != np.int64:
                        biases = biases.astype(np.int64)
                else:  # int32_t
                    if biases.dtype != np.int32:
                        biases = biases.astype(np.int32)
                biases_array_str = builder.format_array_as_c_literal(biases)
        
        weight_sum_array_str = builder.format_array_as_c_literal(weight_sum) if weight_sum is not None else ""
        
        # Generate input data and run inference
        rng_state = self.rng.__getstate__()
        self.rng = np.random.default_rng(self.seed)
        input_data = self.generate_input_data()
        self.rng.__setstate__(rng_state)
        
        # Quantize input - keep original shape for inference (model has Flatten layer)
        input_q = np.round(input_data / input_scale + input_zp).astype(np.int32)
        if kernel_info["input_c_type"] == "int8_t":
            input_q = np.clip(input_q, -128, 127).astype(np.int8)
        else:  # int16_t
            input_q = np.clip(input_q, -32768, 32767).astype(np.int16)
        
        # Run inference using pure LiteRT (no TFLite dependency)
        # Pass input in original shape - model's Flatten layer will handle flattening
        from helia_core_tester.generation.utils.litert_utils import run_inference_litert
        output_data = run_inference_litert(str(tflite_path), input_q, subgraph_index=0)
        
        # Format arrays - format_array_as_c_literal automatically flattens
        input_data_array_str = builder.format_array_as_c_literal(input_q.flatten())
        # Ensure output_data is properly shaped and flattened for C array
        if kernel_info["output_c_type"] == "int16_t":
            expected_output_array_str = builder.format_array_as_c_literal(output_data.flatten().astype(np.int16))
        else:
            expected_output_array_str = builder.format_array_as_c_literal(output_data.flatten().astype(np.int8))
        
        # Calculate buffer size max
        activation_dtype = self.desc.get('activation_dtype', 'S8')
        is_s16 = (activation_dtype == 'S16' or kernel_info["input_c_type"] == "int16_t")
        
        if is_s16 and quant_params_dict.get('per_channel', False):
            buffer_size_max = output_dims['c'] * 4  # sizeof(int32_t) = 4
        else:
            buffer_size_max = builder.calculate_fc_buffer_size_max(
                filter_dims,
                output_dtype=activation_dtype
            )
        
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

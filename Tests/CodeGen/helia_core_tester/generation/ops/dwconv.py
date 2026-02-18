"""
DepthwiseConv2D operation implementation.
"""

from typing import Dict, Optional
from pathlib import Path
import numpy as np
import tensorflow as tf
from helia_core_tester.generation.ops.base import OperationBase


def vector_sum_s8(
    vector_data: np.ndarray,
    vector_cols: int,
    vector_rows: int,
    lhs_offset: int,
    rhs_offset: int,
    bias_data: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Pure-Python port of CMSIS-NN's arm_vector_sum_s8 helper.

    Args:
        vector_data: Flattened (rows * cols) int8 buffer containing the kernel matrix.
        vector_cols: Number of columns per output channel (accumulation depth).
        vector_rows: Number of rows/output channels.
        lhs_offset: Input offset applied during accumulation.
        rhs_offset: Filter/weight offset applied during accumulation.
        bias_data: Optional per-output bias (length == vector_rows), int32.
    
    Returns:
        np.ndarray: int32 array with the kernel sums, matching CMSIS-NN behaviour.
    """
    kernel = np.asarray(vector_data, dtype=np.int8).reshape(vector_rows, vector_cols)
    
    if bias_data is not None:
        vector_sum_buf = np.asarray(bias_data, dtype=np.int32).copy()
    else:
        vector_sum_buf = np.zeros(vector_rows, dtype=np.int32)
    
    if lhs_offset != 0:
        sums = kernel.astype(np.int32).sum(axis=1)
        if rhs_offset != 0:
            sums = sums + vector_cols * rhs_offset
        vector_sum_buf += sums * lhs_offset
    
    return vector_sum_buf


class OpDepthwiseConv2D(OperationBase):
    """
    DepthwiseConv2D operation.
    """
    
    def build_keras_model(self) -> tf.keras.Model:
        """Build Keras model for DepthwiseConv2D operation."""
        input_shape = self.desc['input_shape']
        filter_shape = self.desc['filter_shape']
        
        # Build model with float32 inputs (will be quantized later)
        inputs = tf.keras.Input(shape=input_shape[1:], dtype=tf.float32, name='input')
        
        # Normalize padding (match reference implementation)
        padding = self.desc.get('padding', 'valid')
        if padding is not None:
            padding = str(padding).lower()
        else:
            padding = 'valid'
        
        dwconv_kwargs = {
            'kernel_size': filter_shape[0:2],
            'strides': self.desc.get('strides', [1, 1]),
            'padding': padding,
            'depth_multiplier': self.desc.get('depth_multiplier', 1),
            'use_bias': self.desc.get('use_bias', True),
            'name': 'depthwise_conv2d'
        }
        
        if 'dilation' in self.desc:
            dilation = self.desc['dilation']
            if isinstance(dilation, (int, float)):
                dilation = [int(dilation), int(dilation)]
            elif isinstance(dilation, (list, tuple)):
                if len(dilation) != 2:
                    raise ValueError(f"Invalid dilation: {dilation}. Must be 2 integers or a single integer")
                dilation = [int(dilation[0]), int(dilation[1])]
            else:
                raise ValueError(f"Invalid dilation type: {type(dilation)}. Must be int or list/tuple of 2 ints")
            
            if any(d <= 0 for d in dilation):
                raise ValueError(f"Invalid dilation values: {dilation}. Must be positive integers")
            
            dwconv_kwargs['dilation_rate'] = tuple(dilation)
        
        if dwconv_kwargs['use_bias']:
            dwconv_kwargs['bias_initializer'] = tf.keras.initializers.RandomUniform(minval=-1.0, maxval=1.0)
        
        dwconv = tf.keras.layers.DepthwiseConv2D(**dwconv_kwargs)
        x = dwconv(inputs)
        
        activation = self.desc.get('activation', 'NONE')
        if activation == 'RELU':
            x = tf.keras.layers.ReLU()(x)
        elif activation == 'RELU6':
            x = tf.keras.layers.ReLU(max_value=6)(x)
        elif activation == 'TANH':
            x = tf.keras.layers.Activation('tanh')(x)
        elif activation == 'SIGMOID':
            x = tf.keras.layers.Activation('sigmoid')(x)
        elif activation != 'NONE':
            raise ValueError(f"Unsupported activation: {activation}")
            
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
    
    def _select_cmsis_depthwise_conv_kernel(self) -> Dict[str, str]:
        """
        Pick CMSIS-NN wrapper + C types from descriptor dtypes.
        
        DepthwiseConv2D support:
        - S8 activations, S8 weights  -> arm_depthwise_conv_wrapper_s8, int32 bias
        - S16 activations, S8 weights -> arm_depthwise_conv_wrapper_s16, int64 bias
        """
        act = str(self.desc.get("activation_dtype", "S8")).upper()
        w = str(self.desc.get("weight_dtype", "S8")).upper()
        
        if act == "S8" and w == "S8":
            return {
                "kernel_fn": "arm_depthwise_conv_wrapper_s8",
                "kernel_get_buffer_size_fn": "arm_depthwise_conv_wrapper_s8_get_buffer_size",
                "input_c_type": "int8_t",
                "output_c_type": "int8_t",
                "bias_c_type": "int32_t",
            }
        
        if act == "S16" and w == "S8":
            return {
                "kernel_fn": "arm_depthwise_conv_wrapper_s16",
                "kernel_get_buffer_size_fn": "arm_depthwise_conv_wrapper_s16_get_buffer_size",
                "input_c_type": "int16_t",
                "output_c_type": "int16_t",
                "bias_c_type": "int64_t",
            }
        
        raise NotImplementedError(f"Unsupported DepthwiseConv2D dtype combo: {act} x {w}")
    
    def generate_c_files(self, output_dir: Path) -> None:
        """
        Generate C and H files from templates for DepthwiseConv2D operation.
        """
        from pathlib import Path
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder
        
        name = self.desc['name']
        
        
        tflite_path = output_dir / f"{name}.tflite"
        if not tflite_path.exists():
            raise FileNotFoundError(f"TFLite file not found: {tflite_path}")
        
        # Select CMSIS kernel + types
        kernel_info = self._select_cmsis_depthwise_conv_kernel()
        
        # Load LiteRT model for tensor extraction
        from helia_core_tester.generation.utils.litert_utils import (
            load_litert_model,
            get_operator_tensors_from_litert,
            get_tensor_shape_from_litert,
            get_tensor_quantization_from_litert,
        )
        
        model, subgraph = load_litert_model(str(tflite_path))
        
        # Pick the DepthwiseConv2D operator index.
        # Dilation is commonly lowered to SpaceToBatchND -> DepthwiseConv2D -> BatchToSpaceND,
        # so the first operator is not necessarily DepthwiseConv2D.
        if len(subgraph.operators) == 0:
            raise ValueError("No operators found in model")

        dw_op_index = 0
        bts_op_index = None
        try:
            from ai_edge_litert import schema_py_generated as litert
            for i, op in enumerate(subgraph.operators):
                opcode = model.operatorCodes[op.opcodeIndex]
                if opcode.builtinCode == litert.BuiltinOperator.DEPTHWISE_CONV_2D:
                    dw_op_index = i
                if opcode.builtinCode == litert.BuiltinOperator.BATCH_TO_SPACE_ND:
                    bts_op_index = i
        except Exception:
            # Fallback to first op if we cannot inspect opcodes.
            dw_op_index = 0
            bts_op_index = None

        op_tensors = get_operator_tensors_from_litert(model, subgraph, dw_op_index)
        
        # Extract shapes from LiteRT.
        # Prefer subgraph I/O shapes (model input/output). If missing, fall back to op tensors.
        input_tensor = None
        output_tensor = None
        input_shape = None
        output_shape = None

        if subgraph.inputs and subgraph.outputs:
            input_tensor = subgraph.tensors[subgraph.inputs[0]]
            output_tensor = subgraph.tensors[subgraph.outputs[0]]
            input_shape = get_tensor_shape_from_litert(input_tensor)
            output_shape = get_tensor_shape_from_litert(output_tensor)
        else:
            # Fallback: subgraph I/O not populated. Use first op input and last op output.
            # This preserves model I/O even when the schema omits subgraph inputs/outputs.
            first_op = subgraph.operators[0]
            last_op = subgraph.operators[-1]
            if first_op.inputs is not None and len(first_op.inputs) > 0:
                input_tensor = subgraph.tensors[int(first_op.inputs[0])]
                input_shape = get_tensor_shape_from_litert(input_tensor)
            if last_op.outputs is not None and len(last_op.outputs) > 0:
                output_tensor = subgraph.tensors[int(last_op.outputs[0])]
                output_shape = get_tensor_shape_from_litert(output_tensor)

        if input_shape is None or output_shape is None:
            if not op_tensors['inputs'] or not op_tensors['outputs']:
                raise ValueError("No subgraph inputs/outputs found and op tensors are missing")
            input_shape = op_tensors['inputs'][0]['shape']
            output_shape = op_tensors['outputs'][0]['shape']

        # If BatchToSpaceND exists, use its output tensor for expected output.
        # This matches the dilated depthwise output (SpaceToBatch -> Depthwise -> BatchToSpace),
        # which CMSIS should match.
        expected_tensor = None
        if bts_op_index is not None:
            bts_outs = subgraph.operators[bts_op_index].outputs
            if bts_outs is not None and len(bts_outs) > 0:
                expected_tensor = subgraph.tensors[int(bts_outs[0])]
                expected_shape = get_tensor_shape_from_litert(expected_tensor)
                if expected_shape is not None:
                    output_shape = expected_shape
        
        # Ensure shapes are tuples
        if input_shape is not None:
            input_shape = tuple(input_shape)
        if output_shape is not None:
            output_shape = tuple(output_shape)
        
        # Ensure shapes are 4D (NHWC)
        if len(input_shape) < 4:
            input_shape = (1,) + input_shape if len(input_shape) == 3 else input_shape
        if len(output_shape) < 4:
            output_shape = (1,) + output_shape if len(output_shape) == 3 else output_shape
        
        depth_multiplier = self.desc.get('depth_multiplier', 1)
        input_channels = input_shape[3] if len(input_shape) > 3 else 1
        expected_output_channels = input_channels * depth_multiplier
        if len(output_shape) > 3 and output_shape[3] != expected_output_channels:
            # Fix output_shape to use correct output channels
            output_shape = tuple(list(output_shape[:3]) + [expected_output_channels])
        
        # Extract quantization parameters.
        # Use subgraph I/O quantization to match model I/O when available, else fall back to op tensors.
        if expected_tensor is not None:
            input_quant = get_tensor_quantization_from_litert(input_tensor) if input_tensor is not None else None
            output_quant = get_tensor_quantization_from_litert(expected_tensor)
        elif input_tensor is not None and output_tensor is not None:
            input_quant = get_tensor_quantization_from_litert(input_tensor)
            output_quant = get_tensor_quantization_from_litert(output_tensor)
        else:
            input_quant = op_tensors['inputs'][0]['quantization'] if op_tensors['inputs'] else None
            output_quant = op_tensors['outputs'][0]['quantization'] if op_tensors['outputs'] else None
        
        # Find weight quantization (from weight tensor in inputs)
        weight_quant = None
        for input_tensor_info in op_tensors['inputs']:
            if input_tensor_info['data'] is not None and len(input_tensor_info['shape']) > 1:
                weight_quant = input_tensor_info['quantization']
                break
        
        quant_params = {
            'input': input_quant or {'scale': 1.0, 'zero_point': 0, 'per_channel': False},
            'output': output_quant or {'scale': 1.0, 'zero_point': 0, 'per_channel': False},
            'weight': weight_quant or input_quant or {'scale': 1.0, 'zero_point': 0, 'per_channel': False}
        }
        
        # Extract weights and biases from LiteRT
        weights = op_tensors['weights']
        biases = op_tensors['biases']
        
        # Calculate expected output_channels to validate bias size
        depth_multiplier = self.desc.get('depth_multiplier', 1)
        input_channels = input_shape[3] if len(input_shape) > 3 else 1
        expected_output_channels = input_channels * depth_multiplier
        
        # Validate and fix biases if needed
        if biases is not None:
            bias_shape = biases.shape if hasattr(biases, 'shape') else None
            if bias_shape is not None:
                if len(bias_shape) > 1:
                    print(f"Warning: Biases have wrong shape {bias_shape}, searching for correct 1D bias tensor...")
                    biases = None  # Reset to search for correct one
                # If bias is 1D but wrong size, search for correct one
                elif len(bias_shape) == 1 and bias_shape[0] != expected_output_channels:
                    print(f"Warning: Biases have wrong size {bias_shape[0]} (expected {expected_output_channels}), searching for correct bias tensor...")
                    biases = None  # Reset to search for correct one
        
        # If biases are still wrong or None, search all tensors for correct bias
        if biases is None or (hasattr(biases, 'shape') and len(biases.shape) == 1 and biases.shape[0] != expected_output_channels):
            from helia_core_tester.generation.utils.litert_utils import get_tensor_data_from_litert, get_tensor_shape_from_litert
            op = subgraph.operators[dw_op_index]
            input_indices = set(subgraph.inputs)
            output_indices = set(subgraph.outputs)
            
            # Search all tensors for a 1D tensor matching expected_output_channels
            for tensor_idx, tensor in enumerate(subgraph.tensors):
                if tensor_idx in input_indices or tensor_idx in output_indices:
                    continue
                if tensor_idx in op.inputs and tensor_idx != op.inputs[0]:  # Skip input tensor
                    continue
                
                tensor_data = get_tensor_data_from_litert(tensor, model)
                tensor_shape = get_tensor_shape_from_litert(tensor)
                
                if tensor_data is not None and tensor_shape is not None:
                    # Look for 1D tensor with correct size (bias tensor)
                    if len(tensor_shape) == 1 and tensor_shape[0] == expected_output_channels:
                        biases = tensor_data
                        print(f"Found correct bias tensor: shape={tensor_shape}, size={tensor_data.size}")
                        break
        
        
        # Weight tensor for TFLite DepthwiseConv2D can be in different formats:
        # - TFLite format: [1, H, W, C_OUT] where C_OUT = input_channels * depth_multiplier
        # - Descriptor format: [H, W, I, M] where I=input_channels, M=depth_multiplier
        # CMSIS expects filter_dims: n=depth_multiplier, h=H, w=W, c=output_channels
        # Note: filter_dims.c must be output_channels (not input_channels) because CMSIS-NN
        # uses it as the first dimension in the transposed filter: {filter_dims->c, h, w, n}
        if weights is not None:
            filter_shape = tuple(weights.shape)
            # Ensure weights are int8 in generated C
            if weights.dtype != np.int8:
                weights = weights.astype(np.int8)
        else:
            # Fallback: descriptor is [H, W, I, M]
            fs = tuple(self.desc['filter_shape'])
            filter_shape = fs
        
        builder = TemplateContextBuilder()
        input_dims = builder.nhwc_to_cmsis_dims(input_shape)
        output_dims = builder.nhwc_to_cmsis_dims(output_shape)
        
        # For depthwise conv, CMSIS expects:
        # filter_dims.n = depth_multiplier (usually 1)
        # filter_dims.h = kernel_h
        # filter_dims.w = kernel_w
        # filter_dims.c = output_channels 
        input_channels = input_shape[3]
        
        # For depthwise conv, output_channels = input_channels * depth_multiplier
        # Use descriptor depth_multiplier to calculate correct output_channels
        # (LiteRT may return incorrect output shape for dilated depthwise conv)
        depth_multiplier = self.desc.get('depth_multiplier', 1)
        output_channels = input_channels * depth_multiplier
        
        # Validate against LiteRT output shape if available (for debugging)
        if output_shape[3] != output_channels:
            print(f"Warning: LiteRT output_channels ({output_shape[3]}) != calculated ({output_channels}). Using calculated value.")
        
        
        if len(filter_shape) == 4:
            # Check if TFLite format [1, H, W, C_OUT] or descriptor format [H, W, I, M]
            is_tflite_format = filter_shape[0] == 1
            if is_tflite_format:
                # TFLite format: [1, H, W, C_OUT]
                # Extract H and W from indices 1 and 2
                # Note: filter_dims.n should be 1 (not depth_multiplier)
                # The depth_multiplier is stored in dw_conv_params.ch_mult
                filter_dims = {
                    'n': 1,  # Always 1 for depthwise conv 
                    'h': int(filter_shape[1]),   # kernel_h (from index 1)
                    'w': int(filter_shape[2]),   # kernel_w (from index 2)
                    'c': int(output_channels),   # output_channels (C_OUT from index 3)
                }
            else:
                # Descriptor format: [H, W, I, M]
                # Note: filter_dims.n should be 1 (not depth_multiplier) - 
                # The depth_multiplier is stored in dw_conv_params.ch_mult
                filter_dims = {
                    'n': 1,  # Always 1 for depthwise conv
                    'h': int(filter_shape[0]),   # kernel_h
                    'w': int(filter_shape[1]),   # kernel_w
                    'c': int(output_channels),   # output_channels 
                }
        elif len(filter_shape) == 3:
            # Format: [H, W, I] (depth_multiplier=1)
            filter_dims = {
                'n': 1,  # Always 1 for depthwise conv 
                'h': int(filter_shape[0]),
                'w': int(filter_shape[1]),
                'c': int(output_channels),  # output_channels
            }
        elif len(filter_shape) < 3:
            # Fallback: use descriptor filter_shape if available
            if 'filter_shape' in self.desc:
                desc_filter = self.desc['filter_shape']
                if isinstance(desc_filter, (list, tuple)) and len(desc_filter) >= 2:
                    filter_dims = {
                        'n': 1,
                        'h': int(desc_filter[0]),
                        'w': int(desc_filter[1]),
                        'c': int(output_channels),
                    }
                else:
                    raise ValueError(f"Unsupported filter shape from descriptor: {desc_filter}")
            else:
                raise ValueError(f"Unsupported filter shape: {filter_shape} (and no descriptor filter_shape available)")
        else:
            raise ValueError(f"Unsupported filter shape: {filter_shape}")
        
        
        # Correct kernel size for padding math
        kernel_hw = (filter_dims['h'], filter_dims['w'])
        
        # Build depthwise convolution parameters
        dw_conv_params = builder.build_dw_conv_params(
            self.desc,
            input_shape,
            kernel_hw,
            output_shape,
            quant_params['input'],
            quant_params['output']
        )
        
        # Build quantization parameters (same as Conv2D)
        input_quant = quant_params['input']
        output_quant = quant_params['output']
        weight_quant = quant_params.get('weight', output_quant)
        
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
        if per_channel and isinstance(weight_scale, np.ndarray):
            # Per-channel: effective_scale[i] = (input_scale * weight_scale[i]) / output_scale
            effective_scales = (input_scale * weight_scale) / output_scale
            effective_quant = {
                'scale': effective_scales,
                'zero_point': output_quant.get('zero_point', 0),
                'per_channel': True
            }
            quant_params_dict = builder.build_quant_params(effective_quant, per_channel=True)
            quant_params_dict['effective_scales'] = effective_scales
            from helia_core_tester.generation.utils.tflite_utils import calculate_per_channel_multiplier_shift
            multipliers_raw, shifts_raw = calculate_per_channel_multiplier_shift(effective_scales)
            quant_params_dict['multiplier_array_raw'] = multipliers_raw
            quant_params_dict['shift_array_raw'] = shifts_raw
        else:
            # Per-tensor: effective_scale = (input_scale * weight_scale) / output_scale
            if isinstance(weight_scale, (list, np.ndarray)):
                weight_scale = float(weight_scale[0])
            else:
                weight_scale = float(weight_scale)
            effective_scale = (input_scale * weight_scale) / output_scale
            effective_scale = float(effective_scale)
            effective_quant = {
                'scale': effective_scale,
                'zero_point': output_quant.get('zero_point', 0),
                'per_channel': False
            }
            quant_params_dict = builder.build_quant_params(effective_quant, per_channel=False)
            quant_params_dict['effective_scale'] = effective_scale
            from helia_core_tester.generation.utils.tflite_utils import calculate_multiplier_shift
            multiplier_raw, shift_raw = calculate_multiplier_shift(effective_scale)
            quant_params_dict['multiplier_raw'] = multiplier_raw
            quant_params_dict['shift_raw'] = shift_raw
        
        quant_params_dict['per_channel'] = per_channel
        
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
        
        if kernel_info["input_c_type"] == "int8_t":
            qmin, qmax = -128, 127
            np_in_dtype = np.int8
        elif kernel_info["input_c_type"] == "int16_t":
            qmin, qmax = -32768, 32767
            np_in_dtype = np.int16
        else:
            raise ValueError(f"Unsupported input_c_type: {kernel_info['input_c_type']}")
        
        # Regenerate input data with a tighter range to avoid heavy saturation.
        # Keep deterministic by using the same seed-backed RNG.
        margin = 16  # keep away from int8 edges
        low = (qmin + margin - int(input_zp)) * float(input_scale)
        high = (qmax - margin - int(input_zp)) * float(input_scale)
        if low >= high:
            low = (qmin - int(input_zp)) * float(input_scale)
            high = (qmax - int(input_zp)) * float(input_scale)
        input_data = self.rng.uniform(low, high, size=input_shape).astype(np.float32)

        input_q = np.round(input_data / float(input_scale) + float(input_zp)).astype(np.int32)
        input_q = np.clip(input_q, qmin, qmax).astype(np_in_dtype)
        
        # Run inference (dtype must match interpreter input)
        # Use LiteRT interpreter for inference. For lowered graphs, fetch the
        # DepthwiseConv2D op output tensor directly so we compare against
        # the depthwise op (CMSIS) instead of later ops (e.g., ADD).
        from helia_core_tester.generation.utils.litert_utils import run_inference_litert_tensor
        if bts_op_index is not None:
            bts_outs = subgraph.operators[bts_op_index].outputs
            if bts_outs is not None and len(bts_outs) > 0:
                out_tensor_idx = int(bts_outs[0])
            else:
                out_tensor_idx = int(subgraph.operators[dw_op_index].outputs[0])
        else:
            out_tensor_idx = int(subgraph.operators[dw_op_index].outputs[0])
        output_data = run_inference_litert_tensor(str(tflite_path), input_q, out_tensor_idx)
        
        # Bias handling
        # Detect bias dtype from the LiteRT tensor and use it for CMSIS-NN.
        has_biases = biases is not None and getattr(biases, "size", 0) > 0
        bias_dtype = kernel_info["bias_c_type"]
        if has_biases:
            if biases.dtype == np.int32:
                bias_dtype = "int32_t"
            elif biases.dtype == np.int64:
                bias_dtype = "int64_t"
            else:
                if bias_dtype == "int64_t":
                    biases = biases.astype(np.int64)
                else:
                    biases = biases.astype(np.int32)

            if bias_dtype == "int64_t" and biases.dtype != np.int64:
                biases = biases.astype(np.int64)
            elif bias_dtype == "int32_t" and biases.dtype != np.int32:
                biases = biases.astype(np.int32)
        
        # Compute weight_sum for S8 depthwise convolutions 
        # Weight sum is only needed for S8 optimized kernels (arm_depthwise_conv_s8_opt, arm_depthwise_conv_wrapper_s8)
        weight_sum_array_str = ""
        has_weight_sum = False
        if kernel_info["input_c_type"] == "int8_t" and weights is not None:
            # Reshape weights: if 4D [1, H, W, C_OUT], transpose to [C_OUT, H, W, 1] then reshape to [C_OUT, H*W*1]
            # If 3D [H, W, I], transpose to [I, H, W] then reshape to [I, H*W]
            weights_data = weights
            vector_rows = output_channels
            
            if len(weights_data.shape) == 4:
                # TFLite format: [1, H, W, C_OUT] -> transpose to [C_OUT, H, W, 1] -> reshape to [C_OUT, H*W*1]
                kernel_matrix = weights_data.transpose(3, 0, 1, 2).reshape(vector_rows, -1)
            elif len(weights_data.shape) == 3:
                # Descriptor format: [H, W, I] -> transpose to [I, H, W] -> reshape to [I, H*W]
                kernel_matrix = weights_data.transpose(2, 0, 1).reshape(vector_rows, -1)
            else:
                raise ValueError(f"Unsupported depthwise filter layout for weight sum: {weights_data.shape}")
            
            # Compute weight_sum using vector_sum_s8
            input_zp = quant_params['input'].get('zero_point', 0)
            if isinstance(input_zp, (list, np.ndarray)):
                input_zp = int(input_zp[0])
            else:
                input_zp = int(input_zp)
            
            bias_data_for_sum = None
            if has_biases and biases is not None:
                # Convert bias to int32 if needed
                if biases.dtype != np.int32:
                    bias_data_for_sum = biases.astype(np.int32)
                else:
                    bias_data_for_sum = biases
            
            weight_sum = vector_sum_s8(
                vector_data=kernel_matrix,
                vector_cols=kernel_matrix.shape[1],
                vector_rows=vector_rows,
                lhs_offset=-input_zp,  # input_offset = -input_zero_point
                rhs_offset=0,  # weight offset is 0 for depthwise conv
                bias_data=bias_data_for_sum,
            ).astype(np.int32)
            
            weight_sum_array_str = builder.format_array_as_c_literal(weight_sum)
            has_weight_sum = True
        
        # Format arrays
        weights_array_str = builder.format_array_as_c_literal(weights) if weights is not None else ""
        biases_array_str = builder.format_array_as_c_literal(biases) if has_biases else ""
        input_data_array_str = builder.format_array_as_c_literal(input_q)
        expected_output_array_str = builder.format_array_as_c_literal(output_data)
        
        
        # Calculate buffer size max (conservative estimate for depthwise convolution)
        activation_dtype = self.desc.get('activation_dtype', 'S8')
        buffer_size_max = builder.calculate_depthwise_buffer_size_max(
            input_dims, filter_dims, output_dims,
            output_dtype=activation_dtype
        )
        
        
        # Build template context
        context = {
            'name': name,
            'prefix': name,
            'input_dims': input_dims,
            'filter_dims': filter_dims,
            'output_dims': output_dims,
            'dw_conv_params': dw_conv_params,
            'quant_params': quant_params_dict,
            'weights_array': weights_array_str,
            'biases_array': biases_array_str,
            'has_biases': has_biases,
            'weight_sum_array': weight_sum_array_str,
            'has_weight_sum': has_weight_sum,
            'input_data_array': input_data_array_str,
            'expected_output_array': expected_output_array_str,
            'input_dtype': kernel_info["input_c_type"],
            'output_dtype': kernel_info["output_c_type"],
            'bias_dtype': bias_dtype,
            'kernel_fn': kernel_info["kernel_fn"],
            'kernel_get_buffer_size_fn': kernel_info["kernel_get_buffer_size_fn"],
            'buffer_size_max': buffer_size_max,
        }
        
        # Render templates
        includes_api_dir = output_dir / "includes"
        includes_api_dir.mkdir(parents=True, exist_ok=True)
        
        h_content = self.render_template("depthwise_conv2d/depthwise_conv2d.h.j2", context)
        h_path = includes_api_dir / f"{name}_depthwise_conv2d.h"
        with open(h_path, 'w') as f:
            f.write(h_content)
        
        c_content = self.render_template("depthwise_conv2d/depthwise_conv2d.c.j2", context)
        c_path = output_dir / f"{name}_depthwise_conv2d.c"
        with open(c_path, 'w') as f:
            f.write(c_content)
        
        cmake_context = {
            'name': name,
            'operator': self.desc.get('operator', 'DepthwiseConv2D'),
            'operator_name': 'depthwise_conv2d'
        }
        cmake_content = self.render_template("common/CMakeLists.txt.j2", cmake_context)
        cmake_path = output_dir / "CMakeLists.txt"
        with open(cmake_path, 'w') as f:
            f.write(cmake_content)
        
        print(f"Generated C/H files and CMakeLists.txt for {name}")

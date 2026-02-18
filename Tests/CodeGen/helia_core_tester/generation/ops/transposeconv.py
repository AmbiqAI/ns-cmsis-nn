"""
TransposeConv operation implementation.
"""

from typing import Dict, Any
import numpy as np
import tensorflow as tf
from helia_core_tester.generation.ops.base import OperationBase
from pathlib import Path

class OpTransposeConv(OperationBase):
    """
    TransposeConv operation.
    """
    
    def build_keras_model(self) -> tf.keras.Model:
        """Build Keras model for TransposeConv operation.
        """
        input_shape = self.desc['input_shape']
        filter_shape = self.desc['filter_shape']
        
        # Build input layer (exclude batch dimension in Input shape)
        inputs = tf.keras.Input(
            shape=input_shape[1:],
            batch_size=input_shape[0] if len(input_shape) > 0 else None,
            dtype=tf.float32,
            name='input'
        )
        
        # TransposeConv layer
        # filter_shape is [KH, KW, OutCh, InCh] in TensorFlow format
        transpose_conv_kwargs = {
            'filters': filter_shape[2],  # Number of output channels (third dimension)
            'kernel_size': filter_shape[0:2],  # Kernel height and width (first two dimensions)
            'strides': tuple(self.desc.get('strides', [1, 1])),
            'padding': str(self.desc.get('padding', 'valid')).lower(),
            'use_bias': self.desc.get('use_bias', True),
            'name': 'transpose_conv'
        }
    
        transpose_conv_kwargs['kernel_initializer'] = tf.keras.initializers.GlorotUniform(seed=123)
        
        if transpose_conv_kwargs['use_bias']:
            transpose_conv_kwargs['bias_initializer'] = tf.keras.initializers.RandomUniform(
                minval=-0.5, maxval=0.5, seed=321
            )
        
        layer = tf.keras.layers.Conv2DTranspose(**transpose_conv_kwargs)
        outputs = layer(inputs)
        
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs], name='transpose_conv_model')
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
            rep_rng = np.random.default_rng(42)
            for _ in range(128): 
                if 'input_shape' in self.desc:
                    inputs = rep_rng.uniform(-8.0, 8.0, size=self.desc['input_shape']).astype(np.float32)
                    yield [inputs]
                elif 'input_1_shape' in self.desc and 'input_2_shape' in self.desc:
                    inputs1 = rep_rng.uniform(-8.0, 8.0, size=self.desc['input_1_shape']).astype(np.float32)
                    inputs2 = rep_rng.uniform(-8.0, 8.0, size=self.desc['input_2_shape']).astype(np.float32)
                    yield [inputs1, inputs2]
        
        converter.representative_dataset = representative_data_gen
        
        # Convert and save
        tflite_model = converter.convert()
        with open(out_path, 'wb') as f:
            f.write(tflite_model)
    
    def _select_cmsis_transpose_conv_kernel(self) -> Dict[str, str]:
        """
        Select appropriate CMSIS-NN transpose convolution kernel function.
        
        Returns:
            Dictionary with kernel function name, C types, and buffer size function
        """
        activation_dtype = self.desc.get('activation_dtype', 'S8').upper()
        weight_dtype = self.desc.get('weight_dtype', 'S8').upper()
        
        if activation_dtype == 'S8' and weight_dtype == 'S8':
            return {
                'kernel_fn': 'arm_transpose_conv_wrapper_s8',
                'kernel_get_buffer_size_fn': 'arm_transpose_conv_s8_get_buffer_size',
                'input_c_type': 'int8_t',
                'output_c_type': 'int8_t',
                'weight_c_type': 'int8_t',
                'bias_c_type': 'int32_t'
            }
        else:
            raise NotImplementedError(f"Unsupported TransposeConv dtype combo: {activation_dtype} x {weight_dtype}")
    
    def generate_c_files(self, output_dir: Path) -> None:
        """
        Generate C and H files from templates for TransposeConv operation.
        """
        from pathlib import Path
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder
        # from helia_core_tester.generation.utils.tflite_utils import run_inference
        
        name = self.desc['name']
        tflite_path = output_dir / f"{name}.tflite"
        if not tflite_path.exists():
            raise FileNotFoundError(f"TFLite file not found: {tflite_path}")
        
        # Select CMSIS kernel + types
        kernel_info = self._select_cmsis_transpose_conv_kernel()
        
        # Load LiteRT model for tensor extraction
        from helia_core_tester.generation.utils.litert_utils import (
            load_litert_model, get_operator_tensors_from_litert
        )
        
        model, subgraph = load_litert_model(str(tflite_path))
        
        # Get operator tensors (find TRANSPOSE_CONV if present)
        if len(subgraph.operators) == 0:
            raise ValueError("No operators found in model")

        transpose_op_index = 0
        try:
            from ai_edge_litert import schema_py_generated as litert
            for i, op in enumerate(subgraph.operators):
                opcode = model.operatorCodes[op.opcodeIndex]
                if opcode.builtinCode == litert.BuiltinOperator.TRANSPOSE_CONV:
                    transpose_op_index = i
                    break
        except Exception:
            transpose_op_index = 0
        
        op_tensors = get_operator_tensors_from_litert(model, subgraph, transpose_op_index)
        
        # Extract shapes from LiteRT
        if not op_tensors['inputs']:
            raise ValueError("No input tensors found")
        if not op_tensors['outputs']:
            raise ValueError("No output tensors found")
        
        # For TransposeConv, the operator inputs are: [output_shape_param, weights, input_data]
        # The actual input data tensor is the subgraph input, not necessarily op_tensors['inputs'][0]
        # Find the input data tensor (should be 4D and match subgraph inputs)
        input_shape = None
        subgraph_input_indices = set(subgraph.inputs)
        
        for input_tensor_info in op_tensors['inputs']:
            tensor_shape = input_tensor_info['shape']
            tensor_idx = input_tensor_info.get('index', -1)
            # The actual input data is the subgraph input (4D tensor)
            if tensor_idx in subgraph_input_indices and tensor_shape is not None and len(tensor_shape) == 4:
                input_shape = tensor_shape
                break
        
        # Fallback: use first 4D input tensor
        if input_shape is None:
            for input_tensor_info in op_tensors['inputs']:
                tensor_shape = input_tensor_info['shape']
                if tensor_shape is not None and len(tensor_shape) == 4:
                    input_shape = tensor_shape
                    break
        
        # Final fallback: use descriptor shape
        if input_shape is None:
            input_shape = self.desc.get('input_shape', [1, 1, 1, 1])
            print(f"Warning: Could not find input shape from LiteRT, using descriptor shape: {input_shape}")
        
        # Get output shape
        output_shape = None
        if op_tensors['outputs']:
            output_shape = op_tensors['outputs'][0]['shape']
        if output_shape is None and subgraph.outputs:
            try:
                from helia_core_tester.generation.utils.litert_utils import get_tensor_shape_from_litert
                output_tensor = subgraph.tensors[subgraph.outputs[0]]
                output_shape = get_tensor_shape_from_litert(output_tensor)
            except Exception:
                output_shape = None
        if output_shape is None:
            try:
                interpreter = self.load_litert_interpreter(str(tflite_path))
                out_details = interpreter.get_output_details()
                if out_details:
                    output_shape = tuple(out_details[0]["shape"])
            except Exception:
                output_shape = None
        
        # Ensure shapes are tuples
        if input_shape is not None:
            input_shape = tuple(input_shape)
        if output_shape is not None:
            output_shape = tuple(output_shape)
        
        # Ensure input_shape is 4D (NHWC)
        if len(input_shape) == 1:
            # If shape is [1], use descriptor shape
            input_shape = tuple(self.desc.get('input_shape', [1, 1, 1, 1]))
        elif len(input_shape) < 4:
            # Pad with batch dimension if needed
            input_shape = (1,) + input_shape if len(input_shape) == 3 else input_shape
        
        # Ensure output_shape is 4D (NHWC)
        if len(output_shape) < 4:
            output_shape = (1,) + output_shape if len(output_shape) == 3 else output_shape
        
        # Extract quantization parameters from LiteRT
        # For TransposeConv, find the actual input data tensor (4D, in subgraph inputs)
        # not the output shape parameter (1D)
        input_quant_litert = None
        subgraph_input_indices = set(subgraph.inputs)
        for input_tensor_info in op_tensors['inputs']:
            tensor_idx = input_tensor_info.get('index', -1)
            tensor_shape = input_tensor_info['shape']
            if tensor_idx in subgraph_input_indices and tensor_shape is not None and len(tensor_shape) == 4:
                input_quant_litert = input_tensor_info['quantization']
                break
        
        # Fallback to first input if not found
        if input_quant_litert is None:
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
        
        # Handle per-channel quantization (convert to scalar if needed)
        if isinstance(input_quant['scale'], (list, np.ndarray)):
            input_quant['scale'] = input_quant['scale'][0] if len(input_quant['scale']) > 0 else 1.0
        if isinstance(input_quant['zero_point'], (list, np.ndarray)):
            input_quant['zero_point'] = input_quant['zero_point'][0] if len(input_quant['zero_point']) > 0 else 0
        if isinstance(output_quant['scale'], (list, np.ndarray)):
            output_quant['scale'] = output_quant['scale'][0] if len(output_quant['scale']) > 0 else 1.0
        if isinstance(output_quant['zero_point'], (list, np.ndarray)):
            output_quant['zero_point'] = output_quant['zero_point'][0] if len(output_quant['zero_point']) > 0 else 0
        
        # Find weight quantization (from weight tensor in inputs)
        weight_quant = None
        bias_tensor_data = None
        use_bias = self.desc.get('use_bias', True)
        for input_tensor_info in op_tensors['inputs']:
            if input_tensor_info['data'] is not None:
                tensor_shape = input_tensor_info.get('shape', [])
                if len(tensor_shape) > 1:
                    # weight tensor
                    weight_quant = input_tensor_info['quantization']
                elif len(tensor_shape) == 1 and use_bias:
                    # bias tensor
                    bias_tensor_data = input_tensor_info['data']
        
        quant_params = {
            'input': input_quant or {'scale': 1.0, 'zero_point': 0, 'per_channel': False},
            'output': output_quant or {'scale': 1.0, 'zero_point': 0, 'per_channel': False},
            'weight': weight_quant or input_quant or {'scale': 1.0, 'zero_point': 0, 'per_channel': False}
        }
        
        # Extract weights and biases from LiteRT
        weights = op_tensors['weights']
        if use_bias and bias_tensor_data is not None:
            biases = bias_tensor_data
        else:
            biases = op_tensors['biases']
        
        # Load LiteRT interpreter for inference
        interpreter = self.load_litert_interpreter(str(tflite_path))
        
        if weights is not None:
            filter_shape = tuple(weights.shape)
            if len(filter_shape) == 4:

                filter_dims = {
                    'n': int(filter_shape[0]),  # OutCh (first dimension in TFLite)
                    'h': int(filter_shape[1]),  # KH (second dimension)
                    'w': int(filter_shape[2]),  # KW (third dimension)
                    'c': int(filter_shape[3])   # InCh (fourth dimension)
                }
            else:
                raise ValueError(f"Unsupported filter shape: {filter_shape}")
            # Ensure weights are int8 in generated C
            if weights.dtype != np.int8:
                weights = weights.astype(np.int8)
        else:
            # Fallback: descriptor format [KH, KW, OutCh, InCh]
            fs = tuple(self.desc['filter_shape'])
            filter_dims = {
                'n': int(fs[2]),  # OutCh
                'h': int(fs[0]),  # KH
                'w': int(fs[1]),  # KW
                'c': int(fs[3])   # InCh
            }
        
        builder = TemplateContextBuilder()
        input_dims = builder.nhwc_to_cmsis_dims(input_shape)
        output_dims = builder.nhwc_to_cmsis_dims(output_shape)
        
        # Build transpose conv parameters
        transpose_conv_params = builder.build_transpose_conv_params(
            self.desc,
            input_shape,
            (filter_dims['h'], filter_dims['w']),
            output_shape,
            quant_params['input'],
            quant_params['output']
        )
        
        # Build quantization parameters
        # The effective scale for multiplier/shift is NOT output_scale directly!
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
        if per_channel and isinstance(weight_scale, np.ndarray):
            # Per-channel: effective_scale[i] = (input_scale * weight_scale[i]) / output_scale
            effective_scales = (input_scale * weight_scale) / output_scale
            # Create a temporary quant_params dict with effective scales
            effective_quant = {
                'scale': effective_scales,
                'zero_point': output_quant.get('zero_point', 0),
                'per_channel': True
            }
            quant_params_dict = builder.build_quant_params(effective_quant, per_channel=True)
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
        
        quant_params_dict['per_channel'] = per_channel
        

        has_weight_sum = False
        weight_sum_size = 0
        if kernel_info["input_c_type"] == "int8_t" and weights is not None:

            has_weight_sum = True
        
        # Format weights and biases as C arrays
        weights_array_str = builder.format_array_as_c_literal(weights)
        has_biases = biases is not None and biases.size > 0
        if has_biases:
            # Convert biases to int32_t for CMSIS-NN
            if biases.dtype != np.int32:
                biases = biases.astype(np.int32)
            biases_array_str = builder.format_array_as_c_literal(biases)
        else:
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
        output_data = self.run_inference(str(tflite_path), input_q)
        
        # Format input and output arrays
        input_data_array_str = builder.format_array_as_c_literal(input_q)
        expected_output_array_str = builder.format_array_as_c_literal(output_data.astype(np.int8))
        
        # Calculate buffer size max
        activation_dtype = self.desc.get('activation_dtype', 'S8')
        buffer_size_max = builder.calculate_transpose_conv_buffer_size_max(
            input_dims, filter_dims, output_dims,
            output_dtype=activation_dtype
        )
        

        strides = self.desc.get('strides', [1, 1])
        stride_w = strides[1] if len(strides) > 1 else strides[0]
        stride_h = strides[0] if len(strides) > 0 else 1
        input_c = input_dims['c']
        filter_w = filter_dims['w']
        filter_h = filter_dims['h']
        filter_n = filter_dims['n']
        
        reverse_conv_possible = (stride_w <= 2) and (stride_h <= 2)
        reverse_conv_efficient = (input_c > 16)  # REVERSE_TCOL_EFFICIENT_THRESHOLD = 16
        
        if reverse_conv_possible and reverse_conv_efficient:
            reverse_conv_ctx_size = input_c * filter_w * filter_h * filter_n
        else:
            # If reverse_conv is not     used, we still need a buffer for output_ctx
            # Fallback: use output dimensions (conservative estimate
            reverse_conv_ctx_size = output_dims['w'] * output_dims['h'] * output_dims['c'] * 4
        

        if has_weight_sum:

            weight_sum_size = output_dims['c'] * 4  
        
        # Build template context
        context = {
            'name': name,
            'prefix': name,
            'input_dims': input_dims,
            'filter_dims': filter_dims,
            'output_dims': output_dims,
            'transpose_conv_params': transpose_conv_params,
            'quant_params': quant_params_dict,
            'weights_array': weights_array_str,
            'biases_array': biases_array_str,
            'has_biases': has_biases,
            'has_weight_sum': has_weight_sum,
            'weight_sum_size': weight_sum_size,
            'input_data_array': input_data_array_str,
            'expected_output_array': expected_output_array_str,
            'input_dtype': kernel_info["input_c_type"],
            'output_dtype': kernel_info["output_c_type"],
            'bias_dtype': kernel_info["bias_c_type"],
            'kernel_fn': kernel_info["kernel_fn"],
            'kernel_get_buffer_size_fn': kernel_info["kernel_get_buffer_size_fn"],
            'buffer_size_max': buffer_size_max,
            'reverse_conv_ctx_size': reverse_conv_ctx_size,
        }
        
        # Render templates
        includes_api_dir = output_dir / "includes"
        includes_api_dir.mkdir(parents=True, exist_ok=True)
        
        h_content = self.render_template("transpose_conv/transpose_conv.h.j2", context)
        h_path = includes_api_dir / f"{name}_transpose_conv.h"
        with open(h_path, 'w') as f:
            f.write(h_content)
        
        c_content = self.render_template("transpose_conv/transpose_conv.c.j2", context)
        c_path = output_dir / f"{name}_transpose_conv.c"
        with open(c_path, 'w') as f:
            f.write(c_content)
        
        # Also create a symlink or copy with the old naming convention for compatibility
        # (if needed by CMakeLists.txt pattern matching)
        
        cmake_context = {
            'name': name,
            'operator': self.desc.get('operator', 'TransposeConv'),
            'operator_name': 'transpose_conv'
        }
        cmake_content = self.render_template("common/CMakeLists.txt.j2", cmake_context)
        cmake_path = output_dir / "CMakeLists.txt"
        with open(cmake_path, 'w') as f:
            f.write(cmake_content)
        
        print(f"Generated C/H files and CMakeLists.txt for {name}")

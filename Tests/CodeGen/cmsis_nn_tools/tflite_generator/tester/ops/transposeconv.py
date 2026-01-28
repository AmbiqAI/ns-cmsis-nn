"""
TransposeConv operation implementation.
"""

from typing import Dict, Any
import numpy as np
import tensorflow as tf
from .base import OperationBase
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
        from ..utils.template_context import TemplateContextBuilder
        from ..utils.tflite_utils import run_inference
        
        name = self.desc['name']
        tflite_path = output_dir / f"{name}.tflite"
        if not tflite_path.exists():
            raise FileNotFoundError(f"TFLite file not found: {tflite_path}")
        
        # Select CMSIS kernel + types
        kernel_info = self._select_cmsis_transpose_conv_kernel()
        
        # Load LiteRT model for tensor extraction
        from ..utils.litert_utils import (
            load_litert_model, get_operator_tensors_from_litert
        )
        
        model, subgraph = load_litert_model(str(tflite_path))
        
        # Get operator tensors (first operator)
        if len(subgraph.operators) == 0:
            raise ValueError("No operators found in model")
        
        op_tensors = get_operator_tensors_from_litert(model, subgraph, 0)
        
        # Extract shapes from LiteRT
        if not op_tensors['inputs']:
            raise ValueError("No input tensors found")
        if not op_tensors['outputs']:
            raise ValueError("No output tensors found")
        
        # Get shapes from interpreter (more reliable than LiteRT for some models)
        interpreter = self.load_tflite_interpreter(str(tflite_path))
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        input_shape = tuple(input_details[0]['shape'])
        output_shape = tuple(output_details[0]['shape'])
        
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
        
        # Load interpreter for inference (still needed)
        interpreter = self.load_tflite_interpreter(str(tflite_path))
        
        # Weight tensor format in TFLite for TransposeConv:
        # TFLite stores weights as [OutCh, KH, KW, InCh] (transposed from Keras format)
        # Keras Conv2DTranspose uses [KH, KW, InCh, OutCh]
        # CMSIS expects [C_OUT, HK, WK, C_IN] format
        if weights is not None:
            filter_shape = tuple(weights.shape)
            if len(filter_shape) == 4:
                # TFLite format: [OutCh, KH, KW, InCh] -> CMSIS: [C_OUT, HK, WK, C_IN]
                # So: n=OutCh (index 0), h=KH (index 1), w=KW (index 2), c=InCh (index 3)
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
        
        # Compute weight_sum buffer size for S8 transpose convolutions
        # Use CMSIS-NN's arm_convolve_s8_get_weights_sum_size equivalent (matches test)
        # This is simpler and matches the CMSIS-NN test pattern
        has_weight_sum = False
        weight_sum_size = 0
        if kernel_info["input_c_type"] == "int8_t" and weights is not None:
            # arm_convolve_s8_get_weights_sum_size returns: output_dims->c * sizeof(int32_t)
            # This will be calculated later after output_dims is available
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
        output_data = self.run_inference(interpreter, input_q)
        
        # Format input and output arrays
        input_data_array_str = builder.format_array_as_c_literal(input_q)
        expected_output_array_str = builder.format_array_as_c_literal(output_data.astype(np.int8))
        
        # Calculate buffer size max
        activation_dtype = self.desc.get('activation_dtype', 'S8')
        buffer_size_max = builder.calculate_transpose_conv_buffer_size_max(
            input_dims, filter_dims, output_dims,
            output_dtype=activation_dtype
        )
        
        # Calculate reverse_conv buffer size using CMSIS-NN formula (matches test)
        # arm_transpose_conv_s8_get_reverse_conv_buffer_size returns:
        # - input_dims->c * filter_dims->w * filter_dims->h * filter_dims->n if reverse_conv is possible and efficient
        # - 0 otherwise
        # For efficiency: stride.w <= 2 && stride.h <= 2 && input_dims->c > 16
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
            # If reverse_conv is not used, we still need a buffer for output_ctx
            # Fallback: use output dimensions (conservative estimate)
            reverse_conv_ctx_size = output_dims['w'] * output_dims['h'] * output_dims['c'] * 4
        
        # Calculate weight_sum buffer size using CMSIS-NN function (matches test)
        # Use arm_convolve_s8_get_weights_sum_size equivalent: output_dims.c * sizeof(int32_t)
        # This is simpler and matches CMSIS-NN test pattern
        if has_weight_sum:
            # arm_convolve_s8_get_weights_sum_size returns: output_dims->c * sizeof(int32_t)
            weight_sum_size = output_dims['c'] * 4  # sizeof(int32_t) = 4
        
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
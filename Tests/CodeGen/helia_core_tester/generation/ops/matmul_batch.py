"""
MatMul operation implementation.
"""

from typing import Dict, Any
import numpy as np
import tensorflow as tf
from pathlib import Path
from helia_core_tester.generation.ops.base import OperationBase


class OpMatMul(OperationBase):
    """
    MatMul operation.
    """
    
    def build_keras_model(self) -> tf.keras.Model:
        """Build Keras model for MatMul operation."""
        input_1_shape = self.desc['input_1_shape']
        input_2_shape = self.desc['input_2_shape']
        
        # Get transpose options (adj_x and adj_y)
        adj_x = self.desc.get('adj_x', False)
        adj_y = self.desc.get('adj_y', False)
        
        # Build inputs (exclude batch dimension in Input shape)
        input1 = tf.keras.Input(shape=input_1_shape[1:], dtype=tf.float32, name='input1')
        input2 = tf.keras.Input(shape=input_2_shape[1:], dtype=tf.float32, name='input2')
        
        # Apply transpose if needed
        if adj_x:
            # Transpose last two dimensions: [..., M, K] -> [..., K, M]
            x1 = tf.keras.layers.Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1]))(input1)
        else:
            x1 = input1
            
        if adj_y:
            # Transpose last two dimensions: [..., K, N] -> [..., N, K]
            x2 = tf.keras.layers.Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1]))(input2)
        else:
            x2 = input2
        
        # Batch matrix multiplication: [batch, M, K] @ [batch, K, N] -> [batch, M, N]
        output = tf.keras.layers.Lambda(lambda inputs: tf.matmul(inputs[0], inputs[1]))([x1, x2])
        
        model = tf.keras.Model(inputs=[input1, input2], outputs=output)
        return model

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        """Convert Keras model to TFLite with quantization."""
        super().convert_to_tflite(model, out_path, rep_seed)
    
    def _numpy_dtype_to_c_type(self, np_dtype: np.dtype) -> str:
        """Map numpy dtype to C type string."""
        if np_dtype == np.int8:
            return 'int8_t'
        elif np_dtype == np.int16:
            return 'int16_t'
        else:
            raise ValueError(f"Unsupported numpy dtype: {np_dtype}")
    
    def _select_cmsis_matmul_kernel(self, input_lhs_dtype: np.dtype, input_rhs_dtype: np.dtype, output_dtype: np.dtype) -> Dict[str, str]:
        """
        Select appropriate CMSIS-NN kernel function for MatMul operation based on actual tensor dtypes.
        
        Args:
            input_lhs_dtype: numpy dtype of LHS tensor from TFLite
            input_rhs_dtype: numpy dtype of RHS tensor from TFLite
            output_dtype: numpy dtype of output tensor from TFLite
        
        Returns:
            Dictionary with kernel_fn, kernel_get_buffer_size_fn, 
            input_lhs_c_type, input_rhs_c_type, output_c_type
        """
        input_lhs_c_type = self._numpy_dtype_to_c_type(input_lhs_dtype)
        input_rhs_c_type = self._numpy_dtype_to_c_type(input_rhs_dtype)
        output_c_type = self._numpy_dtype_to_c_type(output_dtype)
        
        # Select kernel based on LHS dtype (output should match LHS)
        if input_lhs_c_type == 'int8_t':
            return {
                'kernel_fn': 'arm_batch_matmul_s8',
                'kernel_get_buffer_size_fn': 'arm_fully_connected_s8_get_buffer_size',
                'input_lhs_c_type': input_lhs_c_type,
                'input_rhs_c_type': input_rhs_c_type,
                'output_c_type': output_c_type
            }
        elif input_lhs_c_type == 'int16_t':
            return {
                'kernel_fn': 'arm_batch_matmul_s16',
                'kernel_get_buffer_size_fn': 'arm_fully_connected_s16_get_buffer_size',
                'input_lhs_c_type': input_lhs_c_type,
                'input_rhs_c_type': input_rhs_c_type,
                'output_c_type': output_c_type
            }
        else:
            raise NotImplementedError(f"Unsupported LHS dtype: {input_lhs_c_type}")
    
    def generate_c_files(self, output_dir) -> None:
        """
        Generate C and H files from templates for MatMul operation.
        """
        from pathlib import Path
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder
        from helia_core_tester.generation.utils.tflite_utils import calculate_multiplier_shift
        
        name = self.desc['name']
        tflite_path = output_dir / f"{name}.tflite"
        if not tflite_path.exists():
            raise FileNotFoundError(f"TFLite file not found: {tflite_path}")
        
        # Load LiteRT interpreter
        interpreter = self.load_litert_interpreter(str(tflite_path))
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # MatMul has two inputs (LHS and RHS)
        if len(input_details) != 2:
            raise ValueError(f"MatMul expects 2 inputs, got {len(input_details)}")
        
        # Extract dtypes from TFLite tensors
        input_lhs_dtype = input_details[0]['dtype']
        input_rhs_dtype = input_details[1]['dtype']
        output_dtype = output_details[0]['dtype']
        
        input_lhs_shape = tuple(input_details[0]['shape'])
        input_rhs_shape = tuple(input_details[1]['shape'])
        output_shape = tuple(output_details[0]['shape'])
        
        # Select CMSIS kernel + types based on actual dtypes from TFLite
        kernel_info = self._select_cmsis_matmul_kernel(input_lhs_dtype, input_rhs_dtype, output_dtype)
        
        # Extract quantization parameters from interpreter (more reliable than LiteRT)
        # LiteRT sometimes returns incorrect scales (e.g., 1.0 instead of actual scale)
        input_lhs_qp = input_details[0].get('quantization_parameters', {})
        input_rhs_qp = input_details[1].get('quantization_parameters', {})
        output_qp = output_details[0].get('quantization_parameters', {})
        
        # LHS quantization parameters
        input_lhs_quant = {
            'scale': input_lhs_qp.get('scales', [1.0]),
            'zero_point': input_lhs_qp.get('zero_points', [0]),
            'per_channel': len(input_lhs_qp.get('scales', [])) > 1
        }
        
        # RHS quantization parameters (always S8)
        input_rhs_quant = {
            'scale': input_rhs_qp.get('scales', [1.0]),
            'zero_point': input_rhs_qp.get('zero_points', [0]),
            'per_channel': len(input_rhs_qp.get('scales', [])) > 1
        }
        
        # Output quantization parameters
        output_quant = {
            'scale': output_qp.get('scales', [1.0]),
            'zero_point': output_qp.get('zero_points', [0]),
            'per_channel': len(output_qp.get('scales', [])) > 1
        }
        
        builder = TemplateContextBuilder()
        
        # Get transpose flags from descriptor
        adj_x = self.desc.get('adj_x', False)
        adj_y = self.desc.get('adj_y', False)
        
        # Convert shapes to CMSIS dims
        # For batch matmul: [batch, M, K] @ [batch, K, N] -> [batch, M, N]
        # CMSIS-NN expects RHS to be transposed: [batch, N, K]
        # For adj_x == True, CMSIS expects LHS already transposed: [batch, K, M]
        if len(input_lhs_shape) == 3 and adj_x:
            transposed_lhs_shape = (input_lhs_shape[0], input_lhs_shape[2], input_lhs_shape[1])
            input_lhs_dims = builder.nhwc_to_cmsis_dims(transposed_lhs_shape)
        else:
            input_lhs_dims = builder.nhwc_to_cmsis_dims(input_lhs_shape)
        
        # For RHS: if adj_y is False, CMSIS-NN expects transposed RHS
        # TFLite has [batch, K, N], CMSIS-NN expects [batch, N, K]
        # nhwc_to_cmsis_dims maps 3D [batch, H, W] to n=1, h=batch, w=H, c=W
        # So for transposed RHS [batch, N, K], we need: n=1, h=batch, w=N, c=K
        if len(input_rhs_shape) == 3:
            if not adj_y:
                # RHS needs to be transposed for CMSIS-NN
                # Original: [batch, K, N] -> Transposed: [batch, N, K]
                # Use nhwc_to_cmsis_dims mapping: n=1, h=batch, w=N, c=K
                transposed_rhs_shape = (input_rhs_shape[0], input_rhs_shape[2], input_rhs_shape[1])
                input_rhs_dims = builder.nhwc_to_cmsis_dims(transposed_rhs_shape)
            else:
                # RHS is already transposed in TFLite, use as-is
                input_rhs_dims = builder.nhwc_to_cmsis_dims(input_rhs_shape)
        else:
            input_rhs_dims = builder.nhwc_to_cmsis_dims(input_rhs_shape)
        
        output_dims = builder.nhwc_to_cmsis_dims(output_shape)
        
        # Build FC parameters for BMM
        # LHS is treated as "input", RHS is treated as "weights"
        fc_params = builder.build_fc_params(
            self.desc,
            input_lhs_quant,  # LHS quantization
            input_rhs_quant,  # RHS quantization (always S8)
            output_quant     # Output quantization
        )
        
        # Build BMM params
        # Note: CMSIS-NN's adj_y is inverted compared to TFLite's adj_y
        # This is because CMSIS-NN expects RHS to be transposed when adj_y is True
        bmm_params = {
            'adj_x': bool(adj_x),
            'adj_y': bool(not adj_y),  # Invert adj_y to match CMSIS-NN convention
            'fc_params': fc_params
        }
        
        # Extract scales and zero points for effective scale calculation
        input_lhs_scale = input_lhs_quant.get('scale', 1.0)
        if isinstance(input_lhs_scale, (list, np.ndarray)):
            input_lhs_scale = float(input_lhs_scale[0])
        else:
            input_lhs_scale = float(input_lhs_scale)
        
        input_rhs_scale = input_rhs_quant.get('scale', 1.0)
        if isinstance(input_rhs_scale, (list, np.ndarray)):
            input_rhs_scale = float(input_rhs_scale[0])
        else:
            input_rhs_scale = float(input_rhs_scale)
        
        output_scale = output_quant.get('scale', 1.0)
        if isinstance(output_scale, (list, np.ndarray)):
            output_scale = float(output_scale[0])
        else:
            output_scale = float(output_scale)
        
        # Effective scale: (input_lhs_scale * input_rhs_scale) / output_scale
        effective_scale = (input_lhs_scale * input_rhs_scale) / output_scale
        effective_scale = float(effective_scale)
        
        effective_quant = {
            'scale': effective_scale,
            'zero_point': output_quant.get('zero_point', 0),
            'per_channel': False
        }
        quant_params_dict = builder.build_quant_params(effective_quant, per_channel=False)
        
        # NOTE: Do NOT reduce multiplier here for S16 batch matmul.
        # arm_batch_matmul_s16 internally applies REDUCE_MULTIPLIER on the Q31 multiplier.
        # Reducing here would zero-out the multiplier and produce all-zero outputs.
        
        # Generate input data and quantize
        rng_state = self.rng.__getstate__()
        self.rng = np.random.default_rng(self.seed)
        
        input_lhs_data = self.rng.uniform(-1.0, 1.0, size=input_lhs_shape).astype(np.float32)
        input_rhs_data = self.rng.uniform(-1.0, 1.0, size=input_rhs_shape).astype(np.float32)
        
        self.rng.__setstate__(rng_state)
        
        # Extract LHS quantization parameters
        input_lhs_scale_val = input_lhs_quant.get('scale', 1.0)
        input_lhs_zp_val = input_lhs_quant.get('zero_point', 0)
        if isinstance(input_lhs_scale_val, (list, np.ndarray)):
            input_lhs_scale_val = float(input_lhs_scale_val[0])
        if isinstance(input_lhs_zp_val, (list, np.ndarray)):
            input_lhs_zp_val = int(input_lhs_zp_val[0])
        
        # Extract RHS quantization parameters
        input_rhs_scale_val = input_rhs_quant.get('scale', 1.0)
        input_rhs_zp_val = input_rhs_quant.get('zero_point', 0)
        if isinstance(input_rhs_scale_val, (list, np.ndarray)):
            input_rhs_scale_val = float(input_rhs_scale_val[0])
        if isinstance(input_rhs_zp_val, (list, np.ndarray)):
            input_rhs_zp_val = int(input_rhs_zp_val[0])
        
        # Determine quantization limits and numpy dtypes from TFLite dtypes
        def get_quantization_params(np_dtype):
            if np_dtype == np.int8:
                return np.int8, -128, 127
            elif np_dtype == np.int16:
                return np.int16, -32768, 32767
            else:
                raise ValueError(f"Unsupported numpy dtype: {np_dtype}")
        
        input_lhs_np_dtype, input_lhs_qmin, input_lhs_qmax = get_quantization_params(input_lhs_dtype)
        input_rhs_np_dtype, input_rhs_qmin, input_rhs_qmax = get_quantization_params(input_rhs_dtype)
        
        # Quantize LHS
        input_lhs_q = np.round(input_lhs_data / float(input_lhs_scale_val) + float(input_lhs_zp_val)).astype(np.int32)
        input_lhs_q = np.clip(input_lhs_q, input_lhs_qmin, input_lhs_qmax).astype(input_lhs_np_dtype)
        
        # Quantize RHS
        input_rhs_q = np.round(input_rhs_data / float(input_rhs_scale_val) + float(input_rhs_zp_val)).astype(np.int32)
        input_rhs_q = np.clip(input_rhs_q, input_rhs_qmin, input_rhs_qmax).astype(input_rhs_np_dtype)
        
        # Run TFLite inference (use dtypes as extracted from TFLite)
        interpreter.set_tensor(input_details[0]['index'], input_lhs_q)
        interpreter.set_tensor(input_details[1]['index'], input_rhs_q)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        output_data = np.array(output_data)
        
        # Ensure output dtype matches TFLite output dtype
        if output_data.dtype != output_dtype:
            output_data = output_data.astype(output_dtype)
        
        # Transpose RHS for CMSIS-NN if adj_y is False
        # CMSIS-NN expects RHS to be transposed: [batch, K, N] -> [batch, N, K]
        # if not adj_y, transpose RHS
        input_rhs_q_for_cmsis = input_rhs_q.copy()
        if not adj_y and len(input_rhs_shape) == 3:
            # Transpose last two dimensions: [batch, K, N] -> [batch, N, K]
            input_rhs_q_for_cmsis = np.transpose(input_rhs_q_for_cmsis, (0, 2, 1))

        # Transpose LHS for CMSIS-NN if adj_x is True
        # CMSIS-NN expects LHS to be already transposed: [batch, M, K] -> [batch, K, M]
        input_lhs_q_for_cmsis = input_lhs_q.copy()
        if adj_x and len(input_lhs_shape) == 3:
            input_lhs_q_for_cmsis = np.transpose(input_lhs_q_for_cmsis, (0, 2, 1))
        
        # Format arrays (use transposed LHS/RHS for CMSIS-NN if required)
        input_lhs_array_str = builder.format_array_as_c_literal(input_lhs_q_for_cmsis)
        input_rhs_array_str = builder.format_array_as_c_literal(input_rhs_q_for_cmsis)
        expected_output_array_str = builder.format_array_as_c_literal(output_data)
        
        # Calculate buffer size max (use RHS dims as filter_dims for buffer size calculation)
        # For batch matmul, buffer size is similar to fully connected
        # Use RHS shape as "filter" for buffer size calculation
        if len(input_rhs_shape) == 3:
            # [batch, K, N] -> filter_dims: n=K, c=N, h=1, w=1
            filter_dims_for_buffer = {
                'n': int(input_rhs_shape[1]),  # K
                'h': 1,
                'w': 1,
                'c': int(input_rhs_shape[2])   # N
            }
        else:
            raise ValueError(f"Unsupported RHS shape for buffer size: {input_rhs_shape}")
        
        activation_dtype = self.desc.get('activation_dtype', 'S8')
        buffer_size_max = builder.calculate_fc_buffer_size_max(
            filter_dims_for_buffer,
            output_dtype=activation_dtype
        )
        
        # Build template context
        context = {
            'name': name,
            'prefix': name,
            'input_lhs_dims': input_lhs_dims,
            'input_rhs_dims': input_rhs_dims,
            'output_dims': output_dims,
            'bmm_params': bmm_params,
            'quant_params': quant_params_dict,
            'input_lhs_array': input_lhs_array_str,
            'input_rhs_array': input_rhs_array_str,
            'expected_output_array': expected_output_array_str,
            'input_dtype': kernel_info["input_lhs_c_type"],  # LHS dtype from TFLite
            'input_rhs_dtype': kernel_info["input_rhs_c_type"],  # RHS dtype from TFLite
            'output_dtype': kernel_info["output_c_type"],  # Output dtype from TFLite
            'kernel_fn': kernel_info["kernel_fn"],
            'kernel_get_buffer_size_fn': kernel_info["kernel_get_buffer_size_fn"],
            'buffer_size_max': buffer_size_max,
        }
        
        # Render templates
        includes_api_dir = output_dir / "includes"
        includes_api_dir.mkdir(parents=True, exist_ok=True)
        
        h_content = self.render_template("matmul_batch/matmul_batch.h.j2", context)
        h_path = includes_api_dir / f"{name}_matmul_batch.h"
        with open(h_path, 'w') as f:
            f.write(h_content)
        
        c_content = self.render_template("matmul_batch/matmul_batch.c.j2", context)
        c_path = output_dir / f"{name}_matmul_batch.c"
        with open(c_path, 'w') as f:
            f.write(c_content)
        
        cmake_context = {
            'name': name,
            'operator': self.desc.get('operator', 'MatMul'),
            'operator_name': 'matmul_batch'
        }
        cmake_content = self.render_template("common/CMakeLists.txt.j2", cmake_context)
        cmake_path = output_dir / "CMakeLists.txt"
        with open(cmake_path, 'w') as f:
            f.write(cmake_content)
        
        print(f"Generated C/H files and CMakeLists.txt for {name}")

"""
Logistic (Sigmoid) operation implementation.
"""

from typing import Dict, Any
import numpy as np
import tensorflow as tf
from pathlib import Path
from helia_core_tester.generation.ops.base import OperationBase


class OpLogistic(OperationBase):
    """
    Logistic (Sigmoid) operation.
    """
    
    def build_keras_model(self) -> tf.keras.Model:
        """Build Keras model for Logistic operation."""
        input_shape = self.desc['input_shape']
        
        # Build model with float32 inputs (will be quantized later)
        inputs = tf.keras.Input(shape=input_shape[1:], dtype=tf.float32, name='input')
        
        # Logistic (Sigmoid) operation
        output = tf.keras.layers.Activation('sigmoid')(inputs)
            
        model = tf.keras.Model(inputs=[inputs], outputs=output)
        return model

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        """Convert Keras model to TFLite with quantization."""
        # Create converter
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Apply quantization based on activation_dtype
        # NOTE: CMSIS-NN only supports S16 for Logistic, so convert S8 to S16
        activation_dtype = self.desc.get('activation_dtype', 'S8')
        if activation_dtype == 'S8':
            # CMSIS-NN Logistic only supports S16, so use S16 quantization
            activation_dtype = 'S16'
        
        if activation_dtype == 'S16':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8
            ]
            converter.inference_input_type = tf.int16
            converter.inference_output_type = tf.int16
        else:
            raise ValueError(f"Unsupported activation_dtype for Logistic: {activation_dtype} (only S16 supported)")

        
        # Generate representative dataset
        def representative_data_gen():
            rng = np.random.default_rng(rep_seed)
            for _ in range(100):
                if 'input_shape' in self.desc:
                    inputs = rng.uniform(-1.0, 1.0, size=self.desc['input_shape']).astype(np.float32)
                    yield [inputs]
                elif 'input_1_shape' in self.desc and 'input_2_shape' in self.desc:
                    inputs1 = rng.uniform(-1.0, 1.0, size=self.desc['input_1_shape']).astype(np.float32)
                    inputs2 = rng.uniform(-1.0, 1.0, size=self.desc['input_2_shape']).astype(np.float32)
                    yield [inputs1, inputs2]
        
        converter.representative_dataset = representative_data_gen
        
        # Convert and save
        tflite_model = converter.convert()
        with open(out_path, 'wb') as f:
            f.write(tflite_model)
    
    def _select_cmsis_logistic_kernel(self) -> Dict[str, str]:
        """
        Select appropriate CMSIS-NN kernel function for Logistic operation.
        
        Returns:
            Dictionary with kernel_fn, input_c_type, output_c_type
        """
        # NOTE: CMSIS-NN only supports S16 for Logistic, so convert S8 to S16
        activation_dtype = self.desc.get('activation_dtype', 'S8')
        if activation_dtype == 'S8':
            activation_dtype = 'S16'
        
        if activation_dtype == 'S16':
            return {
                'kernel_fn': 'arm_logistic_s16',
                'input_c_type': 'int16_t',
                'output_c_type': 'int16_t'
            }
        else:
            raise NotImplementedError(f"Unsupported Logistic dtype: {activation_dtype} (only S16 supported)")
    
    def generate_c_files(self, output_dir: Path) -> None:
        """
        Generate C and H files from templates for Logistic operation.
        """
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder
        from helia_core_tester.generation.utils.tflite_utils import calculate_multiplier_shift
        
        name = self.desc['name']
        tflite_path = output_dir / f"{name}.tflite"
        if not tflite_path.exists():
            raise FileNotFoundError(f"TFLite file not found: {tflite_path}")
        
        # Select CMSIS kernel + types
        kernel_info = self._select_cmsis_logistic_kernel()
        
        # Load LiteRT model for shape and quantization extraction
        from helia_core_tester.generation.utils.litert_utils import get_operator_tensors_from_litert
        model, subgraph = self.load_litert_model(str(tflite_path))
        op_tensors = get_operator_tensors_from_litert(model, subgraph, 0)
        
        # Extract shapes from LiteRT
        input_shape = op_tensors['inputs'][0]['shape']
        output_shape = op_tensors['outputs'][0]['shape']
        
        # Ensure shapes are tuples
        if input_shape is not None:
            input_shape = tuple(input_shape)
        if output_shape is not None:
            output_shape = tuple(output_shape)
        
        # Extract quantization from LiteRT
        input_quant = op_tensors['inputs'][0]['quantization']
        output_quant = op_tensors['outputs'][0]['quantization']
        
        input_scale = input_quant.get('scale', 1.0)
        input_zp = input_quant.get('zero_point', 0)
        output_scale = output_quant.get('scale', 1.0)
        output_zp = output_quant.get('zero_point', 0)
        
        # Handle per-channel quantization (convert to scalar)
        if isinstance(input_scale, (list, np.ndarray)):
            input_scale = float(input_scale[0])
        if isinstance(input_zp, (list, np.ndarray)):
            input_zp = int(input_zp[0])
        if isinstance(output_scale, (list, np.ndarray)):
            output_scale = float(output_scale[0])
        if isinstance(output_zp, (list, np.ndarray)):
            output_zp = int(output_zp[0])
        
        input_scale = float(input_scale)
        input_zp = int(input_zp)
        output_scale = float(output_scale)
        output_zp = int(output_zp)
        
        builder = TemplateContextBuilder()
        
        # Convert shapes to CMSIS dims
        input_dims = builder.nhwc_to_cmsis_dims(input_shape)
        output_dims = builder.nhwc_to_cmsis_dims(output_shape)
        
        # For S16 Logistic: 
        # Constants from CMSIS-NN:
        # kInputIntegerBits = 3
        # kOutputFractionalBits = 15
        #
        # Formula: multiplier = input_scale * 4096.0 * 3.0, then adjust input_left_shift
        kInputIntegerBits = 3
        kOutputFractionalBits = 15
        
        # Check if output_scale is power-of-two (2^-15)
        import math
        def checked_log2(x):
            """Check if x is a power of 2 and return its log2."""
            if x <= 0:
                return False, 0
            log2_val = math.log2(x)
            is_pot = abs(log2_val - round(log2_val)) < 1e-9
            return is_pot, int(round(log2_val))
        
        is_pot_out, log2_out = checked_log2(output_scale)
        
        # Verify output scale is 2^-15 (or close)
        # For S16 logistic, output_scale should be approximately 1/32768 = 2^-15
        
        # Check if input_scale is power-of-two
        is_pot_in, log2_in = checked_log2(input_scale)
        calculated_left_shift = (15 - kInputIntegerBits) + log2_in
        param_scale_pot = is_pot_in and (calculated_left_shift == 0)
        
        if param_scale_pot:
            # Shift-only scaling (multiplier = 0)
            mult = 0
            input_left_shift = calculated_left_shift
        else:
            # Need to compute a multiplier
            # multiplier = input_scale * 4096.0 * 3.0
            multiplier = float(input_scale) * 4096.0 * 3.0
            input_left_shift = 0
            
            # Scale multiplier and adjust left shift until it fits in int16
            while (multiplier <= (32767.0 / 2.0)) and (input_left_shift <= 30):
                multiplier *= 2.0
                input_left_shift += 1
            
            mult = int(multiplier)
            
            # Clamp to int16 range
            if mult > 32767:
                mult = 32767
        
        # Generate input data and quantize
        rng_state = self.rng.__getstate__()
        self.rng = np.random.default_rng(self.seed)
        
        input_data = self.rng.uniform(-1.0, 1.0, size=input_shape).astype(np.float32)
        
        self.rng.__setstate__(rng_state)
        
        # Quantize inputs
        if kernel_info["input_c_type"] == "int16_t":
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
        output_data = interpreter.get_tensor(output_details[0]['index'])
        output_data = np.array(output_data)
        
        # Format arrays
        input_array_str = builder.format_array_as_c_literal(input_q)
        expected_output_array_str = builder.format_array_as_c_literal(output_data)
        
        # Calculate output size
        output_size = int(np.prod(output_shape))
        
        # Build template context
        context = {
            'name': name,
            'prefix': name,
            'input_dims': input_dims,
            'output_dims': output_dims,
            'output_size': output_size,
            'input_multiplier': int(mult),
            'input_left_shift': int(input_left_shift),
            'input_data_array': input_array_str,
            'expected_output_array': expected_output_array_str,
            'input_dtype': kernel_info["input_c_type"],
            'output_dtype': kernel_info["output_c_type"],
            'kernel_fn': kernel_info["kernel_fn"],
        }
        
        # Render templates
        includes_api_dir = output_dir / "includes"
        includes_api_dir.mkdir(parents=True, exist_ok=True)
        
        h_content = self.render_template("logistic/logistic.h.j2", context)
        h_path = includes_api_dir / f"{name}_logistic.h"
        with open(h_path, 'w') as f:
            f.write(h_content)
        
        c_content = self.render_template("logistic/logistic.c.j2", context)
        c_path = output_dir / f"{name}_logistic.c"
        with open(c_path, 'w') as f:
            f.write(c_content)
        
        cmake_context = {
            'name': name,
            'operator': self.desc.get('operator', 'Logistic'),
            'operator_name': 'logistic'
        }
        cmake_content = self.render_template("common/CMakeLists.txt.j2", cmake_context)
        cmake_path = output_dir / "CMakeLists.txt"
        with open(cmake_path, 'w') as f:
            f.write(cmake_content)
        
        print(f"Generated C/H files and CMakeLists.txt for {name}")


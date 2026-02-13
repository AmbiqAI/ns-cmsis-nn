"""
ArgMin operation implementation.
"""

from typing import Dict, Any
import numpy as np
import tensorflow as tf
from pathlib import Path
from cmsis_nn_tools.generation.tflite_generator.tester.ops.base import OperationBase


class OpArgMin(OperationBase):
    """
    ArgMin operation.
    """
    
    def build_keras_model(self) -> tf.keras.Model:
        """Build Keras model for ArgMin operation."""
        input_shape = self.desc['input_shape']
        inputs = tf.keras.Input(shape=input_shape[1:], dtype=tf.float32, name='input')
        
        # Get axis from descriptor (default to last dimension)
        axis = self.desc.get('axis', -1)
        
        # ArgMin operation - returns int32 indices
        x = tf.keras.layers.Lambda(
            lambda x: tf.argmin(x, axis=axis, output_type=tf.int32),
            name='argmin'
        )(inputs)
        
        model = tf.keras.Model(inputs=inputs, outputs=x)
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
            converter.inference_output_type = tf.int32  # ArgMin outputs int32 indices
        elif activation_dtype == 'S16':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8
            ]
            converter.inference_input_type = tf.int16
            converter.inference_output_type = tf.int32  # ArgMin outputs int32 indices
        
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
    
    def _select_cmsis_argmin_kernel(self) -> Dict[str, str]:
        """
        Select appropriate CMSIS-NN kernel function for ArgMin operation.
        
        Returns:
            Dictionary with kernel_fn, input_c_type, output_c_type
        """
        activation_dtype = self.desc.get('activation_dtype', 'S8')
        
        if activation_dtype == 'S8':
            return {
                'kernel_fn': 'arm_argmin_s8',
                'input_c_type': 'int8_t',
                'output_c_type': 'int32_t'
            }
        elif activation_dtype == 'S16':
            return {
                'kernel_fn': 'arm_argmin_s16',
                'input_c_type': 'int16_t',
                'output_c_type': 'int32_t'
            }
        else:
            raise NotImplementedError(f"Unsupported ArgMin dtype: {activation_dtype}")
    
    def generate_c_files(self, output_dir: Path) -> None:
        """
        Generate C and H files from templates for ArgMin operation.
        """
        from cmsis_nn_tools.generation.tflite_generator.tester.utils.template_context import TemplateContextBuilder
        
        name = self.desc['name']
        tflite_path = output_dir / f"{name}.tflite"
        if not tflite_path.exists():
            raise FileNotFoundError(f"TFLite file not found: {tflite_path}")
        
        # Select CMSIS kernel + types
        kernel_info = self._select_cmsis_argmin_kernel()
        
        # Load LiteRT interpreter
        interpreter = self.load_litert_interpreter(str(tflite_path))
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        input_shape = tuple(input_details[0]['shape'])
        output_shape = tuple(output_details[0]['shape'])
        
        builder = TemplateContextBuilder()
        
        # Convert shapes to CMSIS dims
        input_dims = builder.nhwc_to_cmsis_dims(input_shape)
        
        # Extract axis from descriptor (default to -1 for last dimension)
        axis = self.desc.get('axis', -1)
        # Convert to 0-based and account for batch dimension
        # For NHWC: 0=N, 1=H, 2=W, 3=C
        if axis < 0:
            axis = len(input_shape) + axis
        # axis is now 0-3 for NHWC
        
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
        
        # Run inference
        interpreter.set_tensor(input_details[0]['index'], input_q)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        output_data = np.array(output_data).astype(np.int32)
        
        # Format arrays
        input_array_str = builder.format_array_as_c_literal(input_q)
        expected_output_array_str = builder.format_array_as_c_literal(output_data)
        
        # Build template context
        context = {
            'name': name,
            'prefix': name,
            'input_dims': input_dims,
            'axis': axis,
            'input_data_array': input_array_str,
            'expected_output_array': expected_output_array_str,
            'input_dtype': kernel_info["input_c_type"],
            'output_dtype': kernel_info["output_c_type"],
            'kernel_fn': kernel_info["kernel_fn"],
            'output_size': int(np.prod(output_shape)),
        }
        
        # Render templates
        includes_api_dir = output_dir / "includes"
        includes_api_dir.mkdir(parents=True, exist_ok=True)
        
        h_content = self.render_template("argmin/argmin.h.j2", context)
        h_path = includes_api_dir / f"{name}_argmin.h"
        with open(h_path, 'w') as f:
            f.write(h_content)
        
        c_content = self.render_template("argmin/argmin.c.j2", context)
        c_path = output_dir / f"{name}_argmin.c"
        with open(c_path, 'w') as f:
            f.write(c_content)
        
        cmake_context = {
            'name': name,
            'operator': self.desc.get('operator', 'ArgMin'),
            'operator_name': 'argmin'
        }
        cmake_content = self.render_template("common/CMakeLists.txt.j2", cmake_context)
        cmake_path = output_dir / "CMakeLists.txt"
        with open(cmake_path, 'w') as f:
            f.write(cmake_content)
        
        print(f"Generated C/H files and CMakeLists.txt for {name}")


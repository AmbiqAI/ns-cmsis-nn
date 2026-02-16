"""
VariableUpdate operation implementation.
"""

from typing import Dict, Any
import numpy as np
import tensorflow as tf
from helia_core_tester.generation.ops.base import OperationBase


class OpVariableUpdate(OperationBase):
    """
    VariableUpdate operation - variable read/write operations.
    """
    
    def build_keras_model(self) -> tf.keras.Model:
        """Build Keras model for VariableUpdate operation.
        
        VariableUpdate typically involves:
        1. Reading a variable
        2. Updating it with a new value
        3. Returning the updated value
        
        For TFLite, we'll model this as an identity operation that can be
        optimized by the converter to use variables.
        """
        input_shape = self.desc.get('input_shape') or self.desc.get('value_shape')
        
        if input_shape is None:
            raise ValueError("VariableUpdate operation requires 'input_shape' or 'value_shape' in descriptor")
        
        inputs = tf.keras.Input(shape=input_shape[1:], dtype=tf.float32, name='input')
        
        # VariableUpdate - for now, we'll use an identity operation
        # In practice, TFLite will handle variable operations
        x = tf.keras.layers.Lambda(lambda x: x, name='variable_update')(inputs)
        
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
                shape = self.desc.get('input_shape') or self.desc.get('value_shape')
                if shape:
                    inputs = self.rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
                    yield [inputs]
        
        converter.representative_dataset = representative_data_gen
        
        tflite_model = converter.convert()
        with open(out_path, 'wb') as f:
            f.write(tflite_model)
    
    def generate_c_files(self, output_dir) -> None:
        """
        VariableUpdate is not supported by CMSIS-NN.
        
        VariableUpdate is a stateful operation that involves reading and writing
        to persistent variables. CMSIS-NN is designed for stateless inference
        operations and does not provide kernels for variable management.
        
        Raises:
            NotImplementedError: Always, as VariableUpdate is not supported
        """
        raise NotImplementedError(
            "VariableUpdate is not supported by CMSIS-NN. "
            "VariableUpdate is a stateful operation that requires persistent "
            "variable storage, which is outside the scope of CMSIS-NN's stateless "
            "inference kernels. Consider using stateless operations or handling "
            "variable updates at a higher level in your application."
        )


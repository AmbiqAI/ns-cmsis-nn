"""
Fill operation implementation.
"""

from typing import Dict, Any
import numpy as np
import tensorflow as tf
from helia_core_tester.generation.ops.base import OperationBase


class OpFill(OperationBase):
    """
    Fill operation - creates a tensor filled with a constant value.
    """
    
    def build_keras_model(self) -> tf.keras.Model:
        """Build Keras model for Fill operation.
        
        Fill creates a tensor of the specified shape filled with a constant value.
        Since this is a constant operation, we use a Lambda layer that returns
        a constant tensor. We use a dummy input to satisfy Keras model requirements.
        """
        # Get output_shape from descriptor
        output_shape = self.desc.get('output_shape')
        if output_shape is None:
            raise ValueError("Fill operation requires 'output_shape' in descriptor")
        
        # Get fill value (default to 0.0)
        fill_value = self.desc.get('value', 0.0)
        
        # Create a model with a dummy input (Keras requires at least one input)
        # The dummy input is ignored, and we output a constant tensor
        dummy_input = tf.keras.Input(shape=(1,), dtype=tf.float32, name='dummy')
        
        # Create a Lambda layer that returns a constant tensor
        # tf.fill needs the shape as a tensor, so we use tf.constant for the shape
        def fill_op(_):
            # Create shape tensor from output_shape
            shape_tensor = tf.constant(output_shape, dtype=tf.int32)
            return tf.fill(shape_tensor, fill_value)
        
        x = tf.keras.layers.Lambda(fill_op, name='fill')(dummy_input)
        
        model = tf.keras.Model(inputs=dummy_input, outputs=x)
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
            for _ in range(100):
                # Dummy input for Fill operation
                dummy = np.array([0.0], dtype=np.float32)
                yield [dummy]
        
        converter.representative_dataset = representative_data_gen
        
        # Convert and save
        tflite_model = converter.convert()
        with open(out_path, 'wb') as f:
            f.write(tflite_model)

    def generate_c_files(self, output_dir) -> None:
        """
        Generate C and H files - not yet implemented for Fill.
        """
        pass

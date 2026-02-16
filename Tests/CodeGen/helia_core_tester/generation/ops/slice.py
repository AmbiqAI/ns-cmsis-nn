"""
Slice operation implementation.
"""

from typing import Dict, Any
import tensorflow as tf
from helia_core_tester.generation.ops.base import OperationBase


class OpSlice(OperationBase):
    """
    Slice operation.
    """
    
    def build_keras_model(self) -> tf.keras.Model:
        """Build Keras model for Slice operation."""
        input_shape = self.desc['input_shape']
        begin = self.desc.get('begin')
        size = self.desc.get('size')
        
        if begin is None or size is None:
            raise ValueError("Slice operation requires 'begin' and 'size' in descriptor")
        
        inputs = tf.keras.Input(shape=input_shape[1:], dtype=tf.float32, name='input')
        
        # Slice operation
        def slice_op(x):
            return tf.slice(x, begin[1:], size[1:])  # Remove batch dimension
        
        x = tf.keras.layers.Lambda(slice_op, name='slice')(inputs)
        
        model = tf.keras.Model(inputs=inputs, outputs=x)
        return model

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        """Convert Keras model to TFLite with quantization."""
        super().convert_to_tflite(model, out_path, rep_seed)

    def generate_c_files(self, output_dir) -> None:
        """
        Generate C and H files - not yet implemented for Slice.
        """
        pass

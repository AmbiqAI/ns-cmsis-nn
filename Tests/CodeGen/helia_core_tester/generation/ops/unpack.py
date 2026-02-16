"""
Unpack operation implementation.
"""

from typing import Dict, Any
import tensorflow as tf
from helia_core_tester.generation.ops.base import OperationBase


class OpUnpack(OperationBase):
    """
    Unpack operation - splits a tensor along an axis.
    """
    
    def build_keras_model(self) -> tf.keras.Model:
        """Build Keras model for Unpack operation."""
        input_shape = self.desc['input_shape']
        axis = self.desc.get('axis', 0)
        num_tensors = self.desc.get('num_tensors', 1)
        
        inputs = tf.keras.Input(shape=input_shape[1:], dtype=tf.float32, name='input')
        
        # Adjust axis to account for batch dimension removal
        if axis >= 0:
            axis_adjusted = axis - 1 if axis > 0 else axis
        else:
            axis_adjusted = axis
        
        # Unpack operation - split along axis
        x = tf.unstack(inputs, axis=axis_adjusted, num=num_tensors)
        
        # For TFLite, we need to return a single output
        # We'll use the first unpacked tensor as output
        output = x[0] if len(x) > 0 else inputs
        
        model = tf.keras.Model(inputs=inputs, outputs=output)
        return model

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        """Convert Keras model to TFLite with quantization."""
        super().convert_to_tflite(model, out_path, rep_seed)

    def generate_c_files(self, output_dir) -> None:
        """
        Generate C and H files - not yet implemented for Unpack.
        """
        pass

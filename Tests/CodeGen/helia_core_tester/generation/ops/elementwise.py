"""
Elementwise operation implementation.
"""

from typing import Dict, Any
import tensorflow as tf
from helia_core_tester.generation.ops.base import OperationBase


class OpElementwise(OperationBase):
    """
    Elementwise operation.
    """
    
    def build_keras_model(self) -> tf.keras.Model:
        """Build Keras model for Elementwise operation."""
        input_1_shape = self.desc['input_1_shape']
        input_2_shape = self.desc['input_2_shape']
        input1 = tf.keras.Input(shape=input_1_shape[1:], dtype=tf.float32, name='input1')
        input2 = tf.keras.Input(shape=input_2_shape[1:], dtype=tf.float32, name='input2')
        x = tf.keras.layers.Multiply()([input1, input2])
        model = tf.keras.Model(inputs=[input1, input2], outputs=x)
        return model

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        """Convert Keras model to TFLite with quantization."""
        super().convert_to_tflite(model, out_path, rep_seed)

    def generate_c_files(self, output_dir) -> None:
        """Generate C and H files - not yet implemented for Elementwise."""
        raise NotImplementedError("Elementwise generate_c_files is not implemented")

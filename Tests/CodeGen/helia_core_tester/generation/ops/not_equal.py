"""
NotEqual comparison operation implementation.
"""

from typing import Dict, Any
import tensorflow as tf
from helia_core_tester.generation.ops.base import OperationBase


class OpNotEqual(OperationBase):
    """
    NotEqual comparison operation.
    """
    
    def build_keras_model(self) -> tf.keras.Model:
        """Build Keras model for NotEqual operation."""
        input_1_shape = self.desc['input_1_shape']
        input_2_shape = self.desc['input_2_shape']
        
        input1 = tf.keras.Input(shape=input_1_shape[1:], dtype=tf.float32, name='input1')
        input2 = tf.keras.Input(shape=input_2_shape[1:], dtype=tf.float32, name='input2')
        
        # NotEqual operation outputs boolean
        output = tf.keras.layers.Lambda(lambda x: tf.not_equal(x[0], x[1]))([input1, input2])
        
        model = tf.keras.Model(inputs=[input1, input2], outputs=output)
        return model

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        """Convert Keras model to TFLite with quantization."""
        self._convert_with_activation_quantization(model, out_path, output_type=tf.bool, rep_seed=rep_seed)

    def generate_c_files(self, output_dir) -> None:
        """
        Generate C and H files - not yet implemented for NotEqual.
        """
        pass

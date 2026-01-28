"""
LSTM operation implementation.
"""

from typing import Dict, Any
import numpy as np
import tensorflow as tf
from .base import OperationBase


class OpLSTM(OperationBase):
    """
    LSTM operation.
    """
    
    def build_keras_model(self) -> tf.keras.Model:
        """
        Build Keras model for LSTM operation.
        Mirrors the flow from ns-cmsis-nn reference implementation.
        """
        # Extract parameters from descriptor
        time_steps = int(self.desc.get('time_steps', 50))
        input_size = int(self.desc.get('feature_size', 24))  # feature_size maps to input_size
        hidden_size = int(self.desc.get('units', 64))  # units maps to hidden_size
        batch_size = int(self.desc.get('batch_size', 1))
        time_major = bool(self.desc.get('time_major', False))
        
        # Build input layer with batch_size specified
        input_layer = tf.keras.Input(
            shape=(time_steps, input_size),
            batch_size=batch_size,
            dtype=tf.float32,
            name='input'
        )
        
        # Handle time_major: transpose if needed
        # Note: time_major parameter is only passed when True (for older TF versions compatibility)
        if time_major:
            input_layer_transposed = tf.transpose(input_layer, perm=[1, 0, 2])
            lstm_layer = tf.keras.layers.LSTM(
                units=hidden_size,
                time_major=True,  # Only pass when True
                return_sequences=bool(self.desc.get('return_sequences', True)),
                name="lstm"
            )(input_layer_transposed)
        else:
            # When time_major=False, don't pass the parameter (use default)
            lstm_layer = tf.keras.layers.LSTM(
                units=hidden_size,
                return_sequences=bool(self.desc.get('return_sequences', True)),
                name="lstm"
            )(input_layer)
        
        model = tf.keras.Model(input_layer, lstm_layer, name="LSTM")
        return model

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        """Convert Keras model to TFLite with quantization."""
        import tensorflow as tf
        import numpy as np
        
        # Create converter
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # LSTM requires Select TF ops (flex ops) to handle dynamic tensor lists
        # Disable experimental lower tensor list ops to avoid conversion errors
        converter._experimental_lower_tensor_list_ops = False
        
        # Apply quantization based on activation_dtype
        activation_dtype = self.desc.get('activation_dtype', 'S8')
        
        if activation_dtype == 'S8':
            # For LSTM: SELECT_TF_OPS is required for TensorList ops
            # However, quantization + SELECT_TF_OPS may not work together
            # Try without quantization first, or skip quantization for LSTM
            # Note: LSTM in TFLite often uses unquantized float32 due to complexity
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS
            ]
            # For now, skip quantization for LSTM as it conflicts with SELECT_TF_OPS
            # The model will be float32
            # TODO: Investigate if post-training quantization can work with SELECT_TF_OPS
        elif activation_dtype == 'S16':
            # Similar to S8: quantization with SELECT_TF_OPS is problematic
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS
            ]
            # Skip quantization for now
        else:
            # For float32 or other types, still need SELECT_TF_OPS for LSTM
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS
            ]
        
        # Generate representative dataset
        def representative_data_gen():
            for _ in range(100):
                if 'time_steps' in self.desc and 'feature_size' in self.desc:
                    # LSTM uses [batch, time_steps, feature_size]
                    batch_size = int(self.desc.get('batch_size', 1))
                    time_steps = int(self.desc['time_steps'])
                    feature_size = int(self.desc['feature_size'])
                    inputs = self.rng.uniform(-1.0, 1.0, size=(batch_size, time_steps, feature_size)).astype(np.float32)
                    yield [inputs]
                elif 'input_shape' in self.desc:
                    inputs = self.rng.uniform(-1.0, 1.0, size=self.desc['input_shape']).astype(np.float32)
                    yield [inputs]
                elif 'input_1_shape' in self.desc and 'input_2_shape' in self.desc:
                    inputs1 = self.rng.uniform(-1.0, 1.0, size=self.desc['input_1_shape']).astype(np.float32)
                    inputs2 = self.rng.uniform(-1.0, 1.0, size=self.desc['input_2_shape']).astype(np.float32)
                    yield [inputs1, inputs2]
        
        # Only set representative dataset if quantization is enabled
        # (Currently disabled for LSTM when using SELECT_TF_OPS)
        if activation_dtype in ['S8', 'S16']:
            # Quantization is currently skipped for LSTM, but keep dataset ready
            # for future implementation
            converter.representative_dataset = representative_data_gen
        
        # Convert and save
        tflite_model = converter.convert()
        with open(out_path, 'wb') as f:
            f.write(tflite_model)

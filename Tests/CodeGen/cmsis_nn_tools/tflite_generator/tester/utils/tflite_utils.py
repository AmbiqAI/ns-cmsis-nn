"""
Utilities for working with TFLite models.
Extract quantization parameters, weights, biases, and run inference.
"""

import numpy as np
import tensorflow as tf
from typing import Dict, Tuple, Optional, Any
from pathlib import Path


def load_tflite_model(tflite_path: str) -> Tuple[bytes, tf.lite.Interpreter]:
    """
    Load TFLite model and create interpreter.
    
    Args:
        tflite_path: Path to .tflite file
        
    Returns:
        Tuple of (model_bytes, interpreter)
    """
    with open(tflite_path, 'rb') as f:
        model_bytes = f.read()
    
    interpreter = tf.lite.Interpreter(model_content=model_bytes)
    interpreter.allocate_tensors()
    
    return model_bytes, interpreter


def get_tensor_quantization(interpreter: tf.lite.Interpreter, tensor_index: int) -> Dict[str, Any]:
    """
    Extract quantization parameters from a tensor.
    
    Args:
        interpreter: TFLite interpreter
        tensor_index: Index of the tensor
        
    Returns:
        Dictionary with 'scale' and 'zero_point' keys
    """
    tensor_details = interpreter.get_tensor_details()[tensor_index]
    quantization = tensor_details.get('quantization_parameters', {})
    
    scale = quantization.get('scales', [1.0])
    zero_point = quantization.get('zero_points', [0])
    
    # Handle per-channel vs per-tensor quantization
    if len(scale) == 1:
        return {
            'scale': scale[0],
            'zero_point': zero_point[0],
            'per_channel': False
        }
    else:
        return {
            'scale': np.array(scale),
            'zero_point': np.array(zero_point),
            'per_channel': True
        }


def get_tensor_data(interpreter: tf.lite.Interpreter, tensor_index: int) -> np.ndarray:
    """
    Extract tensor data as numpy array.
    
    Args:
        interpreter: TFLite interpreter
        tensor_index: Index of the tensor
        
    Returns:
        Numpy array with tensor data
    """
    tensor = interpreter.tensor(tensor_index)()
    return np.array(tensor)


def run_inference(interpreter: tf.lite.Interpreter, input_data: np.ndarray) -> np.ndarray:
    """
    Run inference on TFLite model.
    
    Args:
        interpreter: TFLite interpreter
        input_data: Input data as numpy array
        
    Returns:
        Output data as numpy array
    """
    # Get input tensor index and expected shape
    input_details = interpreter.get_input_details()
    input_index = input_details[0]['index']
    expected_shape = input_details[0]['shape']
    
    # Handle batch size mismatch: TFLite may quantize with batch=1 fixed
    # even if the descriptor has batch>1
    input_shape = input_data.shape
    if input_shape != tuple(expected_shape):
        # If batch dimension differs, process batch by batch
        if len(input_shape) == len(expected_shape) and input_shape[0] != expected_shape[0]:
            batch_size = input_shape[0]
            expected_batch = expected_shape[0]
            
            if expected_batch == 1 and batch_size > 1:
                # Process each batch item separately and concatenate
                outputs = []
                for i in range(batch_size):
                    batch_input = input_data[i:i+1]  # Keep batch dimension as 1
                    interpreter.set_tensor(input_index, batch_input)
                    interpreter.invoke()
                    output_details = interpreter.get_output_details()
                    output_index = output_details[0]['index']
                    batch_output = interpreter.get_tensor(output_index)
                    outputs.append(batch_output)
                return np.concatenate(outputs, axis=0)
            else:
                # Reshape to match expected shape
                input_data = input_data.reshape(expected_shape)
        else:
            # Reshape to match expected shape
            input_data = input_data.reshape(expected_shape)
    
    # Set input tensor
    interpreter.set_tensor(input_index, input_data)
    
    # Run inference
    interpreter.invoke()
    
    # Get output tensor
    output_details = interpreter.get_output_details()
    output_index = output_details[0]['index']
    output_data = interpreter.get_tensor(output_index)
    
    return np.array(output_data)


def calculate_multiplier_shift(scale: float) -> Tuple[int, int]:
    """
    Calculate multiplier and shift from quantization scale.
    
    This converts a floating-point scale to integer multiplier and shift
    for CMSIS-NN quantization format.
    
    - Uses math.frexp() to decompose scale
    - Uses round-half-up rounding (math.floor(fraction * (1 << 31) + 0.5))
    - Formula: scale ≈ multiplier / (2^(31 - shift))
    
    Args:
        scale: Quantization scale (float)
        
    Returns:
        Tuple of (multiplier, shift) where:
        - multiplier is in Q31 format (0 to 2^31-1)
        - shift is the exponent from frexp
        - scale ≈ multiplier / (2^(31 - shift))
    """
    import math
    
    if scale == 0.0:
        return 0, 0
    
    # Decompose scale into fraction and exponent such that:
    #   scale = fraction * 2^exponent,  with fraction in [0.5, 1)
    fraction, exponent = math.frexp(scale)
    
    # Convert fraction to a Q31 fixed-point value.
    # Use round-half-up instead of banker's rounding:
    quantized_multiplier = int(math.floor(fraction * (1 << 31) + 0.5))
    
    # Handle the corner-case where rounding might push the value to 1<<31.
    if quantized_multiplier == (1 << 31):
        quantized_multiplier //= 2
        exponent += 1
    
    shift = exponent
    
    return quantized_multiplier, shift


def reduce_multiplier_q31_to_q15(multiplier: int) -> int:
    """Reduce a multiplier value for int16 quantization.

    Certain CMSIS functions for int16 quantized models require the
    multiplier to be reduced from Q31 format to Q15 format. This function
    performs that reduction.
    
    Args:
        multiplier (int): The original multiplier value in Q31 format.

    Returns:
        int: The reduced multiplier value in Q15 format.
    """
    if multiplier < 0x7FFF0000:
        return (multiplier + (1 << 15)) >> 16
    return 0x7FFF


def calculate_per_channel_multiplier_shift(scales: np.ndarray, reduce_to_q15: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate multiplier and shift arrays for per-channel quantization.
    
    Args:
        scales: Array of quantization scales (one per channel)
        reduce_to_q15: If True, reduce multipliers from Q31 to Q15 (for S16)
        
    Returns:
        Tuple of (multiplier_array, shift_array)
    """
    multipliers = []
    shifts = []
    
    for scale in scales:
        mult, shift = calculate_multiplier_shift(float(scale))
        if reduce_to_q15:
            mult = reduce_multiplier_q31_to_q15(mult)
        multipliers.append(mult)
        shifts.append(shift)
    
    return np.array(multipliers, dtype=np.int32), np.array(shifts, dtype=np.int32)


def extract_weights_biases(interpreter: tf.lite.Interpreter) -> Dict[str, Any]:
    """
    Extract weights and biases from TFLite model.
    
    Args:
        interpreter: TFLite interpreter
        
    Returns:
        Dictionary with 'weights' and 'biases' keys
    """
    tensor_details = interpreter.get_tensor_details()
    
    weights = None
    biases = None
    
    input_index = interpreter.get_input_details()[0]['index']
    output_index = interpreter.get_output_details()[0]['index']
    input_shape = interpreter.get_input_details()[0]['shape']
    
    # Find weight and bias tensors
    # For Conv2D: weights are typically 4D (OHWI), biases are 1D
    # For DepthwiseConv2D: weights are 4D (HWIM format: [H, W, I, M])
    for tensor in tensor_details:
        tensor_idx = tensor['index']
        
        # Skip input and output tensors
        if tensor_idx == input_index or tensor_idx == output_index:
            continue
        
        # Get tensor data
        try:
            tensor_data = get_tensor_data(interpreter, tensor_idx)
        except (ValueError, RuntimeError):
            # Skip tensors that don't have data (e.g. intermediate tensors)
            continue
            
        tensor_shape = tensor_data.shape
        
        # Weight tensor: 
        # - 4D for Conv2D/DepthwiseConv2D
        # - 2D for FullyConnected
        # - 3D for PReLU/others
        # Generally, any multi-dimensional tensor that is not the input is a weight candidate
        if weights is None:
            if len(tensor_shape) > 1:
                # Skip if this tensor has the same shape as input (likely the input tensor itself)
                if tuple(tensor_shape) != tuple(input_shape):
                    weights = tensor_data
                    
        # Bias tensor: 1D
        if biases is None and len(tensor_shape) == 1:
            biases = tensor_data
    
    return {
        'weights': weights,
        'biases': biases
    }

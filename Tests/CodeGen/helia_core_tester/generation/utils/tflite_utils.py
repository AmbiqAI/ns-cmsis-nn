"""
Quantization utilities for CMSIS-NN.
These functions convert quantization scales to multiplier/shift format.
"""

import numpy as np
from typing import Dict, Tuple, Any


def scalar_scale_zp(quant_dict: Dict[str, Any]) -> Tuple[float, int]:
    """
    Extract scalar scale and zero_point from a quantization dict (per-tensor or per-channel).
    For per-channel, uses the first element.
    """
    scale = quant_dict.get("scale", 1.0)
    zp = quant_dict.get("zero_point", 0)
    if isinstance(scale, (list, np.ndarray)):
        scale = float(scale[0])
    if isinstance(zp, (list, np.ndarray)):
        zp = int(zp[0])
    return float(scale), int(zp)


def activation_bounds(activation_dtype: str) -> Tuple[int, int]:
    """Return (min, max) int bounds for activation_dtype (e.g. S8, S16)."""
    activation_dtype = str(activation_dtype).upper()
    if activation_dtype == "S8":
        return -128, 127
    if activation_dtype == "S16":
        return -32768, 32767
    return -128, 127


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


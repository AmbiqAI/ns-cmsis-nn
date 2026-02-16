"""
Template context builder for Jinja templates.
Handles dimension conversions, quantization parameters, and array formatting.
"""

import numpy as np
from typing import Dict, Any, List, Tuple


class TemplateContextBuilder:
    """
    Builds context dictionaries for Jinja templates.
    Handles dimension conversions, quantization parameters, and array formatting.
    """
    
    @staticmethod
    def nhwc_to_cmsis_dims(shape: Tuple[int, ...]) -> Dict[str, int]:
        """
        Convert NHWC shape to CMSIS-NN dims format.
        
        Args:
            shape: Shape tuple in NHWC format
            
        Returns:
            Dictionary with n, h, w, c keys
        """
        if len(shape) == 4:
            return {
                'n': int(shape[0]),
                'h': int(shape[1]),
                'w': int(shape[2]),
                'c': int(shape[3])
            }
        elif len(shape) == 3:
            # HWC format
            return {
                'n': 1,
                'h': int(shape[0]),
                'w': int(shape[1]),
                'c': int(shape[2])
            }
        elif len(shape) == 2:
            # HW format
            return {
                'n': 1,
                'h': int(shape[0]),
                'w': int(shape[1]),
                'c': 1
            }
        elif len(shape) == 1:
            # C format
            return {
                'n': 1,
                'h': 1,
                'w': 1,
                'c': int(shape[0])
            }
        else:
            raise ValueError(f"Unsupported shape length: {len(shape)}")
    
    @staticmethod
    def format_array_as_c_literal(arr: np.ndarray, indent: int = 4, max_per_line: int = 16) -> str:
        """
        Format numpy array as C array literal.
        """
        if arr is None or arr.size == 0:
            return ""

        flat = arr.flatten()

        lines = []
        current_line = []

        for i, val in enumerate(flat):
            # Format value based on dtype
            if arr.dtype == np.int8 or arr.dtype == np.uint8:
                val_str = str(int(val))
            elif arr.dtype == np.int16 or arr.dtype == np.uint16:
                val_str = str(int(val))
            elif arr.dtype == np.int32 or arr.dtype == np.uint32:
                val_str = str(int(val))
            elif arr.dtype == np.int64 or arr.dtype == np.uint64:
                val_str = str(int(val))
            elif arr.dtype == np.float32 or arr.dtype == np.float64:
                val_str = f"{float(val):.6f}f"
            else:
                val_str = str(val)

            current_line.append(val_str)

            if len(current_line) >= max_per_line:
                lines.append(" " * indent + ", ".join(current_line) + ",")
                current_line = []

        if current_line:
            lines.append(" " * indent + ", ".join(current_line))

        return "\n".join(lines)

    
    @staticmethod
    def build_conv_params(
        desc: Dict[str, Any],
        input_shape: Tuple[int, ...],
        kernel_shape: Tuple[int, int],
        output_shape: Tuple[int, ...],
        input_quant: Dict[str, Any],
        output_quant: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build convolution parameters for CMSIS-NN.

        Notes:
        - input_shape/output_shape are expected NHWC (TFLite).
        - kernel_shape is (kernel_h, kernel_w).
        - For padding="same", compute padding from (in, out, stride, dilation, kernel) the same way TFLite does
        (symmetric "pad_before" used as CMSIS top/left padding).
        """
        strides = desc.get('strides', [1, 1])
        if isinstance(strides, (int, float)):
            stride_h = stride_w = int(strides)
        else:
            stride_h = int(strides[0])
            stride_w = int(strides[1])

        padding = desc.get('padding', 'valid')
        padding = 'valid' if padding is None else str(padding).lower()

        dilation = desc.get('dilation', [1, 1])
        if isinstance(dilation, (int, float)):
            dil_h = dil_w = int(dilation)
        else:
            dil_h = int(dilation[0])
            dil_w = int(dilation[1])

        kh, kw = int(kernel_shape[0]), int(kernel_shape[1])
        eff_kh = (kh - 1) * dil_h + 1
        eff_kw = (kw - 1) * dil_w + 1

        # NHWC -> H,W extraction
        if len(input_shape) == 4:
            in_h, in_w = int(input_shape[1]), int(input_shape[2])
        elif len(input_shape) == 3:
            in_h, in_w = int(input_shape[0]), int(input_shape[1])
        else:
            raise ValueError(f"Unsupported Conv2D input_shape rank: {len(input_shape)}")

        if len(output_shape) == 4:
            out_h, out_w = int(output_shape[1]), int(output_shape[2])
        elif len(output_shape) == 3:
            out_h, out_w = int(output_shape[0]), int(output_shape[1])
        else:
            raise ValueError(f"Unsupported Conv2D output_shape rank: {len(output_shape)}")

        if padding == 'same':
            pad_total_h = max((out_h - 1) * stride_h + eff_kh - in_h, 0)
            pad_total_w = max((out_w - 1) * stride_w + eff_kw - in_w, 0)
            pad_h = pad_total_h // 2
            pad_w = pad_total_w // 2
        else:  # valid
            pad_h = 0
            pad_w = 0

        # Activation clamp defaults depend on activation dtype
        act_dtype = str(desc.get('activation_dtype', 'S8')).upper()
        if act_dtype == 'S16':
            activation_min = int(desc.get('activation_min', -32768))
            activation_max = int(desc.get('activation_max', 32767))
        else:
            activation_min = int(desc.get('activation_min', -128))
            activation_max = int(desc.get('activation_max', 127))

        in_zp = int(input_quant.get('zero_point', 0))
        out_zp = int(output_quant.get('zero_point', 0))

        return {
            # CMSIS-NN convention: input_offset = -input_zero_point
            'input_offset': int(-in_zp),
            'output_offset': int(out_zp),
            'stride_h': int(stride_h),
            'stride_w': int(stride_w),
            'dilation_h': int(dil_h),
            'dilation_w': int(dil_w),
            'pad_h': int(pad_h),
            'pad_w': int(pad_w),
            'activation_min': int(activation_min),
            'activation_max': int(activation_max),
        }
    
    @staticmethod
    def build_dw_conv_params(
        desc: Dict[str, Any],
        input_shape: Tuple[int, ...],
        kernel_shape: Tuple[int, int],
        output_shape: Tuple[int, ...],
        input_quant: Dict[str, Any],
        output_quant: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build depthwise convolution parameters for CMSIS-NN.
        
        Notes:
        - input_shape/output_shape are expected NHWC (TFLite).
        - kernel_shape is (kernel_h, kernel_w).
        - For padding="same", compute padding from (in, out, stride, dilation, kernel) the same way TFLite does
        (symmetric "pad_before" used as CMSIS top/left padding).
        - ch_mult is the depth multiplier (output_channels / input_channels).
        """
        strides = desc.get('strides', [1, 1])
        if isinstance(strides, (int, float)):
            stride_h = stride_w = int(strides)
        else:
            stride_h = int(strides[0])
            stride_w = int(strides[1])
        
        padding = desc.get('padding', 'valid')
        padding = 'valid' if padding is None else str(padding).lower()
        
        dilation = desc.get('dilation', [1, 1])
        if isinstance(dilation, (int, float)):
            dil_h = dil_w = int(dilation)
        else:
            dil_h = int(dilation[0])
            dil_w = int(dilation[1])
        
        kh, kw = int(kernel_shape[0]), int(kernel_shape[1])
        eff_kh = (kh - 1) * dil_h + 1
        eff_kw = (kw - 1) * dil_w + 1
        
        # NHWC -> H,W extraction
        if len(input_shape) == 4:
            in_h, in_w = int(input_shape[1]), int(input_shape[2])
            in_c = int(input_shape[3])
        elif len(input_shape) == 3:
            in_h, in_w = int(input_shape[0]), int(input_shape[1])
            in_c = int(input_shape[2])
        else:
            raise ValueError(f"Unsupported DepthwiseConv2D input_shape rank: {len(input_shape)}")
        
        if len(output_shape) == 4:
            out_h, out_w = int(output_shape[1]), int(output_shape[2])
            out_c = int(output_shape[3])
        elif len(output_shape) == 3:
            out_h, out_w = int(output_shape[0]), int(output_shape[1])
            out_c = int(output_shape[2])
        else:
            raise ValueError(f"Unsupported DepthwiseConv2D output_shape rank: {len(output_shape)}")
        
        if padding == 'same':
            pad_total_h = max((out_h - 1) * stride_h + eff_kh - in_h, 0)
            pad_total_w = max((out_w - 1) * stride_w + eff_kw - in_w, 0)
            pad_h = pad_total_h // 2
            pad_w = pad_total_w // 2
        else:  # valid
            pad_h = 0
            pad_w = 0
        
        # Activation clamp defaults depend on activation dtype
        act_dtype = str(desc.get('activation_dtype', 'S8')).upper()
        if act_dtype == 'S16':
            activation_min = int(desc.get('activation_min', -32768))
            activation_max = int(desc.get('activation_max', 32767))
        else:
            activation_min = int(desc.get('activation_min', -128))
            activation_max = int(desc.get('activation_max', 127))
        
        in_zp = int(input_quant.get('zero_point', 0))
        out_zp = int(output_quant.get('zero_point', 0))
        
        # Calculate channel multiplier: output_channels / input_channels
        ch_mult = out_c // in_c if in_c > 0 else 1
        depth_multiplier = desc.get('depth_multiplier', ch_mult)
        ch_mult = int(depth_multiplier)
        
        return {
            # CMSIS-NN convention: input_offset = -input_zero_point
            'input_offset': int(-in_zp),
            'output_offset': int(out_zp),
            'ch_mult': int(ch_mult),
            'stride_h': int(stride_h),
            'stride_w': int(stride_w),
            'dilation_h': int(dil_h),
            'dilation_w': int(dil_w),
            'pad_h': int(pad_h),
            'pad_w': int(pad_w),
            'activation_min': int(activation_min),
            'activation_max': int(activation_max),
        }

    
    @staticmethod
    def build_quant_params(quant_params: Dict[str, Any], per_channel: bool = False) -> Dict[str, Any]:
        """
        Build quantization parameters for CMSIS-NN.
        
        Args:
            quant_params: Quantization parameters dictionary
            per_channel: Whether quantization is per-channel
            
        Returns:
            Dictionary with multiplier and shift arrays
        """
        # Import here to avoid circular dependency
        from .tflite_utils import calculate_multiplier_shift, calculate_per_channel_multiplier_shift
        
        if per_channel and isinstance(quant_params.get('scale'), np.ndarray):
            scales = quant_params['scale']
            multipliers, shifts = calculate_per_channel_multiplier_shift(scales)
            
            return {
                'multiplier_array': TemplateContextBuilder.format_array_as_c_literal(multipliers),
                'shift_array': TemplateContextBuilder.format_array_as_c_literal(shifts),
                'per_channel': True
            }
        else:
            scale = quant_params.get('scale', 1.0)
            # Handle numpy scalars (numpy.float32, numpy.float64, etc.)
            if isinstance(scale, np.ndarray):
                scale = float(scale.item() if scale.size == 1 else scale[0])
            elif isinstance(scale, (list, tuple)):
                scale = float(scale[0] if len(scale) > 0 else 1.0)
            else:
                # Convert numpy scalars to Python float
                scale = float(scale)
            
            multiplier, shift = calculate_multiplier_shift(scale)
            
            return {
                'multiplier': multiplier,
                'shift': shift,
                'per_channel': False
            }
    
    @staticmethod
    def calculate_depthwise_buffer_size_max(input_dims: Dict[str, int], 
                                            filter_dims: Dict[str, int],
                                            output_dims: Dict[str, int],
                                            output_dtype: str = 'S8') -> int:
        """
        Calculate a conservative upper bound for depthwise convolution buffer size.
        
        This matches CMSIS-NN's depthwise buffer size calculation:
        - MVE: 4 * CH_IN_BLOCK_MVE * filter_w * filter_h (where CH_IN_BLOCK_MVE = 16)
        - DSP: input_c * filter_w * filter_h * sizeof(int16_t)
        
        For depthwise conv, we use input_dims->c (input channels), not filter_dims->c.
        """
        input_c = input_dims['c']
        filter_w = filter_dims['w']
        filter_h = filter_dims['h']
        
        if output_dtype == 'S16':
            # S16 depthwise buffer size
            # From arm_depthwise_conv_get_buffer_sizes_s16.c:
            # MVE: 4 * input_dims->c * filter_dims->w * filter_dims->h * sizeof(int16_t) + 8
            # DSP: input_dims->c * filter_dims->w * filter_dims->h * sizeof(int16_t)
            buffer_size_mve = 4 * input_c * filter_w * filter_h * 2 + 8  # sizeof(int16_t) = 2, +8 for worst case
            buffer_size_dsp = input_c * filter_w * filter_h * 2  # sizeof(int16_t) = 2
            result = max(buffer_size_mve, buffer_size_dsp)
        else:
            # S8 depthwise buffer size (default)
            # MVE: 4 * CH_IN_BLOCK_MVE * filter_w * filter_h where CH_IN_BLOCK_MVE = 124
            # From arm_depthwise_conv_s8_opt_get_buffer_size_mve:
            # return (4 * CH_IN_BLOCK_MVE * filter_dims->w * filter_dims->h) * (int32_t)sizeof(int8_t);
            CH_IN_BLOCK_MVE = 124  # From arm_nnsupportfunctions.h
            buffer_size_mve = 4 * CH_IN_BLOCK_MVE * filter_w * filter_h  # sizeof(int8_t) = 1
            # DSP: input_c * filter_w * filter_h * sizeof(int16_t)
            # From arm_depthwise_conv_s8_opt_get_buffer_size_dsp:
            # return (input_dims->c * filter_dims->w * filter_dims->h) * sizeof(int16_t);
            buffer_size_dsp = input_c * filter_w * filter_h * 2  # sizeof(int16_t) = 2
            result = max(buffer_size_mve, buffer_size_dsp)
        
        return result
    
    @staticmethod
    def calculate_buffer_size_max(input_dims: Dict[str, int], 
                                  filter_dims: Dict[str, int],
                                  output_dims: Dict[str, int],
                                  output_dtype: str = 'S8') -> int:
        """
        Calculate a conservative upper bound for convolution buffer size.
        
        This uses the maximum of MVE and DSP buffer size calculations to ensure
        we have enough space regardless of which implementation is used.
        
        For S8 (int8_t):
          MVE: 4 * ceil((input_c * filter_w * filter_h) / 16) * 16
          DSP: 2 * ceil((filter_w * filter_h * input_c) / 4) * 4 * sizeof(int16_t)
        
        For S16 (int16_t):
          MVE: 4 * ceil((input_c * filter_w * filter_h) / 8) * 8 * sizeof(int16_t)
          DSP: 2 * input_c * filter_w * filter_h * sizeof(int16_t)
        """
        input_c = input_dims['c']
        filter_w = filter_dims['w']
        filter_h = filter_dims['h']
        
        if output_dtype == 'S16':
            # S16 buffer size calculation
            # MVE: 4 * ceil((input_c * filter_w * filter_h) / 8) * 8 * sizeof(int16_t)
            col_length_mve = input_c * filter_w * filter_h
            col_length_mve = (col_length_mve + 7) // 8
            buffer_size_mve = 4 * col_length_mve * 8 * 2  # sizeof(int16_t) = 2
            
            # DSP: 2 * input_c * filter_w * filter_h * sizeof(int16_t)
            buffer_size_dsp = 2 * input_c * filter_w * filter_h * 2  # sizeof(int16_t) = 2
            
            # Return the maximum to be safe
            return max(buffer_size_mve, buffer_size_dsp)
        else:
            # S8 buffer size calculation (default)
            # MVE buffer size calculation
            col_length_mve = input_c * filter_w * filter_h
            col_length_mve = (col_length_mve + 15) // 16
            buffer_size_mve = 4 * col_length_mve * 16  # sizeof(int8_t) = 1
            
            # DSP buffer size calculation
            rhs_cols = filter_w * filter_h * input_c
            remainder = rhs_cols % 4
            aligned_rhs_cols = rhs_cols + (4 - remainder) if remainder != 0 else rhs_cols
            buffer_size_dsp = 2 * aligned_rhs_cols * 2  # sizeof(int16_t) = 2
            
            result = max(buffer_size_mve, buffer_size_dsp)
            
            # Return the maximum to be safe
            return result
    
    @staticmethod
    def build_transpose_conv_params(
        desc: Dict[str, Any],
        input_shape: Tuple[int, ...],
        kernel_shape: Tuple[int, int],
        output_shape: Tuple[int, ...],
        input_quant: Dict[str, Any],
        output_quant: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build transpose convolution parameters for CMSIS-NN.
        
        Notes:
        - input_shape/output_shape are expected NHWC (TFLite).
        - kernel_shape is (kernel_h, kernel_w).
        - TransposeConv uses padding_offsets which are different from regular conv padding.
        """
        strides = desc.get('strides', [1, 1])
        if isinstance(strides, (int, float)):
            stride_h = stride_w = int(strides)
        else:
            stride_h = int(strides[0])
            stride_w = int(strides[1])
        
        padding = desc.get('padding', 'valid')
        padding = 'valid' if padding is None else str(padding).lower()
        
        dilation = desc.get('dilation', [1, 1])
        if isinstance(dilation, (int, float)):
            dil_h = dil_w = int(dilation)
        else:
            dil_h = int(dilation[0])
            dil_w = int(dilation[1])
        
        kh, kw = int(kernel_shape[0]), int(kernel_shape[1])
        
        # NHWC -> H,W extraction
        if len(input_shape) == 4:
            in_h, in_w = int(input_shape[1]), int(input_shape[2])
        elif len(input_shape) == 3:
            in_h, in_w = int(input_shape[0]), int(input_shape[1])
        else:
            raise ValueError(f"Unsupported TransposeConv input_shape rank: {len(input_shape)}")
        
        if len(output_shape) == 4:
            out_h, out_w = int(output_shape[1]), int(output_shape[2])
        elif len(output_shape) == 3:
            out_h, out_w = int(output_shape[0]), int(output_shape[1])
        else:
            raise ValueError(f"Unsupported TransposeConv output_shape rank: {len(output_shape)}")
        
        # TransposeConv padding calculation
        # For 'same' padding in transpose conv, we need to calculate padding differently
        if padding == 'same':
            # Calculate total padding needed
            pad_total_h = max((in_h - 1) * stride_h + kh - out_h, 0)
            pad_total_w = max((in_w - 1) * stride_w + kw - out_w, 0)
            pad_h = pad_total_h // 2
            pad_w = pad_total_w // 2
            # Padding offsets are the remainder
            pad_offset_h = pad_total_h % 2
            pad_offset_w = pad_total_w % 2
        else:  # valid
            pad_h = 0
            pad_w = 0
            pad_offset_h = 0
            pad_offset_w = 0
        
        # Activation clamp defaults depend on activation dtype
        act_dtype = str(desc.get('activation_dtype', 'S8')).upper()
        if act_dtype == 'S16':
            activation_min = int(desc.get('activation_min', -32768))
            activation_max = int(desc.get('activation_max', 32767))
        else:
            activation_min = int(desc.get('activation_min', -128))
            activation_max = int(desc.get('activation_max', 127))
        
        in_zp = int(input_quant.get('zero_point', 0))
        out_zp = int(output_quant.get('zero_point', 0))
        
        return {
            'input_offset': int(-in_zp),
            'output_offset': int(out_zp),
            'stride_h': int(stride_h),
            'stride_w': int(stride_w),
            'dilation_h': int(dil_h),
            'dilation_w': int(dil_w),
            'pad_h': int(pad_h),
            'pad_w': int(pad_w),
            'pad_offset_h': int(pad_offset_h),
            'pad_offset_w': int(pad_offset_w),
            'activation_min': int(activation_min),
            'activation_max': int(activation_max),
        }
    
    @staticmethod
    def calculate_transpose_conv_buffer_size_max(input_dims: Dict[str, int], 
                                                 filter_dims: Dict[str, int],
                                                 output_dims: Dict[str, int],
                                                 output_dtype: str = 'S8') -> int:
        """
        Calculate a conservative upper bound for transpose convolution buffer size.
        
        TransposeConv requires two buffers:
        1. ctx buffer (from arm_transpose_conv_s8_get_buffer_size)
        2. output_ctx buffer (from arm_transpose_conv_s8_get_reverse_conv_buffer_size)
        
        Returns the maximum of both buffers.
        """
        # For transpose conv, we use a conservative estimate
        # The actual buffer size depends on the transpose_conv_params which we don't have here
        # So we use a conservative upper bound based on output dimensions
        output_h = output_dims['h']
        output_w = output_dims['w']
        output_c = output_dims['c']
        
        # output_ctx buffer size: output width * output height * output channel * 4
        output_ctx_size = output_w * output_h * output_c * 4
        
        # ctx buffer size: similar to regular conv but for transpose
        input_c = input_dims['c']
        filter_w = filter_dims['w']
        filter_h = filter_dims['h']
        
        if output_dtype == 'S16':
            # S16 buffer size (conservative estimate)
            buffer_size_mve = 4 * 8 * filter_w * filter_h * 2  # sizeof(int16_t) = 2
            buffer_size_dsp = 2 * input_c * filter_w * filter_h * 2
            ctx_size = max(buffer_size_mve, buffer_size_dsp)
        else:
            # S8 buffer size (default)
            col_length_mve = input_c * filter_w * filter_h
            col_length_mve = (col_length_mve + 15) // 16
            buffer_size_mve = 4 * col_length_mve * 16
            
            rhs_cols = filter_w * filter_h * input_c
            remainder = rhs_cols % 4
            aligned_rhs_cols = rhs_cols + (4 - remainder) if remainder != 0 else rhs_cols
            buffer_size_dsp = 2 * aligned_rhs_cols * 2
            ctx_size = max(buffer_size_mve, buffer_size_dsp)
        
        # Return maximum of ctx and output_ctx
        return max(ctx_size, output_ctx_size)
    
    @staticmethod
    def build_fc_params(
        desc: Dict[str, Any],
        input_quant: Dict[str, Any],
        weight_quant: Dict[str, Any],
        output_quant: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build fully connected parameters for CMSIS-NN.
        
        Notes:
        - Fully connected uses cmsis_nn_fc_params
        - Activation clamp defaults depend on activation dtype
        """
        # Activation clamp defaults depend on activation dtype
        act_dtype = str(desc.get('activation_dtype', 'S8')).upper()
        if act_dtype == 'S16':
            activation_min = int(desc.get('activation_min', -32768))
            activation_max = int(desc.get('activation_max', 32767))
        else:
            activation_min = int(desc.get('activation_min', -128))
            activation_max = int(desc.get('activation_max', 127))
        
        in_zp = input_quant.get('zero_point', 0)
        if isinstance(in_zp, (list, np.ndarray)):
            in_zp = int(in_zp[0])
        else:
            in_zp = int(in_zp)
        
        weight_zp = weight_quant.get('zero_point', 0)
        if isinstance(weight_zp, (list, np.ndarray)):
            weight_zp = int(weight_zp[0])
        else:
            weight_zp = int(weight_zp)
        
        out_zp = output_quant.get('zero_point', 0)
        if isinstance(out_zp, (list, np.ndarray)):
            out_zp = int(out_zp[0])
        else:
            out_zp = int(out_zp)
        
        # For S16 fully connected, CMSIS-NN requires offsets to be 0
        # (S16 quantization typically uses symmetric quantization with zero_point=0)
        if act_dtype == 'S16':
            input_offset = 0
            filter_offset = 0
            output_offset = 0
        else:
            # For S8, use zero points as offsets
            input_offset = int(-in_zp)
            filter_offset = int(-weight_zp)
            output_offset = int(out_zp)
        
        return {
            'input_offset': input_offset,
            'filter_offset': filter_offset,
            'output_offset': output_offset,
            'activation_min': int(activation_min),
            'activation_max': int(activation_max),
        }
    
    @staticmethod
    def calculate_fc_buffer_size_max(filter_dims: Dict[str, int],
                                     output_dtype: str = 'S8') -> int:
        """
        Calculate a conservative upper bound for fully connected buffer size.
        
        For S8, buffer size depends on filter_dims (col_dim * row_dim).
        For S16, buffer size is typically 0 or minimal.
        """
        # For fully connected, buffer size is typically small or 0
        # Use a conservative estimate based on filter dimensions
        col_dim = filter_dims.get('n', 1)  # input features
        row_dim = filter_dims.get('c', 1)  # output features
        
        if output_dtype == 'S16':
            # S16 typically doesn't need a buffer, but include small estimate
            return max(0, col_dim * 2)  # Conservative estimate
        else:
            # S8 buffer size: typically col_dim * sizeof(int32_t) for weight sum
            # But we use a conservative estimate
            return max(col_dim * 4, row_dim * 4)  # Conservative estimate
    
    @staticmethod
    def build_pool_params(
        desc: Dict[str, Any],
        input_shape: Tuple[int, ...],
        kernel_shape: Tuple[int, int],
        output_shape: Tuple[int, ...],
        output_quant: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build pooling parameters for CMSIS-NN.
        
        Notes:
        - input_shape/output_shape are expected NHWC (TFLite).
        - kernel_shape is (pool_h, pool_w).
        - Pooling uses cmsis_nn_pool_params (no input_offset/output_offset).
        """
        strides = desc.get('strides', [1, 1])
        if isinstance(strides, (int, float)):
            stride_h = stride_w = int(strides)
        else:
            stride_h = int(strides[0])
            stride_w = int(strides[1])
        
        padding = desc.get('padding', 'valid')
        padding = 'valid' if padding is None else str(padding).lower()
        
        pool_h, pool_w = int(kernel_shape[0]), int(kernel_shape[1])
        
        # NHWC -> H,W extraction
        if len(input_shape) == 4:
            in_h, in_w = int(input_shape[1]), int(input_shape[2])
        elif len(input_shape) == 3:
            in_h, in_w = int(input_shape[0]), int(input_shape[1])
        else:
            raise ValueError(f"Unsupported Pooling input_shape rank: {len(input_shape)}")
        
        if len(output_shape) == 4:
            out_h, out_w = int(output_shape[1]), int(output_shape[2])
        elif len(output_shape) == 3:
            out_h, out_w = int(output_shape[0]), int(output_shape[1])
        else:
            raise ValueError(f"Unsupported Pooling output_shape rank: {len(output_shape)}")
        
        # Pooling padding calculation
        if padding == 'same':
            pad_total_h = max((out_h - 1) * stride_h + pool_h - in_h, 0)
            pad_total_w = max((out_w - 1) * stride_w + pool_w - in_w, 0)
            pad_h = pad_total_h // 2
            pad_w = pad_total_w // 2
        else:  # valid
            pad_h = 0
            pad_w = 0
        
        # Activation clamp defaults depend on activation dtype
        act_dtype = str(desc.get('activation_dtype', 'S8')).upper()
        if act_dtype == 'S16':
            activation_min = int(desc.get('activation_min', -32768))
            activation_max = int(desc.get('activation_max', 32767))
        else:
            activation_min = int(desc.get('activation_min', -128))
            activation_max = int(desc.get('activation_max', 127))
        
        return {
            'stride_h': int(stride_h),
            'stride_w': int(stride_w),
            'pad_h': int(pad_h),
            'pad_w': int(pad_w),
            'activation_min': int(activation_min),
            'activation_max': int(activation_max),
        }
    
    @staticmethod
    def calculate_pooling_buffer_size_max(input_dims: Dict[str, int],
                                          output_dims: Dict[str, int],
                                          pooling_type: str = 'AVERAGE',
                                          output_dtype: str = 'S8') -> int:
        """
        Calculate maximum buffer size for pooling operations.
        
        Args:
            input_dims: Input dimensions dict with n, h, w, c
            output_dims: Output dimensions dict with n, h, w, c
            pooling_type: 'AVERAGE' or 'MAX'
            output_dtype: 'S8' or 'S16'
            
        Returns:
            Maximum buffer size in bytes
        """
        # Max pooling doesn't need a buffer
        if pooling_type == 'MAX':
            return 0
        
        # Average pooling buffer size calculation
        # MVE: 0
        # DSP: input_channels * sizeof(int32_t) = input_channels * 4
        # Default: 0
        # We return the maximum (DSP) to be safe
        input_c = input_dims.get('c', 0)
        buffer_size_dsp = input_c * 4  # ch_src * sizeof(int32_t)
        
        return buffer_size_dsp
    
    @staticmethod
    def get_dtype_c_type(dtype: str) -> str:
        """
        Convert dtype string to C type.
        
        Args:
            dtype: Dtype string (S8, S16, etc.)
            
        Returns:
            C type string
        """
        dtype_map = {
            'S8': 'int8_t',
            'S16': 'int16_t',
            'S32': 'int32_t',
            'U8': 'uint8_t',
            'U16': 'uint16_t',
            'U32': 'uint32_t',
        }
        return dtype_map.get(dtype, 'int8_t')

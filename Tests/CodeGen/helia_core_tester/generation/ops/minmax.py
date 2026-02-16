"""
Maximum and Minimum operations implementation for Helia-Core Tester,
"""

from typing import Callable, Iterable, List, Optional, Tuple, Dict, Any
import numpy as np
import tensorflow as tf
from pathlib import Path
from helia_core_tester.generation.ops.base import OperationBase


# ----------------------------
# Helper layers / utilities
# ----------------------------

class CustomMinMaxLayer(tf.keras.layers.Layer):
    """Custom layer wrapper for tf.minimum and tf.maximum operations."""
    def __init__(self, op_fn: Callable, **kwargs):
        super().__init__(**kwargs)
        self.op_fn = op_fn

    def call(self, inputs):
        x1, x2 = inputs
        return self.op_fn(x1, x2)

# This is a shared FakeQuant layer that forces identical calibration range for any
# tensors that pass through it. Use the SAME INSTANCE on both branches.
# This is a hack to get around the unnecessary requantization nodes that
# would otherwise be inserted by the TFLite converter. provided by GPT
class SharedFakeQuant(tf.keras.layers.Layer):
    """
    A shared FakeQuant layer that forces identical calibration range for any
    tensors that pass through it. Use the SAME INSTANCE on both branches.
    """
    def __init__(
        self,
        min_val: float = -1.0,
        max_val: float = 1.0,
        num_bits: int = 16,
        narrow_range: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # non-trainable vars; shared across calls (and both branches)
        self.min_v = tf.Variable(min_val, trainable=False, dtype=tf.float32, name="fq_min")
        self.max_v = tf.Variable(max_val, trainable=False, dtype=tf.float32, name="fq_max")
        self.num_bits = num_bits
        self.narrow_range = narrow_range

    def call(self, x):
        return tf.quantization.fake_quant_with_min_max_vars(
            x, self.min_v, self.max_v, num_bits=self.num_bits, narrow_range=self.narrow_range
        )


def _max_abs(x: np.ndarray) -> float:
    return float(np.max(np.abs(x))) if x.size else 0.0


def _scale_to_match_amplitude(x_src: np.ndarray, x_tgt: np.ndarray) -> np.ndarray:
    """
    Scale x_tgt so that max|x_tgt| == max|x_src|.
    Keeps distribution but aligns the envelope for better shared scale calibration.
    """
    a_src = _max_abs(x_src)
    a_tgt = _max_abs(x_tgt)
    if a_tgt == 0.0:
        return x_tgt  # nothing to scale
    if a_src == 0.0:
        return np.zeros_like(x_tgt, dtype=np.float32)
    return (x_tgt.astype(np.float32) * (a_src / a_tgt)).astype(np.float32)


def _broadcast_axes(shape_ref: Tuple[int, ...], shape_other: Tuple[int, ...]) -> Optional[Tuple[int, ...]]:
    """
    Return axes in shape_ref along which shape_other would be broadcast (i.e., where other==1 < ref).
    If shapes are not broadcast-compatible, return None.
    """
    if len(shape_ref) != len(shape_other):
        return None
    axes = []
    for i, (a, b) in enumerate(zip(shape_ref, shape_other)):
        if b == 1 and a > 1:
            axes.append(i)
        elif b != 1 and a != b:
            return None
    return tuple(axes)


class OpMinMax(OperationBase):
    """
    Maximum and Minimum operation implementation with optional forced shared scale.

    Expected keys in self.desc:
      - 'operator': "Minimum" or "Maximum"
      - 'input_1_shape': tuple/list of ints (including batch)
      - 'input_2_shape': tuple/list of ints (including batch)
      - 'activation_dtype': 'S8' (default) or 'S16'
      - 'force_shared_scale': bool (default False)
      - 'shared_fq_range': (min_val, max_val) for FakeQuant (default (-1.0, 1.0))
      - 'rep_samples': int, number of reps for calibration (default 200)

    Notes (INT16 / 16x8):
      - FSS makes both inputs flow through the same FQ layer => identical scales, zp=0.
      - Rep-dataset logic aligns input envelopes to avoid extra requant nodes.
    """

    def build_keras_model(self) -> tf.keras.Model:
        """Build Keras model for Maximum/Minimum operation."""
        tf.keras.backend.clear_session()

        # Choose op
        op_name = self.desc.get('operator', None)
        if op_name == "Minimum":
            op_fn = tf.minimum
        elif op_name == "Maximum":
            op_fn = tf.maximum
        else:
            raise ValueError(f"Unsupported operator: {op_name}")

        # Shapes (expect batch-first shapes as provided)
        input_1_shape = tuple(self.desc['input_1_shape'])
        input_2_shape = tuple(self.desc['input_2_shape'])

        # Keras inputs exclude the batch dim in the 'shape=' argument
        inp1 = tf.keras.Input(shape=input_1_shape[1:], dtype=tf.float32, name='input1')
        inp2 = tf.keras.Input(shape=input_2_shape[1:], dtype=tf.float32, name='input2')

        # Optional shared FakeQuant injection
        force_shared_scale = bool(self.desc.get('force_shared_scale', False))
        act_dtype = self.desc.get('activation_dtype', 'S8')
        fq_bits = 16 if act_dtype == 'S16' else 8
        fq_min, fq_max = tuple(self.desc.get('shared_fq_range', (-1.0, 1.0)))

        if force_shared_scale:
            fq = SharedFakeQuant(min_val=fq_min, max_val=fq_max, num_bits=fq_bits, name="shared_fq")
            x1 = fq(inp1)
            x2 = fq(inp2)
        else:
            x1, x2 = inp1, inp2

        out = CustomMinMaxLayer(op_fn, name=f"{op_name or 'MinMax'}")([x1, x2])
        model = tf.keras.Model(inputs=[inp1, inp2], outputs=out, name=f"Op{op_name or 'MinMax'}")
        return model

    def convert_to_tflite(self, model: tf.keras.Model, out_path: str, rep_seed: int) -> None:
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
        else:
            raise ValueError(f"Unsupported activation_dtype: {activation_dtype}")

        # Seed RNG for stable calibration
        self.rng = np.random.default_rng(rep_seed)

        # Representative dataset generation (tightened)
        rep_samples = int(self.desc.get('rep_samples', 200))
        fq_min, fq_max = tuple(self.desc.get('shared_fq_range', (-1.0, 1.0)))

        # Shapes
        s1: Tuple[int, ...] = tuple(self.desc.get('input_1_shape', ()))
        s2: Tuple[int, ...] = tuple(self.desc.get('input_2_shape', ()))

        def _gen_pair() -> Tuple[np.ndarray, np.ndarray]:
            """
            Generate (x1, x2) so both inputs see the same amplitude envelope.
            - If shapes equal: x2 = x1.copy()
            - If broadcastable: reduce x1 along broadcast axes to produce x2 with same max|.| per slice,
              then add random sign to ensure symmetric coverage.
            - Else: scale random x2 to match x1 amplitude.
            """
            x1 = self.rng.uniform(fq_min, fq_max, size=s1).astype(np.float32)

            if s1 == s2:
                x2 = x1.copy()
                return x1, x2

            b_axes_2 = _broadcast_axes(s1, s2)  # axes along which x2 would broadcast against x1
            if b_axes_2 is not None:
                # Take max-abs over broadcast axes to align envelopes, then randomize sign for symmetry
                m = np.max(np.abs(x1), axis=b_axes_2, keepdims=True).astype(np.float32)
                # shape of 'm' with keepdims=True should equal s2 exactly
                # random +/- 1 with same shape as s2
                signs = (self.rng.integers(0, 2, size=s2) * 2 - 1).astype(np.float32)
                x2 = (m * signs).astype(np.float32)
                return x1, x2

            # Not broadcast-compatible in that direction; just generate x2 and scale to match amplitude
            x2_raw = self.rng.uniform(fq_min, fq_max, size=s2).astype(np.float32)
            x2 = _scale_to_match_amplitude(x1, x2_raw)
            return x1, x2

        def representative_data_gen() -> Iterable[List[np.ndarray]]:
            for _ in range(rep_samples):
                # If a single 'input_shape' is ever used upstream, keep backward compat:
                if 'input_shape' in self.desc and not (s1 and s2):
                    x = self.rng.uniform(fq_min, fq_max, size=tuple(self.desc['input_shape'])).astype(np.float32)
                    yield [x]
                else:
                    x1, x2 = _gen_pair()
                    yield [x1, x2]

        converter.representative_dataset = representative_data_gen

        # Convert & save
        tflite_model = converter.convert()
        with open(out_path, 'wb') as f:
            f.write(tflite_model)
    
    def _select_cmsis_minmax_kernel(self) -> Dict[str, str]:
        """
        Select appropriate CMSIS-NN kernel function for MinMax operation.
        
        Returns:
            Dictionary with kernel_fn, input_c_type, output_c_type
        """
        activation_dtype = self.desc.get('activation_dtype', 'S8')
        op_name = self.desc.get('operator', 'Maximum')
        
        if activation_dtype == 'S8':
            if op_name == 'Minimum':
                kernel_fn = 'arm_minimum_s8'
            elif op_name == 'Maximum':
                kernel_fn = 'arm_maximum_s8'
            else:
                raise ValueError(f"Unsupported operator: {op_name}")
            return {
                'kernel_fn': kernel_fn,
                'input_c_type': 'int8_t',
                'output_c_type': 'int8_t'
            }
        elif activation_dtype == 'S16':
            if op_name == 'Minimum':
                kernel_fn = 'arm_minimum_s16'
            elif op_name == 'Maximum':
                kernel_fn = 'arm_maximum_s16'
            else:
                raise ValueError(f"Unsupported operator: {op_name}")
            return {
                'kernel_fn': kernel_fn,
                'input_c_type': 'int16_t',
                'output_c_type': 'int16_t'
            }
        else:
            raise NotImplementedError(f"Unsupported MinMax dtype: {activation_dtype}")
    
    def generate_c_files(self, output_dir: Path) -> None:
        """
        Generate C and H files from templates for MinMax operation.
        """
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder
        
        name = self.desc['name']
        tflite_path = output_dir / f"{name}.tflite"
        if not tflite_path.exists():
            raise FileNotFoundError(f"TFLite file not found: {tflite_path}")
        
        # Select CMSIS kernel + types
        kernel_info = self._select_cmsis_minmax_kernel()
        op_name = self.desc.get('operator', 'Maximum')
        
        # Load LiteRT model for shape and quantization extraction
        from helia_core_tester.generation.utils.litert_utils import get_operator_tensors_from_litert
        model, subgraph = self.load_litert_model(str(tflite_path))
        op_tensors = get_operator_tensors_from_litert(model, subgraph, 0)
        
        # Extract shapes from LiteRT (multi-input operator)
        input1_shape = op_tensors['inputs'][0]['shape']
        input2_shape = op_tensors['inputs'][1]['shape'] if len(op_tensors['inputs']) > 1 else input1_shape
        output_shape = op_tensors['outputs'][0]['shape']
        
        # Ensure shapes are tuples
        if input1_shape is not None:
            input1_shape = tuple(input1_shape)
        if input2_shape is not None:
            input2_shape = tuple(input2_shape)
        if output_shape is not None:
            output_shape = tuple(output_shape)
        
        # Extract quantization from LiteRT
        input1_quant = op_tensors['inputs'][0]['quantization']
        input2_quant = op_tensors['inputs'][1]['quantization'] if len(op_tensors['inputs']) > 1 else input1_quant
        
        input1_scale = input1_quant.get('scale', 1.0)
        input1_zp = input1_quant.get('zero_point', 0)
        input2_scale = input2_quant.get('scale', 1.0)
        input2_zp = input2_quant.get('zero_point', 0)
        
        # Handle per-channel quantization (convert to scalar)
        if isinstance(input1_scale, (list, np.ndarray)):
            input1_scale = float(input1_scale[0])
        if isinstance(input1_zp, (list, np.ndarray)):
            input1_zp = int(input1_zp[0])
        if isinstance(input2_scale, (list, np.ndarray)):
            input2_scale = float(input2_scale[0])
        if isinstance(input2_zp, (list, np.ndarray)):
            input2_zp = int(input2_zp[0])
        
        input1_scale = float(input1_scale)
        input1_zp = int(input1_zp)
        input2_scale = float(input2_scale)
        input2_zp = int(input2_zp)
        
        builder = TemplateContextBuilder()
        
        # Convert shapes to CMSIS dims
        input1_dims = builder.nhwc_to_cmsis_dims(input1_shape)
        input2_dims = builder.nhwc_to_cmsis_dims(input2_shape)
        output_dims = builder.nhwc_to_cmsis_dims(output_shape)
        
        # Generate input data and quantize
        rng_state = self.rng.__getstate__()
        self.rng = np.random.default_rng(self.seed)
        
        input1_data = self.rng.uniform(-1.0, 1.0, size=input1_shape).astype(np.float32)
        input2_data = self.rng.uniform(-1.0, 1.0, size=input2_shape).astype(np.float32)
        
        self.rng.__setstate__(rng_state)
        
        # Quantize inputs
        if kernel_info["input_c_type"] == "int8_t":
            np_in_dtype = np.int8
            qmin, qmax = -128, 127
        elif kernel_info["input_c_type"] == "int16_t":
            np_in_dtype = np.int16
            qmin, qmax = -32768, 32767
        else:
            raise ValueError(f"Unsupported input_c_type: {kernel_info['input_c_type']}")
        
        input1_q = np.round(input1_data / float(input1_scale) + float(input1_zp)).astype(np.int32)
        input1_q = np.clip(input1_q, qmin, qmax).astype(np_in_dtype)
        
        input2_q = np.round(input2_data / float(input2_scale) + float(input2_zp)).astype(np.int32)
        input2_q = np.clip(input2_q, qmin, qmax).astype(np_in_dtype)
        
        # Run inference using LiteRT interpreter
        interpreter = self.load_litert_interpreter(str(tflite_path))
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        interpreter.set_tensor(input_details[0]['index'], input1_q)
        interpreter.set_tensor(input_details[1]['index'], input2_q)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        output_data = np.array(output_data)
        
        # Format arrays
        input1_array_str = builder.format_array_as_c_literal(input1_q)
        input2_array_str = builder.format_array_as_c_literal(input2_q)
        expected_output_array_str = builder.format_array_as_c_literal(output_data)
        
        # Build template context
        context = {
            'name': name,
            'prefix': name,
            'input1_dims': input1_dims,
            'input2_dims': input2_dims,
            'output_dims': output_dims,
            'input1_data_array': input1_array_str,
            'input2_data_array': input2_array_str,
            'expected_output_array': expected_output_array_str,
            'input_dtype': kernel_info["input_c_type"],
            'output_dtype': kernel_info["output_c_type"],
            'kernel_fn': kernel_info["kernel_fn"],
            'operator': op_name,
        }
        
        # Render templates
        includes_api_dir = output_dir / "includes"
        includes_api_dir.mkdir(parents=True, exist_ok=True)
        
        h_content = self.render_template("minmax/minmax.h.j2", context)
        h_path = includes_api_dir / f"{name}_minmax.h"
        with open(h_path, 'w') as f:
            f.write(h_content)
        
        c_content = self.render_template("minmax/minmax.c.j2", context)
        c_path = output_dir / f"{name}_minmax.c"
        with open(c_path, 'w') as f:
            f.write(c_content)
        
        cmake_context = {
            'name': name,
            'operator': self.desc.get('operator', 'MinMax'),
            'operator_name': 'minmax'
        }
        cmake_content = self.render_template("common/CMakeLists.txt.j2", cmake_context)
        cmake_path = output_dir / "CMakeLists.txt"
        with open(cmake_path, 'w') as f:
            f.write(cmake_content)
        
        print(f"Generated C/H files and CMakeLists.txt for {name}")
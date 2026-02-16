"""
LiteRT single-op model builder utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

try:
    from ai_edge_litert import schema_py_generated as litert
    import flatbuffers
    LITERT_AVAILABLE = True
except ImportError:  # pragma: no cover - runtime dependency
    litert = None
    flatbuffers = None
    LITERT_AVAILABLE = False


_DTYPE_MAP = {
    "int8": litert.TensorType.INT8 if LITERT_AVAILABLE else None,
    "int16": litert.TensorType.INT16 if LITERT_AVAILABLE else None,
}


def _default_quant(tensor_type: int) -> Optional[Tuple[Sequence[float], Sequence[int]]]:
    if not LITERT_AVAILABLE:
        return None
    if tensor_type == litert.TensorType.INT8:
        return ([0.125], [0])
    if tensor_type == litert.TensorType.INT16:
        return ([1.0 / 32768.0], [0])
    return None


_TENSOR_TYPE_TO_NP = (
    {
        litert.TensorType.INT8: np.int8,
        litert.TensorType.INT16: np.int16,
        litert.TensorType.INT32: np.int32,
        litert.TensorType.UINT8: np.uint8,
        litert.TensorType.INT64: np.int64,
        litert.TensorType.FLOAT32: np.float32,
    }
    if LITERT_AVAILABLE
    else {}
)


@dataclass
class TensorSpec:
    name: str
    shape: Iterable[int]
    tensor_type: int
    is_input: bool = False
    is_output: bool = False
    quantization: Optional[Tuple[Sequence[float], Sequence[int]]] = None
    data: Optional[Sequence[int] | np.ndarray] = None


class LiteRtSingleOpBuilder:
    def __init__(self, op_name: str):
        if not LITERT_AVAILABLE:
            raise ImportError("ai_edge_litert is not available. Install it with: pip install ai-edge-litert")
        self.op_name = op_name
        self._tensors: list[TensorSpec] = []
        self._op_inputs: list[int] = []
        self._op_outputs: list[int] = []
        self._op_options = None
        self._op_options_type = None

    def add_tensor(self, spec: TensorSpec) -> int:
        idx = len(self._tensors)
        self._tensors.append(spec)
        return idx

    def add_operator(
        self,
        op_name: str,
        *,
        inputs: Sequence[int],
        outputs: Sequence[int],
        options,
        options_type,
    ) -> None:
        self.op_name = op_name
        self._op_inputs = list(inputs)
        self._op_outputs = list(outputs)
        self._op_options = options
        self._op_options_type = options_type

    def build(self) -> bytes:
        model = litert.ModelT()
        model.version = 3

        # Buffers: buffer[0] is empty per TFLite convention
        buffers: list[litert.BufferT] = []
        buffers.append(litert.BufferT())

        tensors: list[litert.TensorT] = []
        subgraph_inputs: list[int] = []
        subgraph_outputs: list[int] = []

        for idx, spec in enumerate(self._tensors):
            tensor = litert.TensorT()
            tensor.name = spec.name
            tensor.shape = list(int(v) for v in spec.shape)
            tensor.type = spec.tensor_type
            tensor.isVariable = False

            if spec.quantization is not None:
                scales, zero_points = spec.quantization
                q = litert.QuantizationParametersT()
                q.scale = list(float(v) for v in scales)
                q.zeroPoint = list(int(v) for v in zero_points)
                q.quantizedDimension = 0
                tensor.quantization = q

            if spec.data is not None:
                buf = litert.BufferT()
                np_dtype = _TENSOR_TYPE_TO_NP.get(spec.tensor_type, np.int8)
                data = np.array(spec.data, dtype=np_dtype).tobytes()
                buf.data = data
                buffers.append(buf)
                tensor.buffer = len(buffers) - 1
            else:
                tensor.buffer = 0

            tensors.append(tensor)

            if spec.is_input:
                subgraph_inputs.append(idx)
            if spec.is_output:
                subgraph_outputs.append(idx)

        opcode = litert.OperatorCodeT()
        builtin = getattr(litert.BuiltinOperator, self.op_name, None)
        if builtin is None:
            raise ValueError(f"Unsupported builtin op '{self.op_name}'")
        opcode.builtinCode = builtin

        op = litert.OperatorT()
        op.opcodeIndex = 0
        op.inputs = list(self._op_inputs)
        op.outputs = list(self._op_outputs)
        op.builtinOptionsType = self._op_options_type
        op.builtinOptions = self._op_options

        subgraph = litert.SubGraphT()
        subgraph.tensors = tensors
        subgraph.inputs = subgraph_inputs
        subgraph.outputs = subgraph_outputs
        subgraph.operators = [op]
        subgraph.name = "main"

        model.operatorCodes = [opcode]
        model.subgraphs = [subgraph]
        model.buffers = buffers

        builder = flatbuffers.Builder(1024)
        model_offset = model.Pack(builder)
        file_identifier = getattr(litert.Model, "FileIdentifier", lambda: b"TFL3")()
        builder.Finish(model_offset, file_identifier)
        return bytes(builder.Output())


def build_arg_op(
    op: str = "argmax",
    *,
    input_shape: Iterable[int] = (1, 4, 4, 3),
    axis: int = -1,
    dtype: str = "int8",
    golden_filename: str | None = None,
    work_dir: str | Path | None = None,
    seed: int = 0,
) -> bytes:
    op_lower = op.lower()
    if op_lower not in {"argmax", "argmin"}:
        raise ValueError(f"Unsupported arg operator '{op}'. Use 'argmax' or 'argmin'.")

    tensor_type = _DTYPE_MAP.get(dtype.lower())
    if tensor_type is None:
        raise ValueError(f"Unsupported dtype '{dtype}'.")

    input_shape = tuple(int(dim) for dim in input_shape)
    rank = len(input_shape)
    if rank == 0:
        raise ValueError("Input shape must have at least one dimension for ARG operators.")

    axis_norm = axis
    if axis_norm < 0:
        axis_norm += rank
    if axis_norm < 0 or axis_norm >= rank:
        raise ValueError(f"Axis {axis} out of range for rank {rank}.")

    output_shape = list(input_shape)
    del output_shape[axis_norm]
    if not output_shape:
        output_shape = [1]

    builder = LiteRtSingleOpBuilder(op_name="ARG_MAX" if op_lower == "argmax" else "ARG_MIN")

    input_tensor_idx = builder.add_tensor(
        TensorSpec(
            name="input",
            shape=input_shape,
            tensor_type=tensor_type,
            is_input=True,
            quantization=_default_quant(tensor_type),
        )
    )

    axis_tensor_idx = builder.add_tensor(
        TensorSpec(
            name="axis",
            shape=(1,),
            tensor_type=litert.TensorType.INT32,
            data=[axis_norm],
        )
    )

    output_tensor_idx = builder.add_tensor(
        TensorSpec(
            name="output",
            shape=tuple(output_shape),
            tensor_type=litert.TensorType.INT32,
            is_output=True,
        )
    )

    if op_lower == "argmax":
        options = litert.ArgMaxOptionsT()
        options.outputType = litert.TensorType.INT32
        options_type = litert.BuiltinOptions.ArgMaxOptions
        op_name = "ARG_MAX"
    else:
        options = litert.ArgMinOptionsT()
        options.outputType = litert.TensorType.INT32
        options_type = litert.BuiltinOptions.ArgMinOptions
        op_name = "ARG_MIN"

    builder.add_operator(
        op_name,
        inputs=[input_tensor_idx, axis_tensor_idx],
        outputs=[output_tensor_idx],
        options=options,
        options_type=options_type,
    )

    model_bytes = builder.build()

    if golden_filename is not None:
        if work_dir is None:
            raise ValueError("work_dir must be provided when golden_filename is set")
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        golden_path = work_dir / golden_filename

        rng = np.random.default_rng(seed)
        np_dtype = np.int8 if tensor_type == litert.TensorType.INT8 else np.int16
        lo, hi = np.iinfo(np_dtype).min, np.iinfo(np_dtype).max
        input_data = rng.integers(lo, hi + 1, size=input_shape, dtype=np_dtype)

        if op_lower == "argmax":
            output_data = np.argmax(input_data, axis=axis_norm).astype(np.int32)
        else:
            output_data = np.argmin(input_data, axis=axis_norm).astype(np.int32)

        np.savez(golden_path, input_0=input_data, output_0=output_data)

    return model_bytes

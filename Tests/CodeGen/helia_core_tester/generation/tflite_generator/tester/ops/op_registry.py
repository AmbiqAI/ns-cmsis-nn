"""Operation registry: operator name -> op class for TFLite generation."""

from .fully_connected import OpFullyConnected
from .conv2d import OpConv2D
from .dwconv import OpDepthwiseConv2D
from .matmul_batch import OpMatMul
from .elementwise import OpElementwise
from .add import OpAdd
from .sub import OpSub
from .mul import OpMul
from .minmax import OpMinMax
from .relu import OpRelu
from .relu6 import OpRelu6
from .leakyrelu import OpLeakyRelu
from .softmax import OpSoftmax
from .tanh import OpTanh
from .logistic import OpLogistic
from .hard_swish import OpHardSwish
from .prelu import OpPReLU
from .quantize import OpQuantize
from .dequantize import OpDequantize
from .pooling import OpPooling
from .transpose import OpTranspose
from .stridedslice import OpStridedSlice
from .pad import OpPad
from .lstm import OpLSTM
from .svdf import OpSVDF
from .mean import OpMean
from .reducemax import OpReduceMax
from .reducemin import OpReduceMin
from .argmax import OpArgMax
from .argmin import OpArgMin
from .fill import OpFill
from .zeros_like import OpZerosLike
from .reshape import OpReshape
from .shape import OpShape
from .slice import OpSlice
from .squeeze import OpSqueeze
from .space_to_depth import OpSpaceToDepth
from .depth_to_space import OpDepthToSpace
from .split import OpSplit
from .pack import OpPack
from .unpack import OpUnpack
from .concatenation import OpConcatenation
from .space_to_batch_nd import OpSpaceToBatchND
from .batch_to_space_nd import OpBatchToSpaceND
from .variable_update import OpVariableUpdate
from .transposeconv import OpTransposeConv
from .equal import OpEqual
from .not_equal import OpNotEqual
from .greater import OpGreater
from .greater_equal import OpGreaterEqual
from .less import OpLess
from .less_equal import OpLessEqual

OP_MAP = {
    "FullyConnected": OpFullyConnected,
    "Conv2D": OpConv2D,
    "DepthwiseConv2D": OpDepthwiseConv2D,
    "MatMul": OpMatMul,
    "Elementwise": OpElementwise,
    "Add": OpAdd,
    "Sub": OpSub,
    "Mul": OpMul,
    "Maximum": OpMinMax,
    "Minimum": OpMinMax,
    "Relu": OpRelu,
    "Relu6": OpRelu6,
    "LeakyRelu": OpLeakyRelu,
    "Softmax": OpSoftmax,
    "Tanh": OpTanh,
    "Logistic": OpLogistic,
    "HardSwish": OpHardSwish,
    "PReLU": OpPReLU,
    "Quantize": OpQuantize,
    "Dequantize": OpDequantize,
    "Pooling": OpPooling,
    "Transpose": OpTranspose,
    "StridedSlice": OpStridedSlice,
    "Pad": OpPad,
    "LSTM": OpLSTM,
    "SVDF": OpSVDF,
    "Mean": OpMean,
    "ReduceMax": OpReduceMax,
    "ReduceMin": OpReduceMin,
    "ArgMax": OpArgMax,
    "ArgMin": OpArgMin,
    "Fill": OpFill,
    "ZerosLike": OpZerosLike,
    "Reshape": OpReshape,
    "Shape": OpShape,
    "Slice": OpSlice,
    "Squeeze": OpSqueeze,
    "SpaceToDepth": OpSpaceToDepth,
    "DepthToSpace": OpDepthToSpace,
    "Split": OpSplit,
    "Pack": OpPack,
    "Unpack": OpUnpack,
    "Concatenation": OpConcatenation,
    "SpaceToBatchND": OpSpaceToBatchND,
    "BatchToSpaceND": OpBatchToSpaceND,
    "VariableUpdate": OpVariableUpdate,
    "TransposeConv": OpTransposeConv,
    "Equal": OpEqual,
    "NotEqual": OpNotEqual,
    "Greater": OpGreater,
    "GreaterEqual": OpGreaterEqual,
    "Less": OpLess,
    "LessEqual": OpLessEqual,
}
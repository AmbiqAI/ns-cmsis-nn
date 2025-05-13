local_src := $(wildcard $(subdirectory)/Source/ActivationFunctions/*.c)
local_src += $(wildcard $(subdirectory)/Source/BasicMathFunctions/*.c)
local_src += $(wildcard $(subdirectory)/Source/ConcatenationFunctions/*.c)
local_src += $(wildcard $(subdirectory)/Source/ConvolutionFunctions/*.c)
local_src += $(wildcard $(subdirectory)/Source/FullyConnectedFunctions/*.c)
local_src += $(wildcard $(subdirectory)/Source/LSTMFunctions/*.c)
local_src += $(wildcard $(subdirectory)/Source/NNSupportFunctions/*.c)
local_src += $(wildcard $(subdirectory)/Source/PadFunctions/*.c)
local_src += $(wildcard $(subdirectory)/Source/PoolingFunctions/*.c)
local_src += $(wildcard $(subdirectory)/Source/QuantizationFunctions/*.c)
local_src += $(wildcard $(subdirectory)/Source/ReshapeFunctions/*.c)
local_src += $(wildcard $(subdirectory)/Source/SoftmaxFunctions/*.c)
local_src += $(wildcard $(subdirectory)/Source/SVDFunctions/*.c)
local_src += $(wildcard $(subdirectory)/Source/TransposeFunctions/*.c)

includes_api += $(subdirectory)/Include
includes_api += $(subdirectory)/Include/Internal

local_bin := $(BINDIR)/$(subdirectory)
bindirs   += $(local_bin)
$(eval $(call make-library, $(local_bin)/ns-cmsis-nn.a, $(local_src)))

#include "config_data.h"
#include "input.h"
#include "input_bias.h"
#include "input_weights.h"
#if defined(CMSIS_NN_USE_SINGLE_ROUNDING)
#include "output_ref_single_rounding_data.h"
#else
#include "output.h"
#endif
#include "output_multiplier.h"
#include "output_shift.h"

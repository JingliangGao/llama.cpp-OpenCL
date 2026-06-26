#pragma once

#include "ponn.h"
#include "ggml-impl.h"

#include <vector>

PONN_DATA_TYPE_E ponn_utils_get_data_type(ggml_type type);
PONN_DATA_TYPE_E ponn_utils_get_inference_data_type(ggml_type type);
void ponn_utils_get_stride_div(ggml_type type, std::vector<int>& div);
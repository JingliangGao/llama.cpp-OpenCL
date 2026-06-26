#include "utils.h"

PONN_DATA_TYPE_E ponn_utils_get_data_type(ggml_type type) {
    (void)type;
    return PONN_DATA_FLOAT;
}

PONN_DATA_TYPE_E ponn_utils_get_inference_data_type(const ggml_type dtype) {
    PONN_DATA_TYPE_E ponnDataType = ponnGetInferenceDataType();
    PONN_DATA_TYPE_E srcDataType = ponn_utils_get_data_type(dtype);
    if (ponnDataType == PONN_DATA_HALF && srcDataType == PONN_DATA_FLOAT) {
        return PONN_DATA_HALF;
    }
    return srcDataType;
}

void ponn_utils_get_stride_div(ggml_type type, std::vector<int>& div) {
    (void)type;
    div.resize(4, 1);
}
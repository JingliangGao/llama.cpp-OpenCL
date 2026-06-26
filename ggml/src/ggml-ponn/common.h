#pragma once

#include "ponn.h"
#include "ggml-backend-impl.h"

#include <map>
#include <vector>

struct ggml_backend_ponn_extra {
    void* handle = nullptr;
    size_t offset = 0;
    size_t size = 0;
    std::vector<void*> extraPonnData;
};

struct ggml_backend_ponn_buffer_context {
    void* handle = nullptr;
    void* dev_ptr = nullptr;
    std::vector<ggml_backend_ponn_extra*> tensor_extras;
    ~ggml_backend_ponn_buffer_context() {
        if (handle) {
            ponnDirectFree(handle);
        }
    }
};

struct ggml_backend_ponn_context {
    bool init_done = false;
    int n_devices = 1;
    void* ponn_sin = nullptr;
    void* ponn_cos = nullptr;
    void* one_tensor = nullptr;
    size_t one_tensor_size = 0;
    ~ggml_backend_ponn_context() {
        if (one_tensor) {
            ponnFree(one_tensor);
            one_tensor = nullptr;
            one_tensor_size = 0;
        }
        if (ponn_sin) {
            ponnFree(ponn_sin);
            ponn_sin = nullptr;
        }
        if (ponn_cos) {
            ponnFree(ponn_cos);
            ponn_cos = nullptr;
        }
    }
    void* get_one_tensor(size_t size, PONN_DATA_TYPE_E ponn_datatype) {
        (void)ponn_datatype;
        if (size > one_tensor_size) {
            if (one_tensor) {
                ponnFree(one_tensor);
            }
            one_tensor = ponnMallocBuf(size);
            one_tensor_size = size;
        }
        return one_tensor;
    }
};
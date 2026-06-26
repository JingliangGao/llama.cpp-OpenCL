#pragma once

#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include <map>
#include <vector>

#define GGML_PONN_TRACE()

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    PONN_DATA_FLOAT,
    PONN_DATA_HALF,
} PONN_DATA_TYPE_E;

typedef enum {
    DEVICE_TO_DEVICE,
    HOST_TO_DEVICE,
    DEVICE_TO_HOST,
} PONN_MEMCPY_TYPE_E;

typedef void* PONN_MEM_H;

void ponnInit(void);
void ponnDeinit(void);

PONN_MEM_H ponnMallocBuf(size_t size);
PONN_MEM_H ponnDirectMalloc(size_t size);
void ponnFreeBuf(PONN_MEM_H ptr);
void ponnDirectFree(PONN_MEM_H ptr);
void ponnFree(PONN_MEM_H ptr);
void ponnFreeSubBuf(PONN_MEM_H ptr);

void ponnMemcpy(PONN_MEM_H dst, size_t dst_offset, PONN_MEM_H src, size_t src_offset, size_t size, PONN_MEMCPY_TYPE_E type);
void ponnMemcpyEx(PONN_MEM_H dst, PONN_DATA_TYPE_E dst_type, PONN_MEM_H src, PONN_DATA_TYPE_E src_type, size_t size, const std::vector<int>& dims, PONN_MEMCPY_TYPE_E type);
void ponnMemcpyNoContiguous(PONN_MEM_H input, PONN_MEM_H output, const std::vector<int>& inputDims, const std::vector<size_t>& inputStrides, const std::vector<int>& outputDims, const std::vector<size_t>& outputStrides, PONN_DATA_TYPE_E stype, PONN_DATA_TYPE_E dtype, size_t src_offset, size_t dst_offset);

void ponnPermute(PONN_MEM_H input, PONN_MEM_H output, PONN_DATA_TYPE_E dtype, const std::vector<int>& dims, const std::vector<int>& perm);

void ponnMulMatFp(PONN_MEM_H input0, PONN_MEM_H input1, PONN_MEM_H output, PONN_DATA_TYPE_E input0_type, PONN_DATA_TYPE_E input1_type, PONN_DATA_TYPE_E output_type, const std::vector<int>& input0Spatial, const std::vector<int>& input1Spatial, const std::vector<int>& outputSpatial, int batch, int n, int m, int k, int group, bool expanded, float alpha);
void ponnMulMatFpBias(PONN_MEM_H input0, PONN_MEM_H input1, PONN_MEM_H input2, PONN_MEM_H output, PONN_DATA_TYPE_E input0_type, PONN_DATA_TYPE_E input1_type, PONN_DATA_TYPE_E input2_type, PONN_DATA_TYPE_E output_type, const std::vector<int>& input0Spatial, const std::vector<int>& input1Spatial, const std::vector<int>& input2Spatial, const std::vector<int>& outputSpatial, int batch, int n, int m, int k, int group, float alpha);

void ponnRope(PONN_MEM_H input, PONN_MEM_H posId, PONN_MEM_H sinData, PONN_MEM_H cosData, PONN_MEM_H output, const std::vector<int>& inputDims, const std::vector<int>& posIdDims, const std::vector<int>& sinDims, const std::vector<int>& cosDims, const std::vector<int>& outputDims, int rotary_dim, bool is_neox, PONN_DATA_TYPE_E dtype);
void ponnRopeYarn(PONN_MEM_H input, PONN_MEM_H posId, PONN_MEM_H input2, PONN_MEM_H output, const std::vector<int>& inputDims, const std::vector<int>& posIdDims, const std::vector<int>& input2Dims, const std::vector<int>& outputDims, int rotary_dim, int n_ctx_orig, float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow, int rotary_dim2, bool is_neox, PONN_DATA_TYPE_E dtype);

PONN_DATA_TYPE_E ponnGetInferenceDataType(void);
size_t ponnGetMaxMemoryAllocateSize(void);
void ponnSyncStream(void* stream);
void* ponnGetStream(void);
void ponnClearBuffer(void);

PONN_MEM_H ponnPrepare(ggml_tensor* tensor);
void ponnFinish(ggml_tensor* tensor, PONN_MEM_H mem);

#ifdef __cplusplus
}
#endif
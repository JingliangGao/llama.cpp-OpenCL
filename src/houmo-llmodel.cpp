#include "houmo-llmodel.h"
#include "ggml.h"
#include "gguf.h"
#include "llama-impl.h"
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#ifndef _WIN32                                                                                              // JingliangGao 2026/06/27
#include <unistd.h>                                                                                         // JingliangGao 2026/06/27
#endif                                                                                                      // JingliangGao 2026/06/27
#ifdef ENABLE_HOUMO_DADAO
#include "tcim_c_api_loader.h"
#include "houmo-embedding-layer.h"
#include "json.hpp"
using json = nlohmann::ordered_json;

static tcim_api_table_t g_tcim_api;
static bool g_tcim_loaded = false;

static bool load_tcim_library() {
    if (g_tcim_loaded) return true;

    const char* lib_path = std::getenv("TCIM_LIBRARY_PATH");                                                   // JingliangGao 2026/06/27
    if (!lib_path) {                                                                                          // JingliangGao 2026/06/27
#ifdef _WIN32
        lib_path = "tcim_c_api_bridge.dll";
#else                                                                                                         // JingliangGao 2026/06/27
        if (access("/opt/system/lib/xpu/houmo/libtcim_runtime_c_api.so", F_OK) == 0) {                        // JingliangGao 2026/06/27
            lib_path = "/opt/system/lib/xpu/houmo/libtcim_runtime_c_api.so";                                  // JingliangGao 2026/06/27
        } else if (access("/usr/lib/xpu/houmo/libtcim_runtime_c_api.so", F_OK) == 0) {                        // JingliangGao 2026/06/27
            lib_path = "/usr/lib/xpu/houmo/libtcim_runtime_c_api.so";                                        // JingliangGao 2026/06/27
        } else {                                                                                              // JingliangGao 2026/06/27
            lib_path = "libtcim_runtime_c_api.so";                                                             // JingliangGao 2026/06/27
        }                                                                                                     // JingliangGao 2026/06/27
#endif
    }

    if (!tcim_load_library(lib_path, &g_tcim_api)) {
        LLAMA_LOG_ERROR("Failed to load TCIM library: %s\n", tcim_get_last_error());
        return false;
    }

    g_tcim_loaded = true;
    return true;
}

struct kv_cache_item {
    std::vector<tcim_tensor_t> tensors;
    std::vector<std::string> names;
    int seq_id = 0;
};

struct lora_item {
    void * adapter;
    float scale;
};

class HouMoLLModel::Impl {
  public:
    Impl() = default;
    ~Impl() { cleanup_tensors(); }

    bool houmo_load(llama_model_loader &loader);
    bool houmo_load_fromfile(llama_model_loader &loader);

    void houmo_init(int seq_max);
    int houmo_prefill(std::vector<llama_token> &tokens, int seq_id,
                      float *logits);

    int houmo_decode(std::vector<llama_token> &batches,
                     std::vector<int> &seq_ids, std::vector<float *> logits);

    int n_decode_batch() { return n_decode_batch_; }
    int n_prefill_batch() { return n_prefill_batch_; }
    uint32_t n_context_length() { return context_length_; }
    void lora_init(std::string file_path, void *adapter);
    void lora_set_scale(void *adapter, float scale);
    void lora_clear();
    void seq_rm(int seq_id, int p0, int p1);
  private:
    void parser_gguf_parameters(struct gguf_context *ctx);
    void cleanup_tensors();
  private:
    int prefill_length_ = 256;
    int context_length_ = 4096;
    int n_blocks_ = 28;
    int64_t n_embd_ = 3584;
    tcim_module_t prefill_model_ = nullptr;
    tcim_module_t decode_model_ = nullptr;
    int device_num_ = 1;
    int n_prefill_batch_ = 1;
    int n_decode_batch_ = 1;
    std::map<std::string, tcim_tensor_t> prefill_input_map_;
    std::map<std::string, tcim_tensor_t> prefill_output_map_;
    std::map<std::string, tcim_tensor_t> decode_input_map_;
    std::map<std::string, tcim_tensor_t> decode_output_map_;
    std::vector<std::string> input_names_;
    std::vector<std::string> decode_output_names_;
    std::vector<std::string> prefill_output_names_;
    std::map<int, kv_cache_item> kv_cache_;
    std::map<void *, lora_item> lora_map_;
    std::vector<float> logits_;
};

void HouMoLLModel::Impl::cleanup_tensors() {}

void HouMoLLModel::Impl::parser_gguf_parameters(struct gguf_context *ctx) {}

bool HouMoLLModel::Impl::houmo_load_fromfile(llama_model_loader &loader) { return false; }

bool HouMoLLModel::Impl::houmo_load(llama_model_loader &loader) { return false; }

void HouMoLLModel::Impl::houmo_init(int seq_max) {}

int HouMoLLModel::Impl::houmo_prefill(std::vector<llama_token> &tokens, int seq_id, float *logits) { return -1; }

int HouMoLLModel::Impl::houmo_decode(std::vector<llama_token> &batches, std::vector<int> &seq_ids, std::vector<float *> logits) { return -1; }

void HouMoLLModel::Impl::lora_init(std::string file_path, void *adapter) {}

void HouMoLLModel::Impl::lora_set_scale(void *adapter, float scale) {}

void HouMoLLModel::Impl::lora_clear() {}

void HouMoLLModel::Impl::seq_rm(int seq_id, int p0, int p1) {}

HouMoLLModel::HouMoLLModel() : impl_(new Impl()) {}

HouMoLLModel::~HouMoLLModel() { delete impl_; }

bool HouMoLLModel::houmo_load(llama_model_loader &loader) { return impl_->houmo_load(loader); }

void HouMoLLModel::houmo_init(int seq_max) { impl_->houmo_init(seq_max); }

int HouMoLLModel::houmo_prefill(std::vector<llama_token> &tokens, int seq_id, float *logits) { return impl_->houmo_prefill(tokens, seq_id, logits); }

int HouMoLLModel::houmo_decode(std::vector<llama_token> &batches, std::vector<int> &seq_ids, std::vector<float *> logits) { return impl_->houmo_decode(batches, seq_ids, logits); }

int HouMoLLModel::n_decode_batch() { return impl_->n_decode_batch(); }

int HouMoLLModel::n_prefill_batch() { return impl_->n_prefill_batch(); }

uint32_t HouMoLLModel::n_context_length() { return impl_->n_context_length(); }

void HouMoLLModel::lora_init(std::string file_name, void *adapter) { impl_->lora_init(file_name, adapter); }

void HouMoLLModel::lora_set_scale(void *adapter, float scale) { impl_->lora_set_scale(adapter, scale); }

void HouMoLLModel::lora_clear() { impl_->lora_clear(); }

void HouMoLLModel::seq_rm(int seq_id, int p0, int p1) { impl_->seq_rm(seq_id, p0, p1); }

#else

class HouMoLLModel::Impl {
  public:
    Impl() = default;
    ~Impl() = default;

    bool houmo_load(llama_model_loader &loader) { return false; }
    void houmo_init(int seq_max) {}
    int houmo_prefill(std::vector<llama_token> &tokens, int seq_id, float *logits) { return -1; }
    int houmo_decode(std::vector<llama_token> &batches, std::vector<int> &seq_ids, std::vector<float *> logits) { return -1; }
    int n_decode_batch() { return 1; }
    int n_prefill_batch() { return 1; }
    uint32_t n_context_length() { return 4096; }
    void lora_init(std::string file_path, void *adapter) {}
    void lora_set_scale(void *adapter, float scale) {}
    void lora_clear() {}
    void seq_rm(int seq_id, int p0, int p1) {}
};

HouMoLLModel::HouMoLLModel() : impl_(new Impl()) {}

HouMoLLModel::~HouMoLLModel() { delete impl_; }

bool HouMoLLModel::houmo_load(llama_model_loader &loader) { return impl_->houmo_load(loader); }

void HouMoLLModel::houmo_init(int seq_max) { impl_->houmo_init(seq_max); }

int HouMoLLModel::houmo_prefill(std::vector<llama_token> &tokens, int seq_id, float *logits) { return impl_->houmo_prefill(tokens, seq_id, logits); }

int HouMoLLModel::houmo_decode(std::vector<llama_token> &batches, std::vector<int> &seq_ids, std::vector<float *> logits) { return impl_->houmo_decode(batches, seq_ids, logits); }

int HouMoLLModel::n_decode_batch() { return impl_->n_decode_batch(); }

int HouMoLLModel::n_prefill_batch() { return impl_->n_prefill_batch(); }

uint32_t HouMoLLModel::n_context_length() { return impl_->n_context_length(); }

void HouMoLLModel::lora_init(std::string file_name, void *adapter) { impl_->lora_init(file_name, adapter); }

void HouMoLLModel::lora_set_scale(void *adapter, float scale) { impl_->lora_set_scale(adapter, scale); }

void HouMoLLModel::lora_clear() { impl_->lora_clear(); }

void HouMoLLModel::seq_rm(int seq_id, int p0, int p1) { impl_->seq_rm(seq_id, p0, p1); }

#endif
#pragma once

#include <map>
#include <string>
#include <vector>

#include "llama-batch.h"
#include "llama-model-loader.h"

class HouMoLLModel {
  public:
    HouMoLLModel();
    ~HouMoLLModel();

    bool houmo_load(llama_model_loader &loader);
    void houmo_init(int seq_max = 4);
    int houmo_prefill(std::vector<llama_token> &tokens, int seq_id,
                      float *logits);
    int houmo_decode(std::vector<llama_token> &batches,
                     std::vector<int> &seq_ids, std::vector<float *> logits);
    int n_decode_batch();
    int n_prefill_batch();
    uint32_t n_context_length();
    void lora_init(std::string file_name, void *adapter);
    void lora_set_scale(void *adapter, float scale);
    void lora_clear();
    void seq_rm(int seq_id, int p0, int p1);
  private:
    class Impl;
    Impl *impl_;
};
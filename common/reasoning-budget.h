#pragma once

#include "llama.h"

#include <cstdint>
#include <vector>

enum common_reasoning_budget_state {
    REASONING_BUDGET_IDLE,
    REASONING_BUDGET_COUNTING,
    REASONING_BUDGET_FORCING,
    REASONING_BUDGET_WAITING_UTF8,
    REASONING_BUDGET_DONE,
};

struct llama_sampler * common_reasoning_budget_init(
        const struct llama_vocab       * vocab,
        const std::vector<llama_token> & start_tokens,
        const std::vector<llama_token> & end_tokens,
        const std::vector<llama_token> & forced_tokens,
        int32_t                          budget,
        const std::vector<llama_token> & prefill_tokens = {});

enum common_reasoning_budget_state common_reasoning_budget_get_state(struct llama_sampler * smpl);

bool common_reasoning_budget_force(struct llama_sampler * smpl);

struct llama_sampler * common_reasoning_budget_init(
        const struct llama_vocab       * vocab,
        const std::vector<llama_token> & start_tokens,
        const std::vector<llama_token> & end_tokens,
        const std::vector<llama_token> & forced_tokens,
        int32_t                          budget,
        common_reasoning_budget_state    initial_state);
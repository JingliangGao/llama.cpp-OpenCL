#include "reasoning-budget.h"
#include "common.h"

#include "log.h"

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

struct token_matcher {
    std::vector<llama_token> tokens;
    size_t pos = 0;

    bool advance(llama_token token) {
        if (tokens.empty()) {
            return false;
        }

        if (token == tokens[pos]) {
            pos++;
            if (pos >= tokens.size()) {
                pos = 0;
                return true;
            }
        } else {
            pos = 0;
            if (token == tokens[0]) {
                pos = 1;
            }
        }
        return false;
    }

    void reset() { pos = 0; }
};

struct common_reasoning_budget_ctx {
    const llama_vocab * vocab;

    token_matcher start_matcher;
    token_matcher end_matcher;
    std::vector<llama_token> forced_tokens;

    int32_t budget;
    int32_t remaining;

    common_reasoning_budget_state state;

    size_t force_pos;
};

static const char * common_reasoning_budget_name(const struct llama_sampler * /*smpl*/) {
    return "reasoning-budget";
}

static void common_reasoning_budget_accept(struct llama_sampler * smpl, llama_token token) {
    auto * ctx = (common_reasoning_budget_ctx *) smpl->ctx;

    switch (ctx->state) {
        case REASONING_BUDGET_IDLE:
        {
            if (ctx->start_matcher.advance(token)) {
                ctx->state = REASONING_BUDGET_COUNTING;
                ctx->remaining = ctx->budget;
                LOG_INF("reasoning-budget: activated, budget=%d tokens\n", ctx->budget);

                if (ctx->remaining <= 0) {
                    ctx->state = REASONING_BUDGET_FORCING;
                    ctx->force_pos = 0;
                    LOG_INF("reasoning-budget: budget=0, forcing immediately\n");
                }
            }
            break;
        }
        case REASONING_BUDGET_COUNTING:
        case REASONING_BUDGET_WAITING_UTF8:
        {
            if (ctx->end_matcher.advance(token)) {
                ctx->state = REASONING_BUDGET_DONE;
                LOG_INF("reasoning-budget: deactivated (natural end)\n");
                break;
            }

            bool utf8_complete = true;
            (void) ctx->vocab;
            (void) token;

            if (ctx->state == REASONING_BUDGET_WAITING_UTF8) {
                if (utf8_complete) {
                    ctx->state = REASONING_BUDGET_FORCING;
                    ctx->force_pos = 0;
                    ctx->end_matcher.reset();
                    LOG_INF("reasoning-budget: UTF-8 complete, now forcing end sequence\n");
                }
            } else if (ctx->state == REASONING_BUDGET_COUNTING) {
                ctx->remaining--;
                if (ctx->remaining <= 0) {
                    if (utf8_complete) {
                        ctx->state = REASONING_BUDGET_FORCING;
                        ctx->force_pos = 0;
                        ctx->end_matcher.reset();
                        LOG_INF("reasoning-budget: budget exhausted, forcing end sequence\n");
                    } else {
                        ctx->state = REASONING_BUDGET_WAITING_UTF8;
                        ctx->end_matcher.reset();
                        LOG_INF("reasoning-budget: budget exhausted, waiting for UTF-8 completion\n");
                    }
                }
            }
            break;
        }
        case REASONING_BUDGET_FORCING:
            break;
        case REASONING_BUDGET_DONE:
        {
            if (ctx->start_matcher.advance(token)) {
                ctx->state = REASONING_BUDGET_COUNTING;
                ctx->remaining = ctx->budget;
                LOG_INF("reasoning-budget: re-activated, budget=%d tokens\n", ctx->budget);
                if (ctx->remaining <= 0) {
                    ctx->state = REASONING_BUDGET_FORCING;
                    ctx->force_pos = 0;
                    LOG_INF("reasoning-budget: budget=0, forcing immediately\n");
                }
            }
            break;
        }
    }
}

static void common_reasoning_budget_apply(struct llama_sampler * smpl, llama_token_data_array * cur_p) {
    auto * ctx = (common_reasoning_budget_ctx *) smpl->ctx;

    if (ctx->state != REASONING_BUDGET_FORCING) {
        return;
    }

    if (ctx->force_pos >= ctx->forced_tokens.size()) {
        return;
    }

    const llama_token forced = ctx->forced_tokens[ctx->force_pos];

    for (size_t i = 0; i < cur_p->size; i++) {
        if (cur_p->data[i].id != forced) {
            cur_p->data[i].logit = -INFINITY;
        }
    }

    ctx->force_pos++;
    if (ctx->force_pos >= ctx->forced_tokens.size()) {
        ctx->state = REASONING_BUDGET_DONE;
        LOG_INF("reasoning-budget: forced sequence complete, done\n");
    }
}

static void common_reasoning_budget_reset(struct llama_sampler * smpl) {
    auto * ctx = (common_reasoning_budget_ctx *) smpl->ctx;
    ctx->state = REASONING_BUDGET_IDLE;
    ctx->remaining = ctx->budget;
    ctx->start_matcher.reset();
    ctx->end_matcher.reset();
    ctx->force_pos = 0;
}

static struct llama_sampler * common_reasoning_budget_init_state(
        const struct llama_vocab * vocab, const std::vector<llama_token> & start_tokens,
        const std::vector<llama_token> & end_tokens, const std::vector<llama_token> & forced_tokens,
        int32_t budget, common_reasoning_budget_state initial_state);

static struct llama_sampler * common_reasoning_budget_clone(const struct llama_sampler * smpl);

static void common_reasoning_budget_free(struct llama_sampler * smpl) {
    delete (common_reasoning_budget_ctx *) smpl->ctx;
}

static struct llama_sampler_i common_reasoning_budget_i = {
    /* .name              = */ common_reasoning_budget_name,
    /* .accept            = */ common_reasoning_budget_accept,
    /* .apply             = */ common_reasoning_budget_apply,
    /* .reset             = */ common_reasoning_budget_reset,
    /* .clone             = */ common_reasoning_budget_clone,
    /* .free              = */ common_reasoning_budget_free,
    /* .backend_init      = */ nullptr,
    /* .backend_accept    = */ nullptr,
    /* .backend_apply     = */ nullptr,
    /* .backend_set_input = */ nullptr,
};

static struct llama_sampler * common_reasoning_budget_clone(const struct llama_sampler * smpl) {
    const auto * ctx = (const common_reasoning_budget_ctx *) smpl->ctx;
    return llama_sampler_init(
        /* .iface = */ &common_reasoning_budget_i,
        /* .ctx   = */ new common_reasoning_budget_ctx {
            /* .vocab         = */ ctx->vocab,
            /* .start_matcher = */ { ctx->start_matcher.tokens, ctx->start_matcher.pos },
            /* .end_matcher   = */ { ctx->end_matcher.tokens, ctx->end_matcher.pos },
            /* .forced_tokens = */ ctx->forced_tokens,
            /* .budget        = */ ctx->budget,
            /* .remaining     = */ ctx->remaining,
            /* .state         = */ ctx->state,
            /* .force_pos     = */ ctx->force_pos,
        }
    );
}

static struct llama_sampler * common_reasoning_budget_init_state(
        const struct llama_vocab             * vocab,
        const std::vector<llama_token>       & start_tokens,
        const std::vector<llama_token>       & end_tokens,
        const std::vector<llama_token>       & forced_tokens,
        int32_t                                budget,
        common_reasoning_budget_state          initial_state) {
    if (initial_state == REASONING_BUDGET_COUNTING && budget <= 0) {
        initial_state = REASONING_BUDGET_FORCING;
    }

    return llama_sampler_init(
        /* .iface = */ &common_reasoning_budget_i,
        /* .ctx   = */ new common_reasoning_budget_ctx {
            /* .vocab         = */ vocab,
            /* .start_matcher = */ { start_tokens, 0 },
            /* .end_matcher   = */ { end_tokens, 0 },
            /* .forced_tokens = */ forced_tokens,
            /* .budget        = */ budget,
            /* .remaining     = */ budget,
            /* .state         = */ initial_state,
            /* .force_pos     = */ 0,
        }
    );
}

struct llama_sampler * common_reasoning_budget_init(
        const struct llama_vocab       * vocab,
        const std::vector<llama_token> & start_tokens,
        const std::vector<llama_token> & end_tokens,
        const std::vector<llama_token> & forced_tokens,
        int32_t                          budget,
        const std::vector<llama_token> & prefill_tokens) {
    common_reasoning_budget_state initial_state = REASONING_BUDGET_IDLE;
    if (!prefill_tokens.empty() && !start_tokens.empty() &&
            prefill_tokens.size() >= start_tokens.size() &&
            std::equal(start_tokens.begin(), start_tokens.end(), prefill_tokens.begin())) {
        initial_state = REASONING_BUDGET_COUNTING;
        if (!end_tokens.empty() &&
                prefill_tokens.size() >= start_tokens.size() + end_tokens.size()) {
            auto end_start = prefill_tokens.end() - (ptrdiff_t) end_tokens.size();
            if (end_start >= prefill_tokens.begin() + (ptrdiff_t) start_tokens.size() &&
                    std::equal(end_tokens.begin(), end_tokens.end(), end_start)) {
                initial_state = REASONING_BUDGET_IDLE;
            }
        }
    }
    return common_reasoning_budget_init_state(vocab, start_tokens, end_tokens, forced_tokens, budget, initial_state);
}

struct llama_sampler * common_reasoning_budget_init(
        const struct llama_vocab       * vocab,
        const std::vector<llama_token> & start_tokens,
        const std::vector<llama_token> & end_tokens,
        const std::vector<llama_token> & forced_tokens,
        int32_t                          budget,
        common_reasoning_budget_state    initial_state) {
    return common_reasoning_budget_init_state(vocab, start_tokens, end_tokens, forced_tokens, budget, initial_state);
}

enum common_reasoning_budget_state common_reasoning_budget_get_state(struct llama_sampler * smpl) {
    if (!smpl) {
        return REASONING_BUDGET_IDLE;
    }
    auto * ctx = (common_reasoning_budget_ctx *) smpl->ctx;
    return ctx->state;
}

bool common_reasoning_budget_force(struct llama_sampler * smpl) {
    if (!smpl) {
        return false;
    }
    auto * ctx = (common_reasoning_budget_ctx *) smpl->ctx;
    if (ctx->state == REASONING_BUDGET_COUNTING) {
        ctx->state = REASONING_BUDGET_FORCING;
        ctx->force_pos = 0;
        return true;
    }
    return false;
}
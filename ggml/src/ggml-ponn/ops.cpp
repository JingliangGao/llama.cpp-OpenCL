#include "common.h"
#include "ggml-impl.h"

#include <cassert>
#include <vector>

void ggml_ponn_sqr(ggml_backend_ponn_context& ctx, ggml_tensor* dst) {
    (void)ctx;
    (void)dst;
    GGML_ASSERT(0);
}

void ggml_ponn_clamp(ggml_backend_ponn_context & ctx, ggml_tensor * dst) {
    (void)ctx;
    (void)dst;
    GGML_ASSERT(0);
}

void ggml_ponn_dup(ggml_backend_ponn_context& ctx, ggml_tensor* dst) {
    (void)ctx;
    (void)dst;
}

void ponn_dup_no_contiguous(ggml_tensor *dst) {
    (void)dst;
}

void ggml_ponn_get_rows(ggml_backend_ponn_context& ctx, ggml_tensor* dst) {
    (void)ctx;
    (void)dst;
}

void ggml_ponn_soft_max(ggml_backend_ponn_context& ctx, ggml_tensor* dst) {
    (void)ctx;
    (void)dst;
}

void update_ggml_ponn_rope_cache(float freq_base, int rotary_dim, int n_ctx_orig, float beta_fast, float beta_slow, float ext_factor, float attn_factor, float freq_scale, ggml_backend_ponn_context& ctx) {
    (void)freq_base;
    (void)rotary_dim;
    (void)n_ctx_orig;
    (void)beta_fast;
    (void)beta_slow;
    (void)ext_factor;
    (void)attn_factor;
    (void)freq_scale;
    (void)ctx;
}

void ggml_ponn_rope(ggml_backend_ponn_context& ctx, ggml_tensor* dst) {
    (void)ctx;
    (void)dst;
}
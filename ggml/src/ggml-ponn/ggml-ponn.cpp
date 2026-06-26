#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "ggml-backend-meta.h"
#include "common.h"
#include "ponn.h"

#include <cassert>
#include <cstring>
#include <vector>

static bool ggml_backend_ponn_supported(void) {
    return true;
}

static struct ggml_backend* ggml_backend_ponn_init(void) {
    auto ponn_ctx = new ggml_backend_ponn_context();
    ponn_ctx->init_done = true;
    ponnInit();

    auto backend = ggml_backend_new(ponn_ctx);
    return backend;
}

static void ggml_backend_ponn_free(struct ggml_backend* backend) {
    ggml_backend_ponn_context* ponn_ctx = (ggml_backend_ponn_context*)backend->context;
    delete ponn_ctx;
    delete backend;
    ponnClearBuffer();
}

static struct ggml_backend_buffer_type* ggml_backend_ponn_buffer_type(void) {
    static const struct ggml_backend_buffer_type_i buffer_type_interface = {
        /* .get_name         = */ nullptr,
        /* .alloc_buffer     = */ nullptr,
        /* .get_alignment    = */ nullptr,
        /* .get_max_size     = */ nullptr,
        /* .get_alloc_size   = */ nullptr,
        /* .is_host          = */ nullptr,
    };
    return ggml_backend_buffer_type_new(&buffer_type_interface, nullptr);
}

static struct ggml_backend_sched_type* ggml_backend_ponn_sched_type(void) {
    return nullptr;
}

static struct ggml_backend_graph_type* ggml_backend_ponn_graph_type(void) {
    return nullptr;
}

static const struct ggml_backend_i ggml_backend_ponn_interface = {
    /* .type              = */ GGML_BACKEND_TYPE_GPU,
    /* .name              = */ "ponn",
    /* .supported         = */ ggml_backend_ponn_supported,
    /* .init              = */ ggml_backend_ponn_init,
    /* .free              = */ ggml_backend_ponn_free,
    /* .buffer_type       = */ ggml_backend_ponn_buffer_type,
    /* .sched_type        = */ ggml_backend_ponn_sched_type,
    /* .graph_type        = */ ggml_backend_ponn_graph_type,
};

GGML_API const struct ggml_backend_i* ggml_backend_ponn_get_interface(void) {
    return &ggml_backend_ponn_interface;
}
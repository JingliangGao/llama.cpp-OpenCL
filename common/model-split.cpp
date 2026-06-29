#include "model-split.h"
#include "common.h"
#include "log.h"

#include "ggml-backend.h"

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdlib>

#ifdef GGML_KYLIN_SUPPORT                                                                                   // JingliangGao 2026/06/27

// Helper to convert size string to bytes (supports K, M, G, T suffixes)
size_t common_parse_size(const std::string & size_str) {
    if (size_str.empty()) {
        return 0;
    }

    char * end = nullptr;
    double value = std::strtod(size_str.c_str(), &end);

    if (end == size_str.c_str()) {
        return 0; // No valid number found
    }

    if (*end != '\0') {
        // Check for suffix
        char suffix = *end;
        switch (suffix) {
            case 'T': case 't':
                value *= 1024.0 * 1024 * 1024 * 1024;
                break;
            case 'G': case 'g':
                value *= 1024.0 * 1024 * 1024;
                break;
            case 'M': case 'm':
                value *= 1024.0 * 1024;
                break;
            case 'K': case 'k':
                value *= 1024.0;
                break;
            default:
                return 0; // Invalid suffix
        }
    }

    return static_cast<size_t>(value);
}

// Get default split configuration
struct common_model_split_config common_model_split_default_config(void) {
    struct common_model_split_config config;
    config.enabled = false;
    config.vram_budget_bytes = 0; // auto-detect
    config.vram_usage_ratio = 0.85f;
    config.min_layer_size_bytes = 0; // auto-detect
    config.max_gpu_layers = 0; // unlimited
    return config;
}

// Estimate model size from GGUF file without loading the model
uint64_t common_model_estimate_size(const std::string & model_path) {
    std::ifstream file(model_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        LOG_ERR("%s: failed to open model file '%s'\n", __func__, model_path.c_str());
        return 0;
    }

    // Get file size
    uint64_t file_size = static_cast<uint64_t>(file.tellg());
    file.seekg(0, std::ios::beg);

    // Read magic number
    char magic[4] = {0};
    file.read(magic, sizeof(magic));
    if (magic[0] != 'G' || magic[1] != 'G' || magic[2] != 'U' || magic[3] != 'F') {
        LOG_ERR("%s: invalid GGUF file '%s'\n", __func__, model_path.c_str());
        return 0;
    }

    // Return file size as model size estimate
    return file_size;
}

// Get total available VRAM across all GPUs in bytes
size_t common_get_available_vram(void) {
    size_t total_vram = 0;

    // Count GPU devices
    size_t n_devices = ggml_backend_dev_count();
    for (size_t i = 0; i < n_devices; ++i) {
        auto * dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            size_t free_mem, total_mem;
            ggml_backend_dev_memory(dev, &free_mem, &total_mem);
            total_vram += total_mem;
        }
    }

    return total_vram;
}

// Auto-determine n_gpu_layers based on model size and VRAM budget
int common_model_auto_split_n_gpu_layers(
    const std::string & model_path,
    const struct common_model_split_config & config,
    int model_n_layers) {

    if (model_n_layers <= 0) {
        LOG_ERR("%s: invalid model layer count: %d\n", __func__, model_n_layers);
        return -1;
    }

    // Get model size
    uint64_t model_size = common_model_estimate_size(model_path);
    if (model_size == 0) {
        LOG_ERR("%s: failed to estimate model size for '%s'\n", __func__, model_path.c_str());
        return -1;
    }

    // Get VRAM budget
    size_t vram_budget = config.vram_budget_bytes;
    if (vram_budget == 0) {
        vram_budget = common_get_available_vram();
    }

    if (vram_budget == 0) {
        LOG_WRN("%s: no VRAM available, disabling auto-split\n", __func__);
        return -1;
    }

    // Apply VRAM usage ratio
    size_t effective_vram = static_cast<size_t>(static_cast<double>(vram_budget) * config.vram_usage_ratio);

    // Calculate average layer size
    uint64_t avg_layer_size = model_size / static_cast<uint64_t>(model_n_layers);

    // Check if model is small enough to fit entirely in VRAM
    if (model_size <= effective_vram) {
        LOG_INF("%s: model size (%.2f GB) fits in VRAM (%.2f GB), loading full model\n",
                __func__,
                static_cast<double>(model_size) / (1024.0 * 1024 * 1024),
                static_cast<double>(effective_vram) / (1024.0 * 1024 * 1024));
        return -1; // No splitting needed
    }

    // Calculate how many layers can fit in VRAM
    int n_gpu_layers = static_cast<int>(effective_vram / avg_layer_size);

    // Clamp to valid range
    if (n_gpu_layers < 0) {
        n_gpu_layers = 0;
    }
    if (n_gpu_layers > model_n_layers) {
        n_gpu_layers = model_n_layers;
    }

    // Apply max_gpu_layers limit if set
    if (config.max_gpu_layers > 0) {
        if (n_gpu_layers > config.max_gpu_layers) {
            n_gpu_layers = config.max_gpu_layers;
        }
    }

    // If we can't offload any layers, disable splitting
    if (n_gpu_layers == 0) {
        LOG_INF("%s: model size (%.2f GB) exceeds VRAM (%.2f GB), but no layers fit in VRAM, running on CPU\n",
                __func__,
                static_cast<double>(model_size) / (1024.0 * 1024 * 1024),
                static_cast<double>(effective_vram) / (1024.0 * 1024 * 1024));
        return 0;
    }

    LOG_INF("%s: model size: %.2f GB, VRAM budget: %.2f GB, avg layer size: %.2f MB, offloading %d/%d layers\n",
            __func__,
            static_cast<double>(model_size) / (1024.0 * 1024 * 1024),
            static_cast<double>(effective_vram) / (1024.0 * 1024 * 1024),
            static_cast<double>(avg_layer_size) / (1024.0 * 1024),
            n_gpu_layers, model_n_layers);

    return n_gpu_layers;
}

// Parse split config from string
bool common_model_split_parse_config(const std::string & value, struct common_model_split_config & config) {
    config = common_model_split_default_config();

    if (value.empty()) {
        return false;
    }

    // Check if enabled
    bool is_enabled = false;
    std::string options_str;

    if (value == "on" || value == "true" || value == "1") {
        is_enabled = true;
        options_str = "";
    } else if (value.size() > 3 && value.substr(0, 3) == "on,") {
        is_enabled = true;
        options_str = value.substr(3);
    } else if (value.size() > 5 && value.substr(0, 5) == "true,") {
        is_enabled = true;
        options_str = value.substr(5);
    } else if (value.size() > 2 && value.substr(0, 2) == "1,") {
        is_enabled = true;
        options_str = value.substr(2);
    } else {
        return false;
    }

    if (!is_enabled) {
        return false;
    }

    config.enabled = true;

    // Parse key=value pairs
    if (options_str.empty()) {
        return true;
    }

    std::istringstream iss(options_str);
    std::string token;
    while (std::getline(iss, token, ',')) {
        size_t eq_pos = token.find('=');
        if (eq_pos == std::string::npos) {
            LOG_ERR("%s: invalid option format: '%s'\n", __func__, token.c_str());
            return false;
        }

        std::string key = token.substr(0, eq_pos);
        std::string val = token.substr(eq_pos + 1);

        if (key == "vram") {
            config.vram_budget_bytes = common_parse_size(val);
        } else if (key == "ratio") {
            config.vram_usage_ratio = std::stof(val);
            if (config.vram_usage_ratio <= 0.0f || config.vram_usage_ratio > 1.0f) {
                LOG_ERR("%s: invalid ratio value: %s\n", __func__, val.c_str());
                return false;
            }
        } else if (key == "max_layers") {
            config.max_gpu_layers = std::stoi(val);
            if (config.max_gpu_layers < 0) {
                LOG_ERR("%s: invalid max_layers value: %s\n", __func__, val.c_str());
                return false;
            }
        } else {
            LOG_ERR("%s: unknown option: '%s'\n", __func__, key.c_str());
            return false;
        }
    }

    return true;
}

// Global auto-split config (set via CLI)
static struct common_model_split_config g_split_config = common_model_split_default_config();

void common_model_split_set_config(const struct common_model_split_config & config) {
    g_split_config = config;
}

const struct common_model_split_config & common_model_split_get_config(void) {
    return g_split_config;
}

void common_model_auto_split_apply(common_params & params) {
    if (!g_split_config.enabled) {
        return; // Auto-split is disabled
    }

    // Only apply if n_gpu_layers is not explicitly set (still -1)
    if (params.n_gpu_layers != -1) {
        LOG_INF("%s: n_gpu_layers already set to %d, skipping auto-split\n", __func__, params.n_gpu_layers);
        return;
    }

    // Load model to get layer count
    // We need to load the model metadata first to get n_layer
    struct llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0; // Don't load any layers yet

    struct llama_model * model = llama_model_load_from_file(params.model.c_str(), mparams);
    if (!model) {
        LOG_ERR("%s: failed to load model '%s' for auto-split\n", __func__, params.model.c_str());
        return;
    }

    int n_layers = llama_model_n_layer(model);
    llama_model_free(model);

    if (n_layers <= 0) {
        LOG_ERR("%s: invalid layer count: %d\n", __func__, n_layers);
        return;
    }

    // Calculate optimal n_gpu_layers
    int n_gpu_layers = common_model_auto_split_n_gpu_layers(
        params.model, g_split_config, n_layers);

    if (n_gpu_layers >= 0) {
        params.n_gpu_layers = n_gpu_layers;
        LOG_INF("%s: auto-split set n_gpu_layers to %d\n", __func__, n_gpu_layers);
    } else {
        LOG_INF("%s: auto-split determined no splitting is needed\n", __func__);
    }
}

#endif // GGML_KYLIN_SUPPORT                                                                               // JingliangGao 2026/06/27

#pragma once

#ifdef GGML_KYLIN_SUPPORT                                                                                   // JingliangGao 2026/06/27
//                                                                                                         // JingliangGao 2026/06/27
// Model auto-split utilities                                                                               // JingliangGao 2026/06/27
//                                                                                                         // JingliangGao 2026/06/27
// This module provides functionality to automatically determine                                            // JingliangGao 2026/06/27
// whether a model should be split across GPUs based on its file size,                                     // JingliangGao 2026/06/27
// and automatically configures n_gpu_layers accordingly.                                                 // JingliangGao 2026/06/27
//                                                                                                         // JingliangGao 2026/06/27

#include <cstdint>
#include <string>

struct common_params;

// Auto-split configuration parameters
struct common_model_split_config {
    bool enabled;                    // whether to enable auto-split (default: false)
    size_t vram_budget_bytes;        // available VRAM budget in bytes (0 = auto-detect)
    float vram_usage_ratio;          // target VRAM usage ratio (0.0-1.0, default: 0.85)
    size_t min_layer_size_bytes;     // minimum size per layer to consider for splitting
    int max_gpu_layers;              // maximum number of layers to offload (0 = unlimited)
};

// Get default split configuration
struct common_model_split_config common_model_split_default_config(void);

// Estimate model size from GGUF file without loading the model
// Returns 0 on failure, otherwise returns estimated model size in bytes
uint64_t common_model_estimate_size(const std::string & model_path);

// Get total available VRAM across all GPUs in bytes
// Returns 0 if no GPU is available or VRAM detection is not supported
size_t common_get_available_vram(void);

// Auto-determine n_gpu_layers based on model size and VRAM budget
// Returns the recommended number of GPU layers, or -1 if no offloading is needed
// If split is not needed, returns -1 to indicate no extra operation
int common_model_auto_split_n_gpu_layers(
    const std::string & model_path,
    const struct common_model_split_config & config,
    int model_n_layers);

// Apply auto-split configuration to common_params
// This function will modify params.n_gpu_layers based on model size and VRAM budget
// Only takes effect if config.enabled is true
void common_model_auto_split_apply(common_params & params);

// Parse split config from string (for CLI argument parsing)
// Format: "enabled[,vram=N][,ratio=N][,max_layers=N]"
// Examples: "on", "on,vram=8G", "on,ratio=0.9", "on,vram=16G,ratio=0.85,max_layers=50"
bool common_model_split_parse_config(const std::string & value, struct common_model_split_config & config);

// Helper to convert size string to bytes (supports K, M, G, T suffixes)
size_t common_parse_size(const std::string & size_str);

// Set the global split config (called from CLI argument handler)
void common_model_split_set_config(const struct common_model_split_config & config);

// Get the current global split config
const struct common_model_split_config & common_model_split_get_config(void);

#endif // GGML_KYLIN_SUPPORT                                                                               // JingliangGao 2026/06/27

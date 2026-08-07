#pragma once

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// Returns true when a tensor name belongs to a packed MoE expert tensor.
bool ggml_hotexpert_is_expert_tensor(const char * name);

// Returns the parsed transformer block index, or -1 when unavailable.
int ggml_hotexpert_parse_layer(const char * name);

// Model metadata hook used by the CUDA/HIP runtime descriptor.
void ggml_hotexpert_set_model_meta(int n_layers, int n_experts, int n_experts_used);
int  ggml_hotexpert_model_layers(void);
int  ggml_hotexpert_model_experts(void);
int  ggml_hotexpert_model_topk(void);

#ifdef __cplusplus
}
#endif

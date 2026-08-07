#pragma once

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

GGML_API void         ggml_hotexpert_set_model_info(const char * architecture, int n_layers, int n_experts, int n_expert_used);
GGML_API const char * ggml_hotexpert_model_architecture(void);
GGML_API int          ggml_hotexpert_model_layers(void);
GGML_API int          ggml_hotexpert_model_experts(void);
GGML_API int          ggml_hotexpert_model_expert_used(void);

#ifdef __cplusplus
}
#endif

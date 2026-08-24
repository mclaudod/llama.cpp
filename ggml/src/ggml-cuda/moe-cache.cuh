#pragma once

// MoE expert cache registration entry point, called from ggml_backend_cuda_reg().
// Populates ggml_moe_cache (see ggml-backend-moe-cache.h).

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void ggml_moe_cache_register(const void * owner);

// Surrender the device's cache VRAM under allocator pressure; returns bytes freed.
size_t ggml_moe_cache_trim(int device);

// Experimental scheduler-side expert materialization used through the CUDA backend proc table.
int ggml_cuda_moe_cache_materialize_enabled(void);
int ggml_cuda_moe_cache_materialize(
        void * backend, const char * tensor_name, const void * host_base,
        size_t tensor_size, size_t expert_size, int wtype, int64_t n_expert,
        const int32_t * ids, int n_ids, void * dst_base);

// V1.4: prepare only cache misses in the scheduler work tensor and publish
// a stable device-side expert_id -> weight pointer table for decode MMVQ.
int ggml_cuda_moe_cache_direct_materialize(
        void * backend, const char * tensor_name, const void * host_base,
        size_t tensor_size, size_t expert_size, int wtype, int64_t n_expert,
        const int32_t * ids, int n_ids, void * dst_base);

// V1.5: scheduler-direct materialization driven by the CUDA-resident MoE IDs.
// This removes the synchronous GPU->CPU ID readback from the hot path.
int ggml_cuda_moe_cache_direct_materialize_gpu_ids(
        void * backend, const char * tensor_name, const void * host_base,
        size_t tensor_size, size_t expert_size, int wtype, int64_t n_expert,
        const int32_t * device_ids, int64_t ids_ne0, int64_t ids_ne1,
        size_t ids_nb1, const void * ids_group, void * dst_base);

#ifdef __cplusplus
}

struct ggml_backend_cuda_context;
struct ggml_tensor;

// Returns 1 if the direct-resident MMVQ path executed, otherwise 0.
int ggml_cuda_moe_cache_direct_mmv(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0,
        const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst);
#endif

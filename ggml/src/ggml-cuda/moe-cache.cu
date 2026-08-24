#include "moe-cache.cuh"

#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)

extern "C" size_t ggml_moe_cache_trim(int device) {
    (void) device;
    return 0;
}

extern "C" int ggml_cuda_moe_cache_materialize_enabled(void) {
    return 0;
}

extern "C" int ggml_cuda_moe_cache_materialize(
        void * backend, const char * tensor_name, const void * host_base,
        size_t tensor_size, size_t expert_size, int wtype, int64_t n_expert,
        const int32_t * ids, int n_ids, void * dst_base) {
    (void) backend;
    (void) tensor_name;
    (void) host_base;
    (void) tensor_size;
    (void) expert_size;
    (void) wtype;
    (void) n_expert;
    (void) ids;
    (void) n_ids;
    (void) dst_base;
    return 0;
}

extern "C" int ggml_cuda_moe_cache_direct_materialize(
        void * backend, const char * tensor_name, const void * host_base,
        size_t tensor_size, size_t expert_size, int wtype, int64_t n_expert,
        const int32_t * ids, int n_ids, void * dst_base) {
    (void) backend;
    (void) tensor_name;
    (void) host_base;
    (void) tensor_size;
    (void) expert_size;
    (void) wtype;
    (void) n_expert;
    (void) ids;
    (void) n_ids;
    (void) dst_base;
    return 0;
}

extern "C" int ggml_cuda_moe_cache_direct_materialize_gpu_ids(
        void * backend, const char * tensor_name, const void * host_base,
        size_t tensor_size, size_t expert_size, int wtype, int64_t n_expert,
        const int32_t * device_ids, int64_t ids_ne0, int64_t ids_ne1,
        size_t ids_nb1, const void * ids_group, void * dst_base) {
    (void) backend;
    (void) tensor_name;
    (void) host_base;
    (void) tensor_size;
    (void) expert_size;
    (void) wtype;
    (void) n_expert;
    (void) device_ids;
    (void) ids_ne0;
    (void) ids_ne1;
    (void) ids_nb1;
    (void) ids_group;
    (void) dst_base;
    return 0;
}

int ggml_cuda_moe_cache_direct_mmv(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0,
        const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst) {
    (void) ctx;
    (void) src0;
    (void) src1;
    (void) ids;
    (void) dst;
    return 0;
}

void ggml_moe_cache_register(const void * owner) {
    (void) owner;
}

#else

#include "common.cuh"
#include "mmvq.cuh"
#include "quantize.cuh"
#include "vecdotq.cuh"
#include "ggml-backend-impl.h"
#include "ggml-cuda.h"
#include "../ggml-backend-moe-cache.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <climits>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#define MOE_CACHE_LOG(...) GGML_LOG_INFO(__VA_ARGS__)

static constexpr int    moe_cache_cc_forced_min               = 700;
static constexpr int    moe_cache_cc_ampere                   = 800;
static constexpr size_t moe_cache_expert_bytes_ampere_min     = 512u << 10;
static constexpr size_t moe_cache_expert_bytes_pre_ampere_min = 1u << 20;
static constexpr int    moe_cache_batch_max                   = 8;
static constexpr int    moe_cache_pool_slots_min              = 64;
static constexpr size_t moe_cache_slab_bytes_auto_min         = 1ull << 30;
static constexpr int    moe_cache_node_rows_max               = 64;
static constexpr size_t moe_cache_overlap_bytes_per_token     = 8u << 20;
static constexpr int    moe_cache_copy_desc_max               = 2 * moe_cache_node_rows_max;
static constexpr int    moe_cache_copy_blocks_per_desc        = 2;

struct moe_cache_copy_desc {
    const char * source = nullptr;
    char * destination = nullptr;
    size_t bytes = 0;
};

struct moe_cache_copy_batch {
    moe_cache_copy_desc desc[moe_cache_copy_desc_max];
    int count = 0;
};

struct moe_cache_direct_desc {
    int32_t expert = -1;
    const char * source = nullptr;
    char * destination = nullptr;
    const void * weight_pointer = nullptr;
    size_t bytes = 0;
    int copy_weight = 0;
};

struct moe_cache_direct_batch {
    const void ** pointer_table = nullptr;
    moe_cache_direct_desc desc[moe_cache_node_rows_max];
    int count = 0;
};

static __global__ void moe_cache_copy_batch_kernel(moe_cache_copy_batch batch) {
    const int desc_index = (int)blockIdx.y;
    if (desc_index < 0 || desc_index >= batch.count) {
        return;
    }

    const moe_cache_copy_desc item = batch.desc[desc_index];
    const size_t vector_bytes = sizeof(uint4);
    const size_t thread_offset =
        ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * vector_bytes;
    const size_t stride =
        (size_t)gridDim.x * blockDim.x * vector_bytes;

    for (size_t offset = thread_offset; offset < item.bytes; offset += stride) {
        const size_t remaining = item.bytes - offset;
        if (remaining >= vector_bytes &&
            (((uintptr_t)(item.source + offset) |
              (uintptr_t)(item.destination + offset)) & (vector_bytes - 1)) == 0) {
            *reinterpret_cast<uint4 *>(item.destination + offset) =
                *reinterpret_cast<const uint4 *>(item.source + offset);
        } else {
            for (size_t byte = 0;
                 byte < remaining && byte < vector_bytes;
                 ++byte) {
                item.destination[offset + byte] = item.source[offset + byte];
            }
        }
    }
}

static __global__ void moe_cache_direct_prepare_kernel(moe_cache_direct_batch batch) {
    const int desc_index = (int)blockIdx.y;
    if (desc_index < 0 || desc_index >= batch.count) {
        return;
    }

    const moe_cache_direct_desc item = batch.desc[desc_index];
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        batch.pointer_table[item.expert] = item.weight_pointer;
    }
    if (!item.copy_weight) {
        return;
    }

    const size_t vector_bytes = sizeof(uint4);
    const size_t thread_offset =
        ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * vector_bytes;
    const size_t stride =
        (size_t)gridDim.x * blockDim.x * vector_bytes;

    for (size_t offset = thread_offset; offset < item.bytes; offset += stride) {
        const size_t remaining = item.bytes - offset;
        if (remaining >= vector_bytes &&
            (((uintptr_t)(item.source + offset) |
              (uintptr_t)(item.destination + offset)) & (vector_bytes - 1)) == 0) {
            *reinterpret_cast<uint4 *>(item.destination + offset) =
                *reinterpret_cast<const uint4 *>(item.source + offset);
        } else {
            for (size_t byte = 0;
                 byte < remaining && byte < vector_bytes;
                 ++byte) {
                item.destination[offset + byte] = item.source[offset + byte];
            }
        }
    }
}


struct moe_cache_gpu_ids_batch {
    const void * const * pointer_table = nullptr;
    const char * mapped_host = nullptr;
    char * dst_base = nullptr;
    const int32_t * ids = nullptr;
    size_t expert_size = 0;
    size_t ids_stride = 0;
    int n_expert = 0;
    int top_k = 0;
    int n_tokens = 0;
};

static __global__ void moe_cache_gpu_ids_prepare_kernel(moe_cache_gpu_ids_batch batch) {
    const int expert = (int)blockIdx.y;
    if (expert < 0 || expert >= batch.n_expert) {
        return;
    }

    bool used = false;
    for (int token = 0; token < batch.n_tokens && !used; ++token) {
        const int32_t * row = batch.ids + (size_t)token * batch.ids_stride;
        for (int slot = 0; slot < batch.top_k; ++slot) {
            if (row[slot] == expert) {
                used = true;
                break;
            }
        }
    }
    if (!used) {
        return;
    }

    char * work = batch.dst_base + (size_t)expert * batch.expert_size;
    if (batch.pointer_table[expert] != work) {
        return;
    }

    const char * source = batch.mapped_host + (size_t)expert * batch.expert_size;
    const size_t vector_bytes = sizeof(uint4);
    const size_t thread_offset =
        ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * vector_bytes;
    const size_t stride =
        (size_t)gridDim.x * blockDim.x * vector_bytes;

    for (size_t offset = thread_offset; offset < batch.expert_size; offset += stride) {
        const size_t remaining = batch.expert_size - offset;
        if (remaining >= vector_bytes &&
            (((uintptr_t)(source + offset) |
              (uintptr_t)(work + offset)) & (vector_bytes - 1)) == 0) {
            *reinterpret_cast<uint4 *>(work + offset) =
                *reinterpret_cast<const uint4 *>(source + offset);
        } else {
            for (size_t byte = 0;
                 byte < remaining && byte < vector_bytes;
                 ++byte) {
                work[offset + byte] = source[offset + byte];
            }
        }
    }
}

// V1.4 POC is intentionally limited to the three quantized expert shapes
// used by Qwen3.6-35B-A3B Q4_K_XL on the current Ampere test target.
static bool moe_cache_scheduler_direct_type_supported(ggml_type type) {
    return type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K ||
           type == GGML_TYPE_Q6_K;
}

using moe_cache_vec_dot_q_cuda_t = float (*)(
    const void * GGML_CUDA_RESTRICT,
    const block_q8_1 * GGML_CUDA_RESTRICT,
    const int &, const int &);

template <ggml_type type>
static constexpr __device__ moe_cache_vec_dot_q_cuda_t
moe_cache_get_vec_dot_q_cuda() {
    if constexpr (type == GGML_TYPE_Q4_K) {
        return vec_dot_q4_K_q8_1;
    } else if constexpr (type == GGML_TYPE_Q5_K) {
        return vec_dot_q5_K_q8_1;
    } else if constexpr (type == GGML_TYPE_Q6_K) {
        return vec_dot_q6_K_q8_1;
    } else {
        return nullptr;
    }
}

template <ggml_type type>
static constexpr __host__ __device__ int moe_cache_get_vdr_mmvq() {
    if constexpr (type == GGML_TYPE_Q4_K) {
        return VDR_Q4_K_Q8_1_MMVQ;
    } else if constexpr (type == GGML_TYPE_Q5_K) {
        return VDR_Q5_K_Q8_1_MMVQ;
    } else if constexpr (type == GGML_TYPE_Q6_K) {
        return VDR_Q6_K_Q8_1_MMVQ;
    } else {
        return 1;
    }
}

// Single-token MUL_MAT_ID with one stable pointer-table lookup per routed
// expert.  On Ampere the regular MMVQ ncols=1 path uses four warps; mirror
// that layout so the only material difference is removal of the D2D weight
// copy for cache hits.
template <ggml_type type>
__launch_bounds__(4 * ggml_cuda_get_physical_warp_size(), 1)
static __global__ void moe_cache_direct_mmv_kernel(
        const void * const * expert_ptrs,
        const block_q8_1 * act_q8,
        const int32_t * ids,
        float * dst,
        uint32_t n_in,
        uint32_t n_out,
        uint32_t n_expert,
        uint32_t n_hits,
        uint32_t act_rows,
        uint32_t stride_row_x,
        uint32_t stride_channel_y) {
    constexpr int qk = ggml_cuda_type_traits<type>::qk;
    constexpr int qi = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = moe_cache_get_vdr_mmvq<type>();
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps = 4;
    constexpr int blocks_per_iter = vdr * nwarps * warp_size / qi;
    constexpr moe_cache_vec_dot_q_cuda_t vec_dot_q_cuda =
        moe_cache_get_vec_dot_q_cuda<type>();

    const uint32_t row = blockIdx.x;
    const uint32_t channel = blockIdx.y;
    if (row >= n_out || channel >= n_hits || act_rows == 0) {
        return;
    }

    const int32_t expert = ids[channel];
    if (expert < 0 || (uint32_t)expert >= n_expert) {
        return;
    }
    const void * vx = expert_ptrs[expert];
    if (!vx) {
        return;
    }

    ggml_cuda_pdl_sync();
    const uint32_t act_row = act_rows == 1 ? 0 : channel % act_rows;
    const block_q8_1 * y = act_q8 + (size_t)act_row * stride_channel_y;
    const int blocks_per_row_x = n_in / qk;
    const int tid = warp_size * threadIdx.y + threadIdx.x;
    const int kbx_offset = (int)row * (int)stride_row_x;

    float tmp = 0.0f;
    for (int kbx = tid / (qi / vdr);
         kbx < blocks_per_row_x;
         kbx += blocks_per_iter) {
        const int kby = kbx * (qk / QK8_1);
        const int kqs = vdr * (tid % (qi / vdr));
        tmp += vec_dot_q_cuda(vx, &y[kby], kbx_offset + kbx, kqs);
    }

    __shared__ float tmp_shared[nwarps - 1][warp_size];
    if (threadIdx.y > 0) {
        tmp_shared[threadIdx.y - 1][threadIdx.x] = tmp;
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }

#pragma unroll
    for (int warp = 0; warp < nwarps - 1; ++warp) {
        tmp += tmp_shared[warp][threadIdx.x];
    }
    tmp = warp_reduce_sum<warp_size>(tmp);
    ggml_cuda_pdl_lc();
    if (threadIdx.x == 0) {
        dst[(size_t)channel * n_out + row] = tmp;
    }
}

// Multi-token MUL_MAT_ID variant for speculative target verification.
// blockIdx.y selects the routed slot and threadIdx.y selects the token;
// each token is handled by an independent warp, matching the native MoE MMVQ
// execution topology while replacing expert-base arithmetic with a pointer table.
template <ggml_type type, int rows_per_block = 2>
__launch_bounds__(MMVQ_MAX_BATCH_SIZE * ggml_cuda_get_physical_warp_size(), 1)
static __global__ void moe_cache_direct_mmv_moe_kernel(
        const void * const * expert_ptrs,
        const block_q8_1 * act_q8,
        const int32_t * ids,
        float * dst,
        uint32_t n_in,
        uint32_t n_out,
        uint32_t n_expert,
        uint32_t n_expert_used,
        uint32_t n_tokens,
        uint32_t n_act_channels,
        uint32_t stride_row_x,
        uint32_t stride_channel_y,
        uint32_t stride_token_y,
        uint32_t ids_stride,
        uint32_t stride_channel_dst,
        uint32_t stride_token_dst) {
    constexpr int qk = ggml_cuda_type_traits<type>::qk;
    constexpr int qi = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = moe_cache_get_vdr_mmvq<type>();
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int blocks_per_iter = vdr * warp_size / qi;
    constexpr moe_cache_vec_dot_q_cuda_t vec_dot_q_cuda =
        moe_cache_get_vec_dot_q_cuda<type>();

    const uint32_t token = threadIdx.y;
    const uint32_t channel = blockIdx.y;
    const uint32_t row0 = rows_per_block * blockIdx.x;
    if (token >= n_tokens || channel >= n_expert_used || row0 >= n_out) {
        return;
    }

    const int32_t expert = ids[channel + token * ids_stride];
    if (expert < 0 || (uint32_t)expert >= n_expert) {
        return;
    }
    const void * vx = expert_ptrs[expert];
    if (!vx) {
        return;
    }

    ggml_cuda_pdl_sync();
    const uint32_t channel_y = channel % n_act_channels;
    const block_q8_1 * y = act_q8 +
        (size_t)token * stride_token_y +
        (size_t)channel_y * stride_channel_y;
    const int blocks_per_row_x = n_in / qk;

    float tmp[rows_per_block] = {0.0f};
    for (int kbx = threadIdx.x / (qi / vdr);
         kbx < blocks_per_row_x;
         kbx += blocks_per_iter) {
        const int kby = kbx * (qk / QK8_1);
        const int kqs = vdr * (threadIdx.x % (qi / vdr));
#pragma unroll
        for (int i = 0; i < rows_per_block; ++i) {
            if (row0 + (uint32_t)i < n_out) {
                tmp[i] += vec_dot_q_cuda(
                    vx, &y[kby], (int)((row0 + i) * stride_row_x) + kbx, kqs);
            }
        }
    }

    ggml_cuda_pdl_lc();
#pragma unroll
    for (int i = 0; i < rows_per_block; ++i) {
        tmp[i] = warp_reduce_sum<warp_size>(tmp[i]);
    }
    if (threadIdx.x < rows_per_block && row0 + threadIdx.x < n_out) {
        dst[(size_t)token * stride_token_dst +
            (size_t)channel * stride_channel_dst +
            row0 + threadIdx.x] = tmp[threadIdx.x];
    }
}

template <ggml_type type>
static void moe_cache_direct_mmv_moe_launch(
        const void * const * expert_ptrs,
        const char * act_q8,
        const int32_t * ids,
        float * dst,
        int64_t n_in,
        int64_t n_out,
        int64_t n_expert,
        int64_t n_expert_used,
        int64_t n_tokens,
        int64_t n_act_channels,
        int64_t ids_stride,
        int64_t stride_channel_dst,
        int64_t stride_token_dst,
        cudaStream_t stream) {
    constexpr int rows_per_block = 2;
    const size_t ts0 = ggml_type_size(type);
    GGML_ASSERT(n_in > 0 && n_out > 0 && n_expert > 0 &&
                n_expert_used > 0 && n_tokens > 1 &&
                n_tokens <= MMVQ_MAX_BATCH_SIZE && n_act_channels > 0);
    GGML_ASSERT(n_in % ggml_blck_size(type) == 0);
    const int64_t ne10_padded = GGML_PAD(n_in, MATRIX_ROW_PADDING);
    const uint32_t stride_row_x =
        (uint32_t)(ggml_row_size(type, n_in) / ts0);
    const uint32_t stride_channel_y =
        (uint32_t)(ne10_padded / QK8_1);
    const uint32_t stride_token_y =
        (uint32_t)(n_act_channels * stride_channel_y);
    const int device = ggml_cuda_get_device();
    const int warp_size = ggml_cuda_info().devices[device].warp_size;
    const dim3 grid(
        (unsigned int)((n_out + rows_per_block - 1) / rows_per_block),
        (unsigned int)n_expert_used, 1);
    const dim3 block((unsigned int)warp_size, (unsigned int)n_tokens, 1);
    moe_cache_direct_mmv_moe_kernel<type, rows_per_block>
        <<<grid, block, 0, stream>>>(
            expert_ptrs, (const block_q8_1 *)act_q8, ids, dst,
            (uint32_t)n_in, (uint32_t)n_out, (uint32_t)n_expert,
            (uint32_t)n_expert_used, (uint32_t)n_tokens,
            (uint32_t)n_act_channels, stride_row_x,
            stride_channel_y, stride_token_y, (uint32_t)ids_stride,
            (uint32_t)stride_channel_dst, (uint32_t)stride_token_dst);
}

template <ggml_type type>
static void moe_cache_direct_mmv_launch(
        const void * const * expert_ptrs,
        const char * act_q8,
        const int32_t * ids,
        float * dst,
        int64_t n_in,
        int64_t n_out,
        int64_t n_expert,
        int64_t n_hits,
        int64_t act_rows,
        cudaStream_t stream) {
    const size_t ts0 = ggml_type_size(type);
    GGML_ASSERT(n_in > 0 && n_out > 0 && n_expert > 0 && n_hits > 0);
    GGML_ASSERT(n_in % ggml_blck_size(type) == 0);
    const int64_t ne10_padded = GGML_PAD(n_in, MATRIX_ROW_PADDING);
    const uint32_t stride_row_x = (uint32_t)(ggml_row_size(type, n_in) / ts0);
    const uint32_t stride_channel_y = (uint32_t)(ne10_padded / QK8_1);
    const int device = ggml_cuda_get_device();
    const int warp_size = ggml_cuda_info().devices[device].warp_size;
    const dim3 grid((unsigned int)n_out, (unsigned int)n_hits, 1);
    const dim3 block((unsigned int)warp_size, 4, 1);
    moe_cache_direct_mmv_kernel<type><<<grid, block, 0, stream>>>(
        expert_ptrs, (const block_q8_1 *)act_q8, ids, dst,
        (uint32_t)n_in, (uint32_t)n_out, (uint32_t)n_expert,
        (uint32_t)n_hits, (uint32_t)act_rows,
        stride_row_x, stride_channel_y);
}

static void moe_cache_direct_mmv_ptrs(
        const void * const * expert_ptrs,
        ggml_type type,
        const char * act_q8,
        const int32_t * ids,
        float * dst,
        int64_t n_in,
        int64_t n_out,
        int64_t n_expert,
        int64_t n_hits,
        int64_t act_rows,
        cudaStream_t stream) {
#define MOE_CACHE_DIRECT_CASE(type_name) \
    case type_name: \
        moe_cache_direct_mmv_launch<type_name>( \
            expert_ptrs, act_q8, ids, dst, n_in, n_out, n_expert, \
            n_hits, act_rows, stream); \
        break
    switch (type) {
        MOE_CACHE_DIRECT_CASE(GGML_TYPE_Q4_K);
        MOE_CACHE_DIRECT_CASE(GGML_TYPE_Q5_K);
        MOE_CACHE_DIRECT_CASE(GGML_TYPE_Q6_K);
        default:
            GGML_ABORT("unsupported scheduler-direct MoE type");
    }
#undef MOE_CACHE_DIRECT_CASE
}

static void moe_cache_direct_mmv_moe_ptrs(
        const void * const * expert_ptrs,
        ggml_type type,
        const char * act_q8,
        const int32_t * ids,
        float * dst,
        int64_t n_in,
        int64_t n_out,
        int64_t n_expert,
        int64_t n_expert_used,
        int64_t n_tokens,
        int64_t n_act_channels,
        int64_t ids_stride,
        int64_t stride_channel_dst,
        int64_t stride_token_dst,
        cudaStream_t stream) {
#define MOE_CACHE_DIRECT_MOE_CASE(type_name) \
    case type_name: \
        moe_cache_direct_mmv_moe_launch<type_name>( \
            expert_ptrs, act_q8, ids, dst, n_in, n_out, n_expert, \
            n_expert_used, n_tokens, n_act_channels, ids_stride, \
            stride_channel_dst, stride_token_dst, stream); \
        break
    switch (type) {
        MOE_CACHE_DIRECT_MOE_CASE(GGML_TYPE_Q4_K);
        MOE_CACHE_DIRECT_MOE_CASE(GGML_TYPE_Q5_K);
        MOE_CACHE_DIRECT_MOE_CASE(GGML_TYPE_Q6_K);
        default:
            GGML_ABORT("unsupported scheduler-direct multi-token MoE type");
    }
#undef MOE_CACHE_DIRECT_MOE_CASE
}



enum class moe_cache_slot_state : uint8_t {
    free,
    copying,
    valid,
};

struct moe_cache_key {
    const void * tensor = nullptr;
    int32_t expert = -1;

    bool operator==(const moe_cache_key & other) const {
        return tensor == other.tensor && expert == other.expert;
    }
};

struct moe_cache_key_hash {
    size_t operator()(const moe_cache_key & key) const {
        uint64_t value = (uint64_t)(uintptr_t)key.tensor;
        value ^= value >> 33;
        value *= 0xff51afd7ed558ccdULL;
        value ^= (uint64_t)(uint32_t)key.expert * 0x9e3779b97f4a7c15ULL;
        value ^= value >> 29;
        return (size_t)value;
    }
};

struct moe_cache_device;

struct moe_cache_slot {
    moe_cache_key key;
    uint64_t generation = 0;
    int prev = -1;
    int next = -1;
    int readers = 0;
    moe_cache_slot_state state = moe_cache_slot_state::free;
};

struct moe_cache_pool {
    moe_cache_device * owner = nullptr;
    size_t expert_size = 0;
    int wtype = -1;
    char * slab = nullptr;
    int n_slots = 0;
    bool covers_all_entries = false;

    std::vector<moe_cache_slot> slots;
    std::vector<int> free_slots;
    std::unordered_map<moe_cache_key, int, moe_cache_key_hash> map;
    int lru_head = -1;
    int lru_tail = -1;
};

struct moe_cache_shape {
    size_t expert_size = 0;
    int wtype = -1;
    uint64_t n_entries = 0;
    int64_t n_tensors = 0;
    int pool = -1;
    bool finished = false;
};

struct moe_cache_seen_tensor {
    size_t bytes = 0;
    size_t expert_size = 0;
    int wtype = -1;
    int64_t n_expert = 0;
};

struct moe_cache_host_mapping {
    const char * device_base = nullptr;
    size_t bytes = 0;
    bool owned_registration = false;
};

struct moe_cache_job {
    int pool = -1;
    int slot = -1;
    uint64_t generation = 0;
    moe_cache_key key;
    const void * source = nullptr;
    size_t bytes = 0;
};

struct moe_cache_demand {
    uint16_t count = 0;
    size_t expert_size = 0;
};

struct moe_cache_scheduler_pin {
    moe_cache_pool * pool = nullptr;
    int slot = -1;
};

struct moe_cache_scheduler_pending {
    cudaEvent_t event = nullptr;
    moe_cache_scheduler_pin pins[moe_cache_node_rows_max];
    int n_pins = 0;
    const void * host_base = nullptr;
};


struct moe_cache_direct_route_key {
    const void * dst_base = nullptr;
    int64_t n_expert = 0;
    size_t expert_size = 0;
    int wtype = -1;

    bool operator==(const moe_cache_direct_route_key & other) const {
        return dst_base == other.dst_base && n_expert == other.n_expert &&
               expert_size == other.expert_size && wtype == other.wtype;
    }
};

struct moe_cache_direct_route_key_hash {
    size_t operator()(const moe_cache_direct_route_key & key) const {
        size_t h = std::hash<const void *>{}(key.dst_base);
        h ^= std::hash<int64_t>{}(key.n_expert) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<size_t>{}(key.expert_size) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(key.wtype) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

struct moe_cache_direct_route {
    const void ** device_pointers = nullptr;
    const void ** host_pointers = nullptr;
    // V1.5.1: the scheduler work buffer is allocator-owned and may be reused
    // for different model tensors. Route identity is therefore only the stable
    // dst/pointer-table identity; source-specific async metadata lives in a
    // separate per-device snapshot ring below.
};

struct moe_cache_source_residency_key {
    const void * host_base = nullptr;
    int64_t n_expert = 0;
    size_t expert_size = 0;
    int wtype = -1;

    bool operator==(const moe_cache_source_residency_key & other) const {
        return host_base == other.host_base && n_expert == other.n_expert &&
               expert_size == other.expert_size && wtype == other.wtype;
    }
};

struct moe_cache_source_residency_key_hash {
    size_t operator()(const moe_cache_source_residency_key & key) const {
        size_t h = std::hash<const void *>{}(key.host_base);
        h ^= std::hash<int64_t>{}(key.n_expert) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<size_t>{}(key.expert_size) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(key.wtype) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

struct moe_cache_source_residency {
    int pool_index = -1;
    std::vector<const void *> resident;
};

struct moe_cache_gpu_ids_source {
    const void * host_base = nullptr;
    size_t tensor_size = 0;
    size_t expert_size = 0;
    int wtype = -1;
    int64_t n_expert = 0;
};

struct moe_cache_gpu_ids_snapshot {
    int32_t * ids = nullptr;
    cudaEvent_t event = nullptr;
    int ids_count = 0;
    const void * ids_group = nullptr;
    std::vector<moe_cache_gpu_ids_source> sources;
    bool valid = false;
};

static constexpr int moe_cache_gpu_ids_snapshot_count = 256;

struct moe_cache_direct_epoch {
    cudaStream_t stream = nullptr;
    std::vector<moe_cache_scheduler_pin> pins;
    std::vector<const void *> host_bases;
    bool gpu_ids_guard = false;
};

struct moe_cache_direct_pending {
    cudaEvent_t event = nullptr;
    std::vector<moe_cache_scheduler_pin> pins;
    std::vector<const void *> host_bases;
    bool gpu_ids_guard = false;
};

struct moe_cache_config {
    bool enabled = true;
    bool automatic = true;
    size_t budget_mb = 0;
    // Final validated default for the GPU-ID/direct-resident topology.
    size_t reserve_mb = 512;
    size_t minimum_slab_bytes = moe_cache_slab_bytes_auto_min;
    size_t min_expert_bytes = moe_cache_expert_bytes_pre_ampere_min;
    bool min_expert_explicit = false;
    int max_batch = moe_cache_batch_max;
    bool max_batch_explicit = false;
    int inserts_per_plan = 8;
    int admit_after = 2;
    bool admit_after_explicit = false;
    int readmit_after = 8;
    int queue_max = 256;
    size_t queue_mb = 512;
    int stats_every = 0;
    int max_devices = INT_MAX;
    int min_compute_capability = moe_cache_cc_forced_min;
    bool serial_fill = true;
    bool serial_fill_explicit = false;
    bool force_dedicated_mmv = false;
    int overlap_cpu_rows = -1;
    bool overlap_cpu_rows_explicit = false;

    // Experimental: execute this percentage of genuine persistent-cache
    // misses on the GPU. The selected weights are staged into a small
    // transient VRAM slab and are not inserted into the persistent cache.
    int gpu_miss_percent = 0;

    // V6: one GPU staging kernel reads a whole miss batch from mapped host
    // memory. Set to false to retain the V5.1 per-miss cudaMemcpyAsync path.
    // V7 direct-host MMV bypasses both staging paths when mapping succeeds.
    bool gpu_miss_copy_kernel = true;

    // V7: execute selected misses directly from mapped host RAM in the
    // existing CUDA MMV/MMV-fused kernels. This removes the transient
    // host->VRAM staging pass entirely. If mapping fails, fall back to V6.
    bool gpu_miss_direct = true;

    // Final production topology selected by the RTX 3060 / Qwen3.6 validation:
    // keep MUL_MAT_ID on CUDA, execute resident experts directly from the
    // LeLoch slab, and keep routing IDs GPU-resident with one-scope-late
    // adaptive metadata. These are architecture defaults, not experiment knobs.
    bool scheduler_copy = true;
    bool scheduler_direct = true;
    bool scheduler_gpu_ids = true;

    std::string fail_stage;
};

struct moe_cache_session;

struct moe_cache_scratch {
    size_t input = 0;
    size_t q8 = 0;
    size_t out = 0;
};

struct moe_cache_device {
    moe_cache_device(int logical, int physical) : logical(logical), physical(physical) {}

    int logical;
    int physical;
    std::atomic<bool> dead{false};
    std::mutex dispatch_mu;

    std::vector<std::unique_ptr<moe_cache_pool>> pools;
    std::vector<moe_cache_shape> shapes;
    std::unordered_map<const void *, moe_cache_seen_tensor> seen_tensors;
    std::unordered_map<const void *, moe_cache_host_mapping> host_mappings;
    std::unordered_map<moe_cache_key, moe_cache_demand, moe_cache_key_hash> demand_count;
    size_t visits_since_new_tensor = 0;
    bool budget_ready = false;
    bool budget_registered = false;
    bool budget_claimed = false;
    size_t budget_reserve_bytes = 0;
    size_t budget_limit = 0;
    size_t coordinator_allocated_bytes = 0;
    size_t allocated_bytes = 0;
    moe_cache_scratch scratch_reserve;

    std::deque<moe_cache_job> queue;
    size_t queued_bytes = 0;
    bool worker_started = false;
    bool inflight = false;
    const void * inflight_source = nullptr;
    size_t inflight_bytes = 0;
    std::thread worker;

    cudaStream_t compute_stream = nullptr;
    char * h_input = nullptr;
    char * d_input = nullptr;
    size_t h_input_cap = 0;
    size_t d_input_cap = 0;
    void * d_act_q8 = nullptr;
    size_t act_q8_cap = 0;
    float * d_out = nullptr;
    size_t d_out_cap = 0;
    float * h_out = nullptr;
    size_t h_out_cap = 0;

    // Experimental transient slab for GPU-executed cache misses. This is
    // intentionally outside the persistent cache residency/accounting.
    char * d_gpu_miss = nullptr;
    size_t d_gpu_miss_cap = 0;

    // Scheduler-copy pins remain live until the CUDA backend stream has
    // consumed the source cache slots. Events are recycled after completion.
    std::deque<moe_cache_scheduler_pending> scheduler_pending;
    std::vector<cudaEvent_t> scheduler_free_events;

    // V1.4 direct-resident decode state. Pointer tables are stable for the
    // lifetime of a scheduler work tensor, making CUDA Graph replay safe.
    std::unordered_map<
        moe_cache_direct_route_key, moe_cache_direct_route,
        moe_cache_direct_route_key_hash> scheduler_direct_routes;
    std::unordered_map<
        moe_cache_source_residency_key, moe_cache_source_residency,
        moe_cache_source_residency_key_hash> scheduler_gpu_ids_source_cache;
    moe_cache_direct_epoch scheduler_direct_epoch;
    std::deque<moe_cache_direct_pending> scheduler_direct_pending;
    std::vector<moe_cache_gpu_ids_snapshot> scheduler_gpu_ids_pending;
    const void * scheduler_gpu_ids_current_group = nullptr;
    moe_cache_gpu_ids_snapshot * scheduler_gpu_ids_current_snapshot = nullptr;

    // hits and misses count tensor-expert probes; fusion counters count row pairs.
    long long hits = 0;
    long long misses = 0;
    long long inserts = 0;
    long long fills = 0;
    long long fill_failures = 0;
    long long evictions = 0;
    long long insert_skips = 0;
    long long admission_skips = 0;
    long long dispatch_failures = 0;
    long long collect_failures = 0;
    long long activation_dedup = 0;
    long long overlap_rows = 0;
    long long gpu_miss_rows = 0;
    unsigned long long gpu_miss_bytes = 0;
    long long gpu_miss_fused_rows = 0;
    long long gpu_miss_copy_kernel_calls = 0;
    long long gpu_miss_copy_kernel_descs = 0;
    long long gpu_miss_copy_fallbacks = 0;
    long long gpu_miss_map_failures = 0;
    long long gpu_miss_direct_calls = 0;
    long long gpu_miss_direct_rows = 0;
    long long gpu_miss_direct_fused_rows = 0;
    long long scheduler_copy_calls = 0;
    long long scheduler_copy_hits = 0;
    long long scheduler_copy_misses = 0;
    unsigned long long scheduler_copy_bytes = 0;
    long long scheduler_direct_prepares = 0;
    long long scheduler_direct_executes = 0;
    long long scheduler_direct_hits = 0;
    long long scheduler_direct_misses = 0;
    unsigned long long scheduler_direct_miss_bytes = 0;
    std::atomic<int> scheduler_gpu_ids_guards{0};
    long long scheduler_gpu_ids_prepares = 0;
    long long scheduler_gpu_ids_metadata = 0;
    long long scheduler_gpu_ids_snapshots = 0;
    long long fused_rows = 0;
    long long fused_candidates = 0;
    long long pair_both = 0;
    long long pair_up_only = 0;
    long long pair_gate_only = 0;
    long long pair_neither = 0;
    long long fused_attempts = 0;
    long long fused_nodes = 0;
    long long nodes = 0;
    long long collect_calls = 0;
    std::atomic<int> error_logs{0};
};

struct moe_cache_session {
    moe_cache_config config;
    std::vector<std::unique_ptr<moe_cache_device>> devices;
    std::unordered_map<int, int> layer_devices;
    std::unordered_map<const void *, int> tensor_devices;

    std::mutex mu;
    std::mutex fill_mu;
    std::condition_variable cv;
    std::condition_variable idle_cv;
    std::atomic<bool> stopping{false};
    std::atomic<bool> dormant{false};
    std::atomic<bool> config_announced{false};
    std::atomic<bool> enabled_announced{false};
    std::atomic<bool> batch_bypass_announced{false};
    std::atomic<bool> row_bypass_announced{false};
    int active_scopes = 0;
    int active_nodes = 0;
    struct active_source {
        size_t bytes = 0;
        int references = 0;
    };
    std::unordered_map<const void *, active_source> active_sources;
};

struct moe_cache_pin {
    moe_cache_pool * pool = nullptr;
    int slot = -1;
};

struct moe_cache_node {
    moe_cache_session * session = nullptr;
    moe_cache_device * device = nullptr;
    moe_cache_pool * pool = nullptr;
    int pool_index = -1;
    const void * host_base = nullptr;
    const void * host_base2 = nullptr;
    size_t expert_size = 0;
    int64_t n_in = 0;
    int64_t n_out = 0;
    int64_t n_expert = 0;
    int64_t n_tokens = 0;
    int wtype = -1;
    std::unique_lock<std::mutex> dispatch_lock;
    moe_cache_pin pins[2 * moe_cache_node_rows_max];
    int n_pins = 0;

    // Number of real cache misses moved to transient GPU execution by plan().
    int gpu_miss_rows = 0;

    // dispatch() groups resident rows first and transient miss rows second.
    // collect() uses this map to restore the compact row order expected by CPU.
    int output_map[moe_cache_node_rows_max] = {};

    bool planned = false;
    bool dispatched = false;
};

static std::mutex g_registry_mu;
static std::unordered_set<moe_cache_session *> g_sessions;
static std::atomic<int> g_session_count{0};
struct moe_cache_physical_budget {
    int participants = 0;
    int prepared = 0;
    std::multiset<size_t> reserves;
    size_t outstanding_bytes = 0;
};
static std::mutex g_budget_mu;
static std::unordered_map<int, moe_cache_physical_budget> g_physical_budgets;
struct moe_cache_scope_frame {
    moe_cache_session * requested = nullptr;
    moe_cache_session * active = nullptr;
};
static thread_local std::vector<moe_cache_scope_frame> g_session_stack;
static thread_local int g_session_suppressed = 0;

static void moe_cache_scheduler_release_pending(
        moe_cache_session & session, moe_cache_device & device,
        moe_cache_scheduler_pending & pending);
static bool moe_cache_scheduler_reap_locked(
        moe_cache_session & session, moe_cache_device & device, bool wait);

static size_t moe_cache_trim_session(
        moe_cache_session & session, int physical_device);

static void moe_cache_budget_remove_participant(
        moe_cache_physical_budget & state, size_t reserve_bytes) {
    auto reserve = state.reserves.find(reserve_bytes);
    if (reserve != state.reserves.end()) {
        state.reserves.erase(reserve);
    }
    state.participants = std::max(state.participants - 1, 0);
}

static bool moe_cache_budget_register(moe_cache_session & session) {
    std::lock_guard<std::mutex> lock(g_budget_mu);
    std::vector<moe_cache_device *> registered;
    try {
        registered.reserve(session.devices.size());
        for (const auto & device_ptr : session.devices) {
            moe_cache_device & device = *device_ptr;
            moe_cache_physical_budget & state = g_physical_budgets[device.physical];
            const size_t reserve_bytes = session.config.reserve_mb << 20;
            state.reserves.insert(reserve_bytes);
            state.participants++;
            device.budget_registered = true;
            device.budget_reserve_bytes = reserve_bytes;
            registered.push_back(&device);
        }
        return true;
    } catch (...) {
        for (moe_cache_device * device : registered) {
            auto found = g_physical_budgets.find(device->physical);
            if (found != g_physical_budgets.end()) {
                moe_cache_budget_remove_participant(
                        found->second, device->budget_reserve_bytes);
                if (found->second.participants == 0) {
                    g_physical_budgets.erase(found);
                }
            }
            device->budget_registered = false;
            device->budget_reserve_bytes = 0;
        }
        for (auto found = g_physical_budgets.begin();
             found != g_physical_budgets.end();) {
            if (found->second.participants == 0) {
                found = g_physical_budgets.erase(found);
            } else {
                ++found;
            }
        }
        return false;
    }
}

static void moe_cache_budget_allocation(
        moe_cache_device & device, size_t bytes, bool allocated) {
    if (!device.budget_registered || !device.budget_claimed || bytes == 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_budget_mu);
    auto found = g_physical_budgets.find(device.physical);
    if (found == g_physical_budgets.end()) {
        return;
    }
    moe_cache_physical_budget & state = found->second;
    if (allocated) {
        const size_t remaining = device.budget_limit > device.coordinator_allocated_bytes
            ? device.budget_limit - device.coordinator_allocated_bytes : 0;
        const size_t tracked = std::min(bytes, remaining);
        device.coordinator_allocated_bytes += tracked;
        state.outstanding_bytes = tracked <= state.outstanding_bytes
            ? state.outstanding_bytes - tracked : 0;
    } else {
        const size_t tracked = std::min(bytes, device.coordinator_allocated_bytes);
        device.coordinator_allocated_bytes -= tracked;
        if (tracked <= SIZE_MAX - state.outstanding_bytes) {
            state.outstanding_bytes += tracked;
        } else {
            state.outstanding_bytes = SIZE_MAX;
        }
    }
}

static void moe_cache_budget_reallocation(
        moe_cache_device & device, size_t old_bytes, size_t new_bytes) {
    if (!device.budget_registered || !device.budget_claimed || old_bytes == new_bytes) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_budget_mu);
    auto found = g_physical_budgets.find(device.physical);
    if (found == g_physical_budgets.end()) {
        return;
    }

    moe_cache_physical_budget & state = found->second;
    const size_t tracked_old = std::min(old_bytes, device.coordinator_allocated_bytes);
    const size_t base = device.coordinator_allocated_bytes - tracked_old;
    const size_t remaining = device.budget_limit > base
        ? device.budget_limit - base : 0;
    const size_t tracked_new = std::min(new_bytes, remaining);
    device.coordinator_allocated_bytes = base + tracked_new;

    if (tracked_new >= tracked_old) {
        const size_t consumed = tracked_new - tracked_old;
        state.outstanding_bytes = consumed <= state.outstanding_bytes
            ? state.outstanding_bytes - consumed : 0;
    } else {
        const size_t released = tracked_old - tracked_new;
        state.outstanding_bytes = released <= SIZE_MAX - state.outstanding_bytes
            ? state.outstanding_bytes + released : SIZE_MAX;
    }
}

static void moe_cache_budget_unregister(moe_cache_device & device) {
    if (!device.budget_registered) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_budget_mu);
    auto found = g_physical_budgets.find(device.physical);
    if (found != g_physical_budgets.end()) {
        moe_cache_physical_budget & state = found->second;
        if (device.budget_claimed) {
            const size_t unallocated = device.budget_limit > device.coordinator_allocated_bytes
                ? device.budget_limit - device.coordinator_allocated_bytes : 0;
            state.outstanding_bytes = unallocated <= state.outstanding_bytes
                ? state.outstanding_bytes - unallocated : 0;
            state.prepared = std::max(state.prepared - 1, 0);
        }
        moe_cache_budget_remove_participant(
                state, device.budget_reserve_bytes);
        if (state.participants == 0) {
            g_physical_budgets.erase(found);
        }
    }
    device.budget_registered = false;
    device.budget_claimed = false;
    device.budget_ready = false;
    device.budget_reserve_bytes = 0;
    device.budget_limit = 0;
    device.coordinator_allocated_bytes = 0;
}

static size_t moe_cache_budget_claim(
        moe_cache_session & session, moe_cache_device & device,
        size_t free_memory, size_t & reserve_bytes) {
    std::lock_guard<std::mutex> lock(g_budget_mu);
    reserve_bytes = 0;
    auto found = g_physical_budgets.find(device.physical);
    if (found == g_physical_budgets.end() || !device.budget_registered) {
        return 0;
    }
    moe_cache_physical_budget & state = found->second;
    reserve_bytes = state.reserves.empty()
        ? 0 : *state.reserves.rbegin();
    const size_t available_after_reserve = free_memory > reserve_bytes
        ? free_memory - reserve_bytes : 0;
    const size_t unclaimed = available_after_reserve > state.outstanding_bytes
        ? available_after_reserve - state.outstanding_bytes : 0;
    const int remaining_participants = std::max(
            state.participants - state.prepared, 1);

    size_t claim = unclaimed / (size_t)remaining_participants;

    // Scheduler-copy POC: with target + MTP draft there are normally two
    // cache sessions on the same GPU. The stock coordinator divides the
    // first free-minus-reserve budget equally, even when the secondary
    // session only needs a few hundred MiB for complete coverage. That can
    // strand several GiB in the secondary grant while the main MoE cache is
    // left partial.
    //
    // Final scheduler budget policy: let the first cache participant claim
    // everything except one reserve-sized allowance for each participant that
    // has not prepared yet. This avoids stranding VRAM in the small MTP cache.
    if (session.config.scheduler_copy && remaining_participants > 1) {
        const size_t followers = (size_t)remaining_participants - 1;
        const size_t holdback = reserve_bytes > 0 &&
                followers <= SIZE_MAX / reserve_bytes
            ? followers * reserve_bytes
            : SIZE_MAX;

        if (unclaimed > holdback) {
            claim = unclaimed - holdback;
        }
    }

    if (session.config.budget_mb > 0) {
        claim = std::min(claim, session.config.budget_mb << 20);
    }
    state.prepared++;
    state.outstanding_bytes += claim;
    device.budget_claimed = true;
    return claim;
}

static bool moe_cache_env_i64(
        const char * name, int64_t min_value, int64_t max_value, int64_t & value) {
    const char * text = getenv(name);
    if (!text || !text[0]) {
        return false;
    }

    char * end = nullptr;
    errno = 0;
    const long long parsed = strtoll(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed < min_value || parsed > max_value) {
        MOE_CACHE_LOG("[moe-cache] ignoring invalid %s=%s\n", name, text);
        return false;
    }

    value = parsed;
    return true;
}

static int moe_cache_min_compute_capability(bool automatic) {
    int result = automatic ? moe_cache_cc_ampere : moe_cache_cc_forced_min;
    int64_t value = 0;
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_MIN_CC", 0, 999, value)) {
        result = (int)value;
    }
    return result;
}

static size_t moe_cache_default_min_expert_bytes(int compute_capability) {
    return compute_capability >= moe_cache_cc_ampere
        ? moe_cache_expert_bytes_ampere_min
        : moe_cache_expert_bytes_pre_ampere_min;
}

static void moe_cache_apply_mode_defaults(moe_cache_config & config) {
    config.minimum_slab_bytes = config.automatic
        ? moe_cache_slab_bytes_auto_min : 0;
    if (!config.min_expert_explicit) {
        config.min_expert_bytes = config.automatic
            ? moe_cache_expert_bytes_ampere_min
            : moe_cache_expert_bytes_pre_ampere_min;
    }
    if (!config.max_batch_explicit) {
        config.max_batch = moe_cache_batch_max;
    }
    if (!config.overlap_cpu_rows_explicit) {
        config.overlap_cpu_rows = -1;
    }
}

static moe_cache_config moe_cache_read_config() {
    moe_cache_config config;
    int64_t value = 0;
    bool mode_off = false;
    bool mode_valid = false;

    if (const char * mode = getenv("GGML_CUDA_MOE_CACHE_MODE")) {
        if (strcmp(mode, "auto") == 0) {
            config.automatic = true;
            mode_valid = true;
        } else if (strcmp(mode, "on") == 0) {
            config.automatic = false;
            mode_valid = true;
        } else if (strcmp(mode, "off") == 0) {
            config.enabled = false;
            mode_off = true;
            mode_valid = true;
        } else {
            MOE_CACHE_LOG("[moe-cache] ignoring invalid GGML_CUDA_MOE_CACHE_MODE=%s\n", mode);
        }
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE", 0, 1, value)) {
        config.enabled = value != 0 && !mode_off;
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_BUDGET_MB", 1, 1024 * 1024, value)) {
        config.budget_mb = (size_t)value;
        if (!mode_valid) {
            config.automatic = false;
        }
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_RESERVE_MB", 0, 1024 * 1024, value)) {
        config.reserve_mb = (size_t)value;
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_MIN_EXPERT_KB", 1, 1024 * 1024, value)) {
        config.min_expert_bytes = (size_t)value << 10;
        config.min_expert_explicit = true;
    }
    if (moe_cache_env_i64(
            "GGML_CUDA_MOE_CACHE_MAX_BATCH", 1, moe_cache_batch_max, value)) {
        config.max_batch = (int)value;
        config.max_batch_explicit = true;
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_INSERTS", 1, 1024, value)) {
        config.inserts_per_plan = (int)value;
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_ADMIT_AFTER", 1, 255, value)) {
        config.admit_after = (int)value;
        config.admit_after_explicit = true;
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_THROTTLE", 1, 1024, value)) {
        config.readmit_after = (int)value;
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_QUEUE", 1, 65536, value)) {
        config.queue_max = (int)value;
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_QUEUE_MB", 1, 1024 * 1024, value)) {
        config.queue_mb = (size_t)value;
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_STATS", 0, INT_MAX, value)) {
        config.stats_every = (int)value;
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_NDEV", 1, INT_MAX, value)) {
        config.max_devices = (int)value;
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_SERIAL_FILL", 0, 1, value)) {
        config.serial_fill = value != 0;
        config.serial_fill_explicit = true;
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_DEDICATED_MMV", 0, 1, value)) {
        config.force_dedicated_mmv = value != 0;
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_OVERLAP_CPU_ROWS", 0, 8, value)) {
        config.overlap_cpu_rows = (int)value;
        config.overlap_cpu_rows_explicit = true;
    }
    if (moe_cache_env_i64(
            "GGML_CUDA_MOE_CACHE_GPU_MISS_PERCENT", 0, 100, value)) {
        config.gpu_miss_percent = (int)value;
    }
    if (moe_cache_env_i64(
            "GGML_CUDA_MOE_CACHE_GPU_MISS_COPY_KERNEL", 0, 1, value)) {
        config.gpu_miss_copy_kernel = value != 0;
    }
    if (moe_cache_env_i64(
            "GGML_CUDA_MOE_CACHE_GPU_MISS_DIRECT", 0, 1, value)) {
        config.gpu_miss_direct = value != 0;
    }
    moe_cache_apply_mode_defaults(config);
    config.min_compute_capability = moe_cache_min_compute_capability(config.automatic);
    if (const char * fail = getenv("GGML_CUDA_MOE_CACHE_FAIL")) {
        config.fail_stage = fail;
    }

    return config;
}

static bool moe_cache_fail(const moe_cache_session & session, const char * stage) {
    const std::string & value = session.config.fail_stage;
    if (value.empty()) {
        return false;
    }
    if (value == "all" || value == stage) {
        return true;
    }

    size_t begin = 0;
    while (begin < value.size()) {
        size_t end = value.find(',', begin);
        if (end == std::string::npos) {
            end = value.size();
        }
        if (value.compare(begin, end - begin, stage) == 0) {
            return true;
        }
        begin = end + 1;
    }
    return false;
}

static bool moe_cache_ranges_overlap(
        const void * lhs, size_t lhs_size, const void * rhs, size_t rhs_size) {
    if (!lhs || !rhs || lhs_size == 0 || rhs_size == 0) {
        return false;
    }
    const uintptr_t l = (uintptr_t)lhs;
    const uintptr_t r = (uintptr_t)rhs;
    return (l <= r ? r - l < lhs_size : l - r < rhs_size);
}

static uint64_t moe_cache_name_hash(const char * text) {
    uint64_t hash = 0xcbf29ce484222325ULL;
    while (*text) {
        hash ^= (unsigned char)*text++;
        hash *= 0x100000001b3ULL;
    }
    return hash;
}

static bool moe_cache_layer_number(const char * name, int & layer) {
    const char * marker = strstr(name, "blk.");
    if (!marker) {
        return false;
    }

    const char * first = marker + 4;
    char * end = nullptr;
    errno = 0;
    const long parsed = strtol(first, &end, 10);
    if (errno != 0 || end == first || parsed < 0 || parsed > INT_MAX) {
        return false;
    }
    if (*end != '.' && *end != '\0') {
        return false;
    }
    layer = (int)parsed;
    return true;
}

static bool moe_cache_tensor_name_supported(const char * name) {
    return strstr(name, "_exps") || strstr(name, "_chexps");
}

static bool moe_cache_type_supported(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q2_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_NVFP4:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS:
            return true;
        default:
            return false;
    }
}

static int moe_cache_pool_index_locked(
        moe_cache_device & device, const moe_cache_pool & pool) {
    for (int index = 0; index < (int)device.pools.size(); ++index) {
        if (device.pools[index].get() == &pool) {
            return index;
        }
    }
    return -1;
}

static void moe_cache_source_cache_notify_locked(
        moe_cache_device & device, moe_cache_pool & pool,
        const moe_cache_key & key, const void * resident_pointer) {
    if (!key.tensor || key.expert < 0 ||
        device.scheduler_gpu_ids_source_cache.empty()) {
        return;
    }
    const int pool_index = moe_cache_pool_index_locked(device, pool);
    if (pool_index < 0) {
        return;
    }

    for (auto & entry : device.scheduler_gpu_ids_source_cache) {
        const moe_cache_source_residency_key & source_key = entry.first;
        moe_cache_source_residency & cached = entry.second;
        if (source_key.host_base != key.tensor ||
            source_key.expert_size != pool.expert_size ||
            source_key.wtype != pool.wtype ||
            cached.pool_index != pool_index ||
            key.expert >= source_key.n_expert ||
            key.expert >= (int32_t)cached.resident.size()) {
            continue;
        }
        cached.resident[(size_t)key.expert] = resident_pointer;
    }
}

static void moe_cache_lru_remove(moe_cache_pool & pool, int index) {
    moe_cache_slot & slot = pool.slots[index];
    if (slot.prev >= 0) {
        pool.slots[slot.prev].next = slot.next;
    } else {
        pool.lru_head = slot.next;
    }
    if (slot.next >= 0) {
        pool.slots[slot.next].prev = slot.prev;
    } else {
        pool.lru_tail = slot.prev;
    }
    slot.prev = -1;
    slot.next = -1;
}

static void moe_cache_lru_push_back(moe_cache_pool & pool, int index) {
    moe_cache_slot & slot = pool.slots[index];
    slot.prev = pool.lru_tail;
    slot.next = -1;
    if (pool.lru_tail >= 0) {
        pool.slots[pool.lru_tail].next = index;
    } else {
        pool.lru_head = index;
    }
    pool.lru_tail = index;
}

static void moe_cache_map_erase(moe_cache_pool & pool, int index) {
    moe_cache_slot & slot = pool.slots[index];
    auto it = pool.map.find(slot.key);
    if (it != pool.map.end() && it->second == index) {
        pool.map.erase(it);
    }
}

static void moe_cache_slot_reset(moe_cache_pool & pool, int index, bool add_to_free) {
    moe_cache_slot & slot = pool.slots[index];
    if (slot.state == moe_cache_slot_state::valid) {
        if (pool.owner) {
            moe_cache_source_cache_notify_locked(*pool.owner, pool, slot.key, nullptr);
        }
        moe_cache_lru_remove(pool, index);
    }
    moe_cache_map_erase(pool, index);
    slot.key = {};
    slot.generation++;
    slot.readers = 0;
    slot.state = moe_cache_slot_state::free;
    slot.prev = -1;
    slot.next = -1;
    if (add_to_free) {
        pool.free_slots.push_back(index);
    }
}

static bool moe_cache_cuda_ok(
        moe_cache_device & device, cudaError_t error, const char * operation, bool fatal) {
    if (error == cudaSuccess) {
        return true;
    }

    (void)cudaGetLastError();
    if (device.error_logs.fetch_add(1) < 8) {
        MOE_CACHE_LOG("[moe-cache] CUDA%d %s failed: %s\n",
                device.physical, operation, cudaGetErrorString(error));
    }
    if (fatal && error != cudaErrorMemoryAllocation) {
        device.dead.store(true);
    }
    return false;
}


static const char * moe_cache_map_host_source(
        moe_cache_device & device, const void * host_base, size_t bytes) {
    if (!host_base || bytes == 0) {
        return nullptr;
    }

    const auto cached = device.host_mappings.find(host_base);
    if (cached != device.host_mappings.end() && cached->second.bytes >= bytes) {
        return cached->second.device_base;
    }

    cudaPointerAttributes attributes = {};
    cudaError_t error = cudaPointerGetAttributes(&attributes, host_base);
    if (error == cudaSuccess && attributes.devicePointer) {
        try {
            device.host_mappings[host_base] = {
                (const char *)attributes.devicePointer, bytes, false};
        } catch (...) {
            return nullptr;
        }
        return (const char *)attributes.devicePointer;
    }
    (void)cudaGetLastError();

    unsigned int flags =
        cudaHostRegisterPortable |
        cudaHostRegisterMapped |
        cudaHostRegisterReadOnly;

    bool owned_registration = false;
    error = cudaHostRegister((void *)host_base, bytes, flags);
    if (error == cudaErrorNotSupported) {
        (void)cudaGetLastError();
        flags = cudaHostRegisterPortable | cudaHostRegisterMapped;
        error = cudaHostRegister((void *)host_base, bytes, flags);
    }

    if (error == cudaSuccess) {
        owned_registration = true;
    } else if (error == cudaErrorHostMemoryAlreadyRegistered) {
        (void)cudaGetLastError();
    } else {
        (void)cudaGetLastError();
        device.gpu_miss_map_failures++;
        return nullptr;
    }

    void * device_pointer = nullptr;
    error = cudaHostGetDevicePointer(&device_pointer, (void *)host_base, 0);
    if (error != cudaSuccess || !device_pointer) {
        (void)cudaGetLastError();
        if (owned_registration) {
            (void)cudaHostUnregister((void *)host_base);
        }
        device.gpu_miss_map_failures++;
        return nullptr;
    }

    try {
        device.host_mappings[host_base] = {
            (const char *)device_pointer, bytes, owned_registration};
    } catch (...) {
        if (owned_registration) {
            (void)cudaHostUnregister((void *)host_base);
        }
        return nullptr;
    }

    return (const char *)device_pointer;
}

static bool moe_cache_grow_device(
        moe_cache_device & device, void ** pointer, size_t & capacity,
        size_t required, const char * operation) {
    if (capacity >= required) {
        return true;
    }
    if (required > (std::numeric_limits<size_t>::max() - 256) / 2) {
        return false;
    }

    const size_t requested = required * 2 + 256;
    void * fresh = nullptr;
    cudaError_t error = cudaMalloc(&fresh, requested);
    if (error == cudaErrorMemoryAllocation && *pointer) {
        (void)cudaGetLastError();
        if (!moe_cache_cuda_ok(
                    device, cudaFree(*pointer), "scratch replacement free", true)) {
            return false;
        }
        moe_cache_budget_allocation(device, capacity, false);
        *pointer = nullptr;
        capacity = 0;
        error = cudaMalloc(&fresh, requested);
    }
    if (!moe_cache_cuda_ok(device, error, operation, false)) {
        return false;
    }
    if (*pointer) {
        const size_t old_capacity = capacity;
        if (!moe_cache_cuda_ok(
                    device, cudaFree(*pointer), "scratch replacement free", true)) {
            cudaFree(fresh);
            return false;
        }
        moe_cache_budget_reallocation(device, old_capacity, requested);
    } else {
        moe_cache_budget_allocation(device, requested, true);
    }
    *pointer = fresh;
    capacity = requested;
    return true;
}

static size_t moe_cache_growth_capacity(size_t capacity, size_t required) {
    if (capacity >= required) {
        return capacity;
    }
    if (required > (std::numeric_limits<size_t>::max() - 256) / 2) {
        return 0;
    }
    return required * 2 + 256;
}

static bool moe_cache_scratch_requirements(
        int64_t n_in, int64_t n_out, moe_cache_scratch & result) {
    constexpr size_t max_rows = moe_cache_node_rows_max;
    if (n_in <= 0 || n_out <= 0 ||
        n_in > INT64_MAX - (MATRIX_ROW_PADDING - 1)) {
        return false;
    }
    const int64_t padded_n_in =
        ((n_in + MATRIX_ROW_PADDING - 1) / MATRIX_ROW_PADDING) * MATRIX_ROW_PADDING;
    if ((uint64_t)n_in > SIZE_MAX / (max_rows * sizeof(float)) ||
        (uint64_t)n_out > SIZE_MAX / (max_rows * sizeof(float)) ||
        (uint64_t)(padded_n_in / QK8_1) >
            SIZE_MAX / (max_rows * sizeof(block_q8_1))) {
        return false;
    }

    const size_t ids_bytes = 3 * max_rows * sizeof(int32_t);
    const size_t act_bytes = max_rows * (size_t)n_in * sizeof(float);
    if (ids_bytes > SIZE_MAX - act_bytes) {
        return false;
    }
    const size_t capacities[] = {
        moe_cache_growth_capacity(0, ids_bytes + act_bytes),
        moe_cache_growth_capacity(
                0, max_rows * (size_t)(padded_n_in / QK8_1) * sizeof(block_q8_1)),
        moe_cache_growth_capacity(0, max_rows * (size_t)n_out * sizeof(float)),
    };
    for (size_t capacity : capacities) {
        if (capacity == 0) {
            return false;
        }
    }
    result = {capacities[0], capacities[1], capacities[2]};
    return true;
}

static size_t moe_cache_scratch_total(
        const moe_cache_scratch & current,
        const moe_cache_scratch * additional = nullptr) {
    const size_t capacities[] = {
        additional ? std::max(current.input, additional->input) : current.input,
        additional ? std::max(current.q8, additional->q8) : current.q8,
        additional ? std::max(current.out, additional->out) : current.out,
    };
    size_t total = 0;
    for (size_t capacity : capacities) {
        if (capacity > SIZE_MAX - total) {
            return SIZE_MAX;
        }
        total += capacity;
    }
    return total;
}

static int moe_cache_query_config(
        int automatic, size_t budget_mib, ggml_moe_cache_config * result) {
    if (!result) {
        return 0;
    }

    moe_cache_config config = moe_cache_read_config();
    if (automatic >= 0) {
        config.enabled = true;
        config.automatic = automatic != 0;
        moe_cache_apply_mode_defaults(config);
        config.min_compute_capability =
            moe_cache_min_compute_capability(config.automatic);
    }
    if (budget_mib > 0) {
        config.budget_mb = budget_mib;
    }
    if (!config.enabled || config.budget_mb > (SIZE_MAX >> 20) ||
        config.reserve_mb > (SIZE_MAX >> 20)) {
        return 0;
    }

    result->budget_bytes = config.budget_mb << 20;
    result->reserve_bytes = config.reserve_mb << 20;
    result->minimum_slab_bytes = config.minimum_slab_bytes;
    result->min_expert_bytes = config.min_expert_bytes;
    result->min_expert_explicit = config.min_expert_explicit;
    result->max_batch = config.max_batch;
    result->min_compute_capability = config.min_compute_capability;
    result->min_devices = config.automatic ? 2 : 1;
    result->overlap_cpu_rows = config.overlap_cpu_rows;
    return 1;
}

static int moe_cache_query_device(
        void * opaque, const ggml_moe_cache_config * config,
        ggml_moe_cache_device_caps * result) {
    if (!opaque || !config || !result || !ggml_moe_cache.owner) {
        return 0;
    }

    ggml_backend_dev_t device = (ggml_backend_dev_t)opaque;
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(device);
    if ((const void *)reg != ggml_moe_cache.owner) {
        return 0;
    }

    int logical = -1;
    const size_t count = ggml_backend_reg_dev_count(reg);
    for (size_t index = 0; index < count; index++) {
        if (ggml_backend_reg_dev_get(reg, index) == device) {
            logical = (int)index;
            break;
        }
    }
    const ggml_cuda_device_info & info = ggml_cuda_info();
    if (logical < 0 || logical >= info.device_count) {
        return 0;
    }

    const auto & cuda_device = info.devices[logical];
    if (cuda_device.cc < config->min_compute_capability) {
        return 0;
    }
    result->logical_device = logical;
    result->physical_device = cuda_device.physical_device;
    result->compute_capability = cuda_device.cc;
    result->min_expert_bytes = config->min_expert_explicit
        ? config->min_expert_bytes
        : moe_cache_default_min_expert_bytes(cuda_device.cc);
    return 1;
}

static int moe_cache_query_shape(
        int wtype, int64_t n_in, int64_t n_out, int64_t n_expert,
        size_t expert_size,
        ggml_moe_cache_shape_caps * result) {
    if (!result || n_in <= 0 || n_out <= 0 || n_expert <= 0 ||
        !moe_cache_type_supported((ggml_type)wtype)) {
        return 0;
    }

    const size_t row_size = ggml_row_size((ggml_type)wtype, n_in);
    if (row_size == 0 || (uint64_t)n_out > SIZE_MAX / row_size ||
        expert_size != (size_t)n_out * row_size ||
        expert_size > SIZE_MAX / moe_cache_pool_slots_min) {
        return 0;
    }

    moe_cache_scratch scratch;
    if (!moe_cache_scratch_requirements(n_in, n_out, scratch)) {
        return 0;
    }
    const size_t scratch_bytes = moe_cache_scratch_total(scratch);
    const size_t pool_bytes = expert_size * moe_cache_pool_slots_min;
    if (scratch_bytes == SIZE_MAX || pool_bytes > SIZE_MAX - scratch_bytes) {
        return 0;
    }
    result->scratch_bytes = scratch_bytes;
    result->pool_bytes = pool_bytes;
    result->minimum_bytes = scratch_bytes + pool_bytes;
    return 1;
}

static bool moe_cache_grow_host(
        moe_cache_device & device, void ** pointer, size_t & capacity,
        size_t required, const char * operation) {
    if (capacity >= required) {
        return true;
    }
    if (required > (std::numeric_limits<size_t>::max() - 256) / 2) {
        return false;
    }

    const size_t requested = required * 2 + 256;
    void * fresh = nullptr;
    if (!moe_cache_cuda_ok(device, cudaMallocHost(&fresh, requested), operation, false)) {
        return false;
    }
    if (*pointer) {
        cudaFreeHost(*pointer);
    }
    *pointer = fresh;
    capacity = requested;
    return true;
}

static void moe_cache_worker(moe_cache_session * session, moe_cache_device * device) {
    char * stage = nullptr;
    size_t stage_capacity = 0;
    cudaStream_t stream = nullptr;

    for (;;) {
        moe_cache_job job;
        {
            std::unique_lock<std::mutex> lock(session->mu);
            session->cv.wait(lock, [&] {
                return session->stopping || device->dead.load() ||
                    !device->queue.empty();
            });
            if ((session->stopping || device->dead.load()) &&
                device->queue.empty()) {
                break;
            }

            job = device->queue.front();
            device->queue.pop_front();
            device->queued_bytes = job.bytes <= device->queued_bytes
                ? device->queued_bytes - job.bytes : 0;
            device->inflight = true;
            device->inflight_source = job.source;
            device->inflight_bytes = job.bytes;
        }

        cudaError_t error = cudaSuccess;
        ggml_cuda_set_device(device->logical);

        if (device->dead.load() || moe_cache_fail(*session, "insert")) {
            error = cudaErrorUnknown;
        }
        if (error == cudaSuccess && !stream) {
            int least_priority = 0;
            int greatest_priority = 0;
            error = cudaDeviceGetStreamPriorityRange(
                    &least_priority, &greatest_priority);
            if (error == cudaSuccess) {
                error = cudaStreamCreateWithPriority(
                        &stream, cudaStreamNonBlocking, least_priority);
            } else {
                (void)cudaGetLastError();
                error = cudaStreamCreateWithFlags(
                        &stream, cudaStreamNonBlocking);
            }
        }
        if (error == cudaSuccess && stage_capacity < job.bytes) {
            char * fresh = nullptr;
            cudaError_t alloc_error = cudaMallocHost((void **)&fresh, job.bytes);
            if (alloc_error == cudaSuccess) {
                if (stage) {
                    cudaFreeHost(stage);
                }
                stage = fresh;
                stage_capacity = job.bytes;
            } else {
                (void)cudaGetLastError();
            }
        }

        moe_cache_pool * pool = nullptr;
        char * destination = nullptr;
        {
            std::lock_guard<std::mutex> lock(session->mu);
            if (job.pool >= 0 && job.pool < (int)device->pools.size()) {
                pool = device->pools[job.pool].get();
                if (pool->slab && job.slot >= 0 && job.slot < pool->n_slots) {
                    destination = pool->slab + (size_t)job.slot * pool->expert_size;
                }
            }
        }

        if (error == cudaSuccess && !destination) {
            error = cudaErrorInvalidValue;
        }
        {
            std::unique_lock<std::mutex> fill_lock(
                    session->fill_mu, std::defer_lock);
            if (session->config.serial_fill) {
                fill_lock.lock();
            }
            if (error == cudaSuccess && stage && stage_capacity >= job.bytes) {
                memcpy(stage, job.source, job.bytes);
                error = cudaMemcpyAsync(
                        destination, stage, job.bytes, cudaMemcpyHostToDevice, stream);
                if (error == cudaSuccess) {
                    error = cudaStreamSynchronize(stream);
                }
            } else if (error == cudaSuccess) {
                error = cudaMemcpy(
                        destination, job.source, job.bytes, cudaMemcpyHostToDevice);
            }
        }

        {
            std::lock_guard<std::mutex> lock(session->mu);
            device->inflight = false;
            device->inflight_source = nullptr;
            device->inflight_bytes = 0;

            if (pool && job.slot >= 0 && job.slot < pool->n_slots) {
                moe_cache_slot & slot = pool->slots[job.slot];
                if (slot.state == moe_cache_slot_state::copying &&
                    slot.generation == job.generation && slot.key == job.key) {
                    if (error == cudaSuccess) {
                        slot.state = moe_cache_slot_state::valid;
                        moe_cache_source_cache_notify_locked(
                            *device, *pool, slot.key, destination);
                        moe_cache_lru_push_back(*pool, job.slot);
                        device->demand_count.erase(job.key);
                        device->fills++;
                    } else {
                        moe_cache_slot_reset(*pool, job.slot, true);
                        device->fill_failures++;
                    }
                }
            }
            session->idle_cv.notify_all();
        }

        if (error != cudaSuccess) {
            moe_cache_cuda_ok(*device, error, "expert fill", true);
            moe_cache_trim_session(*session, device->physical);
        }
    }

    if (stream) {
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
    }
    if (stage) {
        cudaFreeHost(stage);
    }
}

static bool moe_cache_start_worker(
        moe_cache_session & session, moe_cache_device & device) {
    if (device.worker_started) {
        return true;
    }
    try {
        device.worker = std::thread(moe_cache_worker, &session, &device);
        device.worker_started = true;
        return true;
    } catch (...) {
        device.dead.store(true);
        MOE_CACHE_LOG("[moe-cache] CUDA%d failed to start fill worker\n", device.physical);
        return false;
    }
}

static int moe_cache_find_pool(
        const moe_cache_device & device, size_t expert_size, int wtype) {
    for (int index = 0; index < (int)device.pools.size(); index++) {
        const moe_cache_pool & pool = *device.pools[index];
        if (pool.expert_size == expert_size && pool.wtype == wtype) {
            return index;
        }
    }
    return -1;
}

static bool moe_cache_prepare_budget(
        moe_cache_session & session, moe_cache_device & device) {
    if (device.budget_ready) {
        return device.budget_limit > 0;
    }
    device.budget_ready = true;

    ggml_cuda_set_device(device.logical);
    size_t free_memory = 0;
    size_t total_memory = 0;
    cudaError_t error = cudaMemGetInfo(&free_memory, &total_memory);
    if (!moe_cache_cuda_ok(device, error, "memory query", false)) {
        device.dead.store(true);
        return false;
    }

    size_t reserve_bytes = 0;
    const size_t available = moe_cache_budget_claim(
            session, device, free_memory, reserve_bytes);
    device.budget_limit = available;

    if (session.config.budget_mb > 0) {
        MOE_CACHE_LOG("[moe-cache] CUDA%d capacity: cap=%zu MiB granted=%zu MiB free=%zu MiB reserve=%zu MiB\n",
                device.physical, session.config.budget_mb, available >> 20,
                free_memory >> 20, reserve_bytes >> 20);
    } else {
        MOE_CACHE_LOG("[moe-cache] CUDA%d capacity: policy=free-minus-reserve granted=%zu MiB free=%zu MiB reserve=%zu MiB\n",
                device.physical, available >> 20, free_memory >> 20,
                reserve_bytes >> 20);
    }
    return available > 0;
}

static bool moe_cache_allocate_pool(
        moe_cache_session & session, moe_cache_device & device,
        moe_cache_shape & shape, size_t budget) {
    if (shape.expert_size == 0 || shape.n_tensors <= 0 || shape.n_entries == 0) {
        return false;
    }
    shape.finished = true;

    size_t slots_by_budget = budget / shape.expert_size;
    size_t slot_count = std::min<size_t>(
            slots_by_budget,
            (size_t)std::min<uint64_t>(shape.n_entries, INT_MAX));
    const size_t type_size = ggml_type_size((ggml_type)shape.wtype);
    if (type_size == 0 || shape.expert_size % type_size != 0) {
        return false;
    }
    const size_t stride_blocks = shape.expert_size / type_size;
    if (stride_blocks == 0) {
        return false;
    }
    slot_count = std::min(slot_count, (size_t)INT_MAX / stride_blocks);
    if (slot_count > INT_MAX) {
        slot_count = INT_MAX;
    }
    if (slot_count < moe_cache_pool_slots_min) {
        return false;
    }

    ggml_cuda_set_device(device.logical);
    char * slab = nullptr;
    cudaError_t error = cudaSuccess;
    while (slot_count >= moe_cache_pool_slots_min) {
        if (moe_cache_fail(session, "slab")) {
            error = cudaErrorMemoryAllocation;
        } else {
            error = cudaMalloc((void **)&slab, slot_count * shape.expert_size);
        }
        if (error == cudaSuccess) {
            break;
        }
        (void)cudaGetLastError();
        slot_count /= 2;
    }
    if (error != cudaSuccess || !slab || slot_count < moe_cache_pool_slots_min) {
        MOE_CACHE_LOG("[moe-cache] CUDA%d skipped %zu KiB expert pool: allocation failed\n",
                device.physical, shape.expert_size >> 10);
        return false;
    }
    const size_t allocated = slot_count * shape.expert_size;
    moe_cache_budget_allocation(device, allocated, true);

    std::unique_ptr<moe_cache_pool> pool(new (std::nothrow) moe_cache_pool());
    if (!pool) {
        cudaFree(slab);
        moe_cache_budget_allocation(device, allocated, false);
        return false;
    }
    try {
        pool->owner = &device;
        pool->expert_size = shape.expert_size;
        pool->wtype = shape.wtype;
        pool->slab = slab;
        pool->n_slots = (int)slot_count;
        pool->covers_all_entries = (uint64_t)slot_count >= shape.n_entries;
        pool->slots.resize(slot_count);
        pool->free_slots.reserve(slot_count);
        pool->map.reserve(slot_count);
        for (int index = (int)slot_count - 1; index >= 0; index--) {
            pool->free_slots.push_back(index);
        }
        device.pools.push_back(std::move(pool));
    } catch (...) {
        cudaFree(slab);
        moe_cache_budget_allocation(device, allocated, false);
        return false;
    }

    shape.pool = (int)device.pools.size() - 1;
    device.allocated_bytes += allocated;

    if (!moe_cache_start_worker(session, device)) {
        cudaFree(device.pools.back()->slab);
        moe_cache_budget_allocation(device, allocated, false);
        device.pools.back()->slab = nullptr;
        device.pools.pop_back();
        device.allocated_bytes -= allocated;
        shape.pool = -1;
        return false;
    }

    MOE_CACHE_LOG("[moe-cache] CUDA%d pool[%d]: type=%s expert=%zu KiB slots=%zu entries=%llu coverage=%s total=%zu MiB\n",
            device.physical, shape.pool, ggml_type_name((ggml_type)shape.wtype),
            shape.expert_size >> 10, slot_count,
            (unsigned long long)shape.n_entries,
            device.pools.back()->covers_all_entries ? "complete" : "partial",
            allocated >> 20);
    bool expected = false;
    if (session.enabled_announced.compare_exchange_strong(expected, true)) {
        MOE_CACHE_LOG("[moe-cache] enabled: first pool allocated on CUDA%d\n", device.physical);
    }

    return true;
}

static void moe_cache_build_pending(
        moe_cache_session & session, moe_cache_device & device) {
    if (!moe_cache_prepare_budget(session, device)) {
        for (moe_cache_shape & shape : device.shapes) {
            if (shape.n_tensors > 0) {
                shape.finished = true;
            }
        }
        return;
    }

    const size_t scratch_reserve =
        moe_cache_scratch_total(device.scratch_reserve);
    const size_t slab_limit =
        scratch_reserve < device.budget_limit
            ? device.budget_limit - scratch_reserve : 0;
    size_t remaining = slab_limit > device.allocated_bytes
        ? slab_limit - device.allocated_bytes : 0;
    if (remaining == 0) {
        for (moe_cache_shape & shape : device.shapes) {
            if (!shape.finished && shape.n_tensors > 0) {
                shape.finished = true;
            }
        }
        return;
    }

    std::vector<moe_cache_shape *> pending;
    long double total_weight = 0.0;
    size_t minimum_remaining = 0;
    for (moe_cache_shape & shape : device.shapes) {
        if (!shape.finished && shape.n_tensors > 0) {
            pending.push_back(&shape);
            total_weight +=
                (long double)shape.expert_size * (long double)shape.n_tensors;
            const size_t minimum = shape.expert_size * moe_cache_pool_slots_min;
            minimum_remaining = minimum <= SIZE_MAX - minimum_remaining
                ? minimum_remaining + minimum : SIZE_MAX;
        }
    }
    std::sort(pending.begin(), pending.end(), [](const auto * lhs, const auto * rhs) {
        const long double lhs_weight =
            (long double)lhs->expert_size * (long double)lhs->n_tensors;
        const long double rhs_weight =
            (long double)rhs->expert_size * (long double)rhs->n_tensors;
        return lhs_weight > rhs_weight;
    });

    auto weighted_share = [](size_t available, long double weight, long double total) {
        if (available == 0 || weight <= 0.0 || total <= 0.0) {
            return (size_t)0;
        }
        const long double value = (long double)available * weight / total;
        return value >= (long double)available ? available : (size_t)value;
    };
    const bool complete_pools = minimum_remaining <= remaining;
    for (moe_cache_shape * shape : pending) {
        const long double weight =
            (long double)shape->expert_size * (long double)shape->n_tensors;
        const size_t minimum = shape->expert_size * moe_cache_pool_slots_min;
        size_t share = weighted_share(remaining, weight, total_weight);
        if (complete_pools) {
            const size_t extra = remaining - minimum_remaining;
            const size_t bonus = weighted_share(extra, weight, total_weight);
            share = minimum + bonus;
        } else if (share < minimum && remaining >= minimum) {
            share = minimum;
        }
        const size_t before = device.allocated_bytes;
        moe_cache_allocate_pool(session, device, *shape, share);
        const size_t consumed = device.allocated_bytes - before;
        remaining = consumed <= remaining ? remaining - consumed : 0;
        total_weight -= weight;
        if (complete_pools) {
            minimum_remaining -= minimum;
        }
    }
}

static int moe_cache_discover_pool(
        moe_cache_session & session, moe_cache_device & device,
        const void * host_base, size_t tensor_size, size_t expert_size,
        int wtype, int64_t n_expert) {
    int pool = moe_cache_find_pool(device, expert_size, wtype);
    if (pool >= 0 && !device.pools[pool]->covers_all_entries) {
        return pool;
    }
    if (pool >= 0 && device.seen_tensors.find(host_base) != device.seen_tensors.end()) {
        return pool;
    }

    moe_cache_shape * shape = nullptr;
    for (moe_cache_shape & candidate : device.shapes) {
        if (candidate.expert_size == expert_size && candidate.wtype == wtype) {
            shape = &candidate;
            break;
        }
    }
    if (!shape) {
        device.shapes.push_back({expert_size, wtype, 0, 0, -1, false});
        shape = &device.shapes.back();
        shape->pool = pool;
    }

    const bool first_visit = device.seen_tensors.emplace(
            host_base,
            moe_cache_seen_tensor{tensor_size, expert_size, wtype, n_expert}).second;
    if (first_visit) {
        if (shape->n_tensors == 0 && shape->pool < 0) {
            shape->finished = false;
        }
        const uint64_t entries = (uint64_t)n_expert;
        shape->n_entries = entries <= UINT64_MAX - shape->n_entries
            ? shape->n_entries + entries : UINT64_MAX;
        shape->n_tensors++;
        device.visits_since_new_tensor = 0;
    } else {
        if (device.visits_since_new_tensor < SIZE_MAX) {
            device.visits_since_new_tensor++;
        }
    }

    if (pool >= 0) {
        device.pools[pool]->covers_all_entries =
            (uint64_t)device.pools[pool]->n_slots >= shape->n_entries;
        return pool;
    }

    if (device.visits_since_new_tensor < device.seen_tensors.size()) {
        return -1;
    }

    moe_cache_build_pending(session, device);
    return moe_cache_find_pool(device, expert_size, wtype);
}

static void moe_cache_log_stats(moe_cache_device & device) {
    size_t used = 0;
    size_t slots = 0;
    for (const auto & pool_ptr : device.pools) {
        const moe_cache_pool & pool = *pool_ptr;
        slots += pool.n_slots;
        used += pool.n_slots - pool.free_slots.size();
    }
    const long long total = device.hits + device.misses;
    MOE_CACHE_LOG("[moe-cache] CUDA%d hits=%lld/%lld (%.1f%%) used=%zu/%zu enqueued=%lld filled=%lld fill-fail=%lld evictions=%lld skips=%lld admission=%lld queue=%zu jobs/%zu MiB dispatch-fail=%lld collect-fail=%lld act-dedup=%lld cpu-overlap=%lld gpu-miss=%lld/%llu MiB gpu-miss-fused=%lld direct=%lld/%lld/%lld copy-kernel=%lld/%lld fallback=%lld map-fail=%lld sched-copy=%lld/%lld/%lld/%llu MiB sched-direct=%lld/%lld/%lld/%lld/%llu MiB sched-gpuids=%lld/%lld/%lld fusion=%lld/%lld pairs=%lld/%lld/%lld/%lld fusion-attempts=%lld fusion-nodes=%lld\n",
            device.physical, device.hits, total,
            total ? 100.0 * (double)device.hits / (double)total : 0.0,
            used, slots, device.inserts, device.fills, device.fill_failures,
            device.evictions, device.insert_skips,
            device.admission_skips, device.queue.size(), device.queued_bytes >> 20,
            device.dispatch_failures, device.collect_failures,
            device.activation_dedup, device.overlap_rows,
            device.gpu_miss_rows, device.gpu_miss_bytes >> 20,
            device.gpu_miss_fused_rows,
            device.gpu_miss_direct_calls,
            device.gpu_miss_direct_rows,
            device.gpu_miss_direct_fused_rows,
            device.gpu_miss_copy_kernel_calls,
            device.gpu_miss_copy_kernel_descs,
            device.gpu_miss_copy_fallbacks,
            device.gpu_miss_map_failures,
            device.scheduler_copy_calls,
            device.scheduler_copy_hits,
            device.scheduler_copy_misses,
            device.scheduler_copy_bytes >> 20,
            device.scheduler_direct_prepares,
            device.scheduler_direct_executes,
            device.scheduler_direct_hits,
            device.scheduler_direct_misses,
            device.scheduler_direct_miss_bytes >> 20,
            device.scheduler_gpu_ids_prepares,
            device.scheduler_gpu_ids_metadata,
            device.scheduler_gpu_ids_snapshots,
            device.fused_rows, device.fused_candidates,
            device.pair_both, device.pair_up_only,
            device.pair_gate_only, device.pair_neither,
            device.fused_attempts,
            device.fused_nodes);
}

static void moe_cache_log_configuration(moe_cache_session & session) {
    bool expected = false;
    if (!session.config_announced.compare_exchange_strong(expected, true)) {
        return;
    }

    const std::string budget = session.config.budget_mb > 0
        ? std::to_string(session.config.budget_mb) + " MiB cap"
        : "free-minus-reserve";
    const std::string overlap_cpu_rows = session.config.overlap_cpu_rows < 0
        ? "auto" : std::to_string(session.config.overlap_cpu_rows);
    const std::string admission = session.config.admit_after_explicit
        ? std::to_string(session.config.admit_after) + "-fixed/" +
            std::to_string(session.config.readmit_after) + "-replace"
        : "1-complete/" + std::to_string(session.config.admit_after) +
            "-partial/" + std::to_string(session.config.readmit_after) + "-replace";
    MOE_CACHE_LOG("[moe-cache] configured: mode=%s devices=%zu budget=%s reserve=%zu MiB min-slab=%zu MiB min-expert=%zu KiB max-batch=%d admit=%s cpu-overlap=%s gpu-miss=%d%% miss-direct=%s miss-copy-kernel=%s topology=gpu-ids-direct source-cache=incremental queue=%d inserts=%d fills=%s\n",
            session.config.automatic ? "auto" : "on",
            session.devices.size(), budget.c_str(),
            session.config.reserve_mb,
            session.config.minimum_slab_bytes >> 20,
            session.config.min_expert_bytes >> 10,
            session.config.max_batch,
            admission.c_str(),
            overlap_cpu_rows.c_str(),
            session.config.gpu_miss_percent,
            session.config.gpu_miss_direct ? "on" : "off",
            session.config.gpu_miss_copy_kernel ? "on" : "off",
            session.config.queue_max,
            session.config.inserts_per_plan,
            session.config.serial_fill ? "serial" : "parallel");
}

static void * moe_cache_session_create(
        void * const * backends, int n_backends,
        const ggml_moe_cache_config * supplied_config) {
    try {
        moe_cache_config config = moe_cache_read_config();
        if (supplied_config) {
            constexpr size_t MiB = 1024 * 1024;
            if (supplied_config->budget_bytes % MiB != 0 ||
                supplied_config->reserve_bytes % MiB != 0 ||
                supplied_config->minimum_slab_bytes % MiB != 0 ||
                supplied_config->min_expert_bytes == 0 ||
                supplied_config->min_expert_explicit < 0 ||
                supplied_config->min_expert_explicit > 1 ||
                supplied_config->max_batch < 1 ||
                supplied_config->max_batch > moe_cache_batch_max ||
                supplied_config->min_devices < 1 ||
                supplied_config->min_compute_capability < 0 ||
                supplied_config->min_compute_capability > 999 ||
                supplied_config->overlap_cpu_rows < -1 ||
                supplied_config->overlap_cpu_rows > 8) {
                return nullptr;
            }
            config.enabled = true;
            config.automatic = supplied_config->min_devices > 1;
            config.budget_mb = supplied_config->budget_bytes / MiB;
            config.reserve_mb = supplied_config->reserve_bytes / MiB;
            config.minimum_slab_bytes = supplied_config->minimum_slab_bytes;
            config.min_expert_bytes = supplied_config->min_expert_bytes;
            config.min_expert_explicit = supplied_config->min_expert_explicit;
            config.max_batch = supplied_config->max_batch;
            config.min_compute_capability = supplied_config->min_compute_capability;
            config.overlap_cpu_rows = supplied_config->overlap_cpu_rows;
        }
        if (!config.enabled) {
            return nullptr;
        }

        std::unique_ptr<moe_cache_session> session(new (std::nothrow) moe_cache_session());
        if (!session) {
            return nullptr;
        }
        session->config = std::move(config);

        std::unordered_set<int> seen_devices;
        size_t default_min_expert_bytes = moe_cache_expert_bytes_ampere_min;
        int minimum_capability = INT_MAX;
        for (int index = 0; index < n_backends &&
                            (int)session->devices.size() < session->config.max_devices; index++) {
            ggml_backend_t backend = (ggml_backend_t)backends[index];
            if (!backend || !ggml_backend_is_cuda(backend)) {
                continue;
            }
            ggml_backend_cuda_context * context =
                (ggml_backend_cuda_context *)backend->context;
            const int logical = context->device;
            const ggml_cuda_device_info & info = ggml_cuda_info();
            if (logical < 0 || logical >= info.device_count) {
                continue;
            }
            const int physical = info.devices[logical].physical_device;
            if (!seen_devices.insert(physical).second) {
                continue;
            }

            cudaDeviceProp properties;
            ggml_cuda_set_device(logical);
            cudaError_t error = cudaGetDeviceProperties(&properties, logical);
            if (error != cudaSuccess) {
                (void)cudaGetLastError();
                MOE_CACHE_LOG("[moe-cache] CUDA%d skipped: device query failed\n", physical);
                continue;
            }
            const int capability = properties.major * 100 + properties.minor * 10;
            if (capability < session->config.min_compute_capability) {
                MOE_CACHE_LOG("[moe-cache] CUDA%d skipped: compute capability %d.%d is below %d.%d\n",
                        physical, properties.major, properties.minor,
                        session->config.min_compute_capability / 100,
                        (session->config.min_compute_capability % 100) / 10);
                continue;
            }

            default_min_expert_bytes = std::max(
                    default_min_expert_bytes,
                    moe_cache_default_min_expert_bytes(capability));
            minimum_capability = std::min(minimum_capability, capability);
            session->devices.emplace_back(new moe_cache_device(logical, physical));
        }

        if (session->devices.empty() ||
            (session->config.automatic && session->devices.size() < 2)) {
            return nullptr;
        }
        if (!session->config.min_expert_explicit) {
            session->config.min_expert_bytes = default_min_expert_bytes;
        }
        if (!session->config.serial_fill_explicit) {
            session->config.serial_fill =
                session->devices.size() < 2 || minimum_capability < moe_cache_cc_ampere;
        }
        if (!moe_cache_budget_register(*session)) {
            return nullptr;
        }

        moe_cache_session * result = session.get();
        try {
            std::lock_guard<std::mutex> lock(g_registry_mu);
            g_sessions.insert(result);
            g_session_count.fetch_add(1, std::memory_order_release);
        } catch (...) {
            for (const auto & device_ptr : session->devices) {
                moe_cache_budget_unregister(*device_ptr);
            }
            throw;
        }
        session.release();
        return result;
    } catch (...) {
        MOE_CACHE_LOG("[moe-cache] failed to create cache session\n");
        return nullptr;
    }
}

static void moe_cache_cancel_queue_locked(
        moe_cache_device & device, const void * base, size_t size, bool all) {
    for (auto it = device.queue.begin(); it != device.queue.end();) {
        if (all || moe_cache_ranges_overlap(it->source, it->bytes, base, size)) {
            device.queued_bytes = it->bytes <= device.queued_bytes
                ? device.queued_bytes - it->bytes : 0;
            if (it->pool >= 0 && it->pool < (int)device.pools.size()) {
                moe_cache_pool & pool = *device.pools[it->pool];
                if (it->slot >= 0 && it->slot < pool.n_slots) {
                    moe_cache_slot & slot = pool.slots[it->slot];
                    if (slot.state == moe_cache_slot_state::copying &&
                        slot.generation == it->generation && slot.key == it->key) {
                        moe_cache_slot_reset(pool, it->slot, true);
                    }
                }
            }
            it = device.queue.erase(it);
        } else {
            ++it;
        }
    }
}

static void moe_cache_free_device(moe_cache_device & device) {
    ggml_cuda_set_device(device.logical);
    if (device.compute_stream) {
        cudaStreamSynchronize(device.compute_stream);
    }
    for (const auto & mapping : device.host_mappings) {
        if (mapping.second.owned_registration) {
            (void)cudaHostUnregister((void *)mapping.first);
        }
    }
    device.host_mappings.clear();
    for (auto & route_entry : device.scheduler_direct_routes) {
        moe_cache_direct_route & route = route_entry.second;
        if (route.device_pointers) {
            cudaFree((void *)route.device_pointers);
            route.device_pointers = nullptr;
        }
        if (route.host_pointers) {
            cudaFreeHost((void *)route.host_pointers);
            route.host_pointers = nullptr;
        }
    }
    device.scheduler_direct_routes.clear();
    device.scheduler_gpu_ids_source_cache.clear();
    for (auto & pending : device.scheduler_direct_pending) {
        if (pending.event) {
            cudaEventSynchronize(pending.event);
            cudaEventDestroy(pending.event);
        }
    }
    device.scheduler_direct_pending.clear();
    for (auto & snapshot : device.scheduler_gpu_ids_pending) {
        if (snapshot.event) {
            cudaEventSynchronize(snapshot.event);
            cudaEventDestroy(snapshot.event);
            snapshot.event = nullptr;
        }
        if (snapshot.ids) {
            cudaFreeHost(snapshot.ids);
            snapshot.ids = nullptr;
        }
    }
    device.scheduler_gpu_ids_pending.clear();
    device.scheduler_gpu_ids_current_group = nullptr;
    device.scheduler_gpu_ids_current_snapshot = nullptr;
    device.scheduler_direct_epoch.pins.clear();
    device.scheduler_direct_epoch.host_bases.clear();
    device.scheduler_direct_epoch.stream = nullptr;
    device.scheduler_direct_epoch.gpu_ids_guard = false;
    device.scheduler_gpu_ids_guards.store(0);
    for (auto & pool_ptr : device.pools) {
        if (pool_ptr->slab) {
            cudaFree(pool_ptr->slab);
            moe_cache_budget_allocation(
                    device, (size_t)pool_ptr->n_slots * pool_ptr->expert_size, false);
            pool_ptr->slab = nullptr;
        }
    }
    if (device.d_input) {
        cudaFree(device.d_input);
        moe_cache_budget_allocation(device, device.d_input_cap, false);
        device.d_input = nullptr;
    }
    if (device.d_act_q8) {
        cudaFree(device.d_act_q8);
        moe_cache_budget_allocation(device, device.act_q8_cap, false);
        device.d_act_q8 = nullptr;
    }
    if (device.d_out) {
        cudaFree(device.d_out);
        moe_cache_budget_allocation(device, device.d_out_cap, false);
        device.d_out = nullptr;
    }
    if (device.d_gpu_miss) {
        cudaFree(device.d_gpu_miss);
        device.d_gpu_miss = nullptr;
    }
    if (device.h_input) {
        cudaFreeHost(device.h_input);
        device.h_input = nullptr;
    }
    if (device.h_out) {
        cudaFreeHost(device.h_out);
        device.h_out = nullptr;
    }
    if (device.compute_stream) {
        cudaStreamDestroy(device.compute_stream);
        device.compute_stream = nullptr;
    }
    for (auto & pending : device.scheduler_pending) {
        if (pending.event) {
            cudaEventDestroy(pending.event);
        }
    }
    device.scheduler_pending.clear();
    for (cudaEvent_t event : device.scheduler_free_events) {
        cudaEventDestroy(event);
    }
    device.scheduler_free_events.clear();
    device.pools.clear();
    device.allocated_bytes = 0;
    device.d_input_cap = 0;
    device.act_q8_cap = 0;
    device.d_out_cap = 0;
    device.d_gpu_miss_cap = 0;
    device.h_input_cap = 0;
    device.h_out_cap = 0;
}

static void moe_cache_session_destroy(void * opaque) {
    moe_cache_session * session = (moe_cache_session *)opaque;
    if (!session) {
        return;
    }

    {
        std::unique_lock<std::mutex> lock(session->mu);
        session->stopping = true;
        for (auto & device_ptr : session->devices) {
            moe_cache_cancel_queue_locked(*device_ptr, nullptr, 0, true);
        }
        session->cv.notify_all();
        session->idle_cv.wait(lock, [&] {
            return session->active_scopes == 0 && session->active_nodes == 0;
        });
    }

    for (auto & device_ptr : session->devices) {
        std::unique_lock<std::mutex> dispatch_lock(device_ptr->dispatch_mu);
        (void)moe_cache_scheduler_reap_locked(*session, *device_ptr, true);
    }

    for (auto & device_ptr : session->devices) {
        if (device_ptr->worker_started && device_ptr->worker.joinable()) {
            device_ptr->worker.join();
        }
    }
    {
        std::lock_guard<std::mutex> registry_lock(g_registry_mu);
        if (g_sessions.erase(session) > 0) {
            g_session_count.fetch_sub(1, std::memory_order_release);
        }
    }

    for (auto & device_ptr : session->devices) {
        if (device_ptr->nodes > 0 || device_ptr->dispatch_failures > 0 ||
            device_ptr->collect_failures > 0) {
            moe_cache_log_stats(*device_ptr);
        }
    }

    for (auto & device_ptr : session->devices) {
        std::unique_lock<std::mutex> dispatch_lock(device_ptr->dispatch_mu);
        moe_cache_free_device(*device_ptr);
        moe_cache_budget_unregister(*device_ptr);
    }

    delete session;
}

static void moe_cache_session_enter(void * opaque) {
    if (g_session_suppressed > 0) {
        g_session_suppressed++;
        return;
    }

    moe_cache_session * session = (moe_cache_session *)opaque;
    if (!session || session->dormant.load()) {
        if (g_session_stack.empty()) {
            return;
        }
        try {
            g_session_stack.push_back({session, nullptr});
        } catch (...) {
            g_session_suppressed++;
        }
        return;
    }
    std::lock_guard<std::mutex> lock(session->mu);
    if (session->dormant.load()) {
        if (g_session_stack.empty()) {
            return;
        }
        try {
            g_session_stack.push_back({session, nullptr});
        } catch (...) {
            g_session_suppressed++;
        }
        return;
    }
    if (session->stopping) {
        if (g_session_stack.empty()) {
            return;
        }
        try {
            g_session_stack.push_back({session, nullptr});
        } catch (...) {
            g_session_suppressed++;
        }
        return;
    }
    try {
        g_session_stack.push_back({session, session});
    } catch (...) {
        g_session_suppressed++;
        return;
    }
    session->active_scopes++;
}

static void moe_cache_direct_finalize_session(moe_cache_session & session);

static void moe_cache_session_leave(void * opaque) {
    if (g_session_suppressed > 0) {
        g_session_suppressed--;
        return;
    }
    moe_cache_session * expected = (moe_cache_session *)opaque;
    auto found = std::find_if(
            g_session_stack.rbegin(), g_session_stack.rend(),
            [expected](const moe_cache_scope_frame & frame) {
                return frame.requested == expected;
            });
    if (found == g_session_stack.rend()) {
        return;
    }
    moe_cache_session * active = found->active;
    if (active) {
        moe_cache_direct_finalize_session(*active);
    }
    g_session_stack.erase(std::next(found).base());
    if (active) {
        std::lock_guard<std::mutex> lock(active->mu);
        if (active->active_scopes > 0) {
            active->active_scopes--;
        }
        active->idle_cv.notify_all();
    }
}

static void * moe_cache_begin(
        const char * name, const void * host_base, size_t expert_size,
        int64_t n_in, int64_t n_out, int wtype, int64_t n_expert,
        int64_t n_tokens, int64_t n_rows) {
    if (g_session_suppressed > 0 || g_session_stack.empty()) {
        return nullptr;
    }
    moe_cache_session * session = g_session_stack.back().active;
    if (!session || session->stopping || session->dormant) {
        return nullptr;
    }
    moe_cache_log_configuration(*session);
    if (!name || !host_base ||
        !moe_cache_tensor_name_supported(name) || n_tokens < 1 ||
        expert_size < session->config.min_expert_bytes ||
        n_in <= 0 || n_out <= 0 || n_expert <= 0 ||
        !moe_cache_type_supported((ggml_type)wtype)) {
        return nullptr;
    }
    if (n_rows < n_tokens || n_rows % n_tokens != 0) {
        return nullptr;
    }
    if (n_tokens > session->config.max_batch) {
        bool expected = false;
        if (session->batch_bypass_announced.compare_exchange_strong(expected, true)) {
            MOE_CACHE_LOG("[moe-cache] bypassing %lld-token node above max-batch=%d; prompt batches normally use the CPU path, while larger decode or speculative batches will not use the cache\n",
                    (long long)n_tokens, session->config.max_batch);
        }
        return nullptr;
    }
    if (n_rows > moe_cache_node_rows_max) {
        bool expected = false;
        if (session->row_bypass_announced.compare_exchange_strong(expected, true)) {
            MOE_CACHE_LOG("[moe-cache] bypassing %lld-row node above row limit=%d; the stock CPU path remains active\n",
                    (long long)n_rows, moe_cache_node_rows_max);
        }
        return nullptr;
    }

    const size_t row_size = ggml_row_size((ggml_type)wtype, n_in);
    if (row_size == 0 || (uint64_t)n_out > SIZE_MAX / row_size ||
        expert_size != (size_t)n_out * row_size ||
        (uint64_t)n_expert > SIZE_MAX / expert_size ||
        expert_size > SIZE_MAX / moe_cache_pool_slots_min) {
        return nullptr;
    }
    const size_t tensor_size = (size_t)n_expert * expert_size;
    moe_cache_scratch scratch_requirements;
    if (!moe_cache_scratch_requirements(
            n_in, n_out, scratch_requirements)) {
        return nullptr;
    }
    int layer = -1;
    const bool has_layer = moe_cache_layer_number(name, layer);

    moe_cache_device * selected = nullptr;
    int pool_index = -1;
    moe_cache_pool * pool = nullptr;
    {
        std::unique_lock<std::mutex> lock(session->mu);
        if (session->stopping) {
            return nullptr;
        }

        int budget_devices = 0;
        int slab_devices = 0;
        int eligible_devices = 0;
        const size_t minimum_pool = expert_size * moe_cache_pool_slots_min;
        for (const auto & device_ptr : session->devices) {
            moe_cache_device & candidate = *device_ptr;
            if (candidate.dead.load() ||
                !moe_cache_prepare_budget(*session, candidate)) {
                continue;
            }
            budget_devices++;
            const size_t reserved = moe_cache_scratch_total(
                    candidate.scratch_reserve, &scratch_requirements);
            const size_t slab_limit =
                candidate.budget_limit > reserved
                    ? candidate.budget_limit - reserved : 0;
            if (slab_limit >= session->config.minimum_slab_bytes) {
                slab_devices++;
            }
            if (slab_limit >= minimum_pool &&
                slab_limit >= session->config.minimum_slab_bytes) {
                eligible_devices++;
            }
        }
        if (budget_devices == 0 ||
            (session->config.automatic &&
             (budget_devices < 2 || slab_devices < 2))) {
            if (session->config.automatic && slab_devices < 2) {
                MOE_CACHE_LOG("[moe-cache] session dormant: only %d devices satisfy the %zu MiB automatic slab floor\n",
                        slab_devices, session->config.minimum_slab_bytes >> 20);
            }
            session->dormant.store(true);
            lock.unlock();
            for (const auto & device_ptr : session->devices) {
                moe_cache_trim_session(*session, device_ptr->physical);
            }
            return nullptr;
        }
        if (session->config.automatic && eligible_devices < 2) {
            return nullptr;
        }

        auto route_weight = [&](moe_cache_device & candidate) -> size_t {
            if (candidate.dead.load() || !candidate.budget_ready) {
                return 0;
            }
            const size_t reserved = moe_cache_scratch_total(
                    candidate.scratch_reserve, &scratch_requirements);
            const size_t slab_limit =
                candidate.budget_limit > reserved
                    ? candidate.budget_limit - reserved : 0;
            if (candidate.allocated_bytes > slab_limit ||
                slab_limit < session->config.minimum_slab_bytes) {
                return 0;
            }
            if (moe_cache_find_pool(candidate, expert_size, wtype) >= 0) {
                return slab_limit;
            }
            const size_t available = slab_limit - candidate.allocated_bytes;
            return available >= minimum_pool ? available : 0;
        };

        bool tensor_override = !has_layer;
        auto find_routed_device = [&](int physical) -> moe_cache_device * {
            for (const auto & device_ptr : session->devices) {
                if (device_ptr->physical == physical) {
                    return device_ptr.get();
                }
            }
            return nullptr;
        };

        auto tensor_route = session->tensor_devices.find(host_base);
        if (tensor_route != session->tensor_devices.end()) {
            selected = find_routed_device(tensor_route->second);
            if (!selected || selected->dead.load() ||
                route_weight(*selected) == 0) {
                session->tensor_devices.erase(tensor_route);
                selected = nullptr;
            } else {
                tensor_override = true;
            }
        }

        if (!selected && has_layer) {
            auto layer_route = session->layer_devices.find(layer);
            if (layer_route != session->layer_devices.end()) {
                selected = find_routed_device(layer_route->second);
                if (!selected || selected->dead.load()) {
                    session->layer_devices.erase(layer);
                    selected = nullptr;
                } else if (route_weight(*selected) == 0) {
                    selected = nullptr;
                    tensor_override = true;
                }
            }
        }

        uint64_t total_weight = 0;
        if (!selected) {
            for (const auto & device_ptr : session->devices) {
                const size_t weight = route_weight(*device_ptr);
                if (weight > UINT64_MAX - total_weight) {
                    total_weight = UINT64_MAX;
                } else {
                    total_weight += weight;
                }
            }
        }
        if (!selected && total_weight == 0) {
            return nullptr;
        }

        if (!selected && has_layer && !tensor_override) {
            long double best_score = -1.0;
            for (const auto & device_ptr : session->devices) {
                moe_cache_device & candidate = *device_ptr;
                const size_t weight = route_weight(candidate);
                if (weight == 0) {
                    continue;
                }
                size_t assigned = 0;
                for (const auto & route : session->layer_devices) {
                    assigned += route.second == candidate.physical;
                }
                const long double score =
                    (long double)weight / (long double)(assigned + 1);
                if (!selected || score > best_score) {
                    selected = &candidate;
                    best_score = score;
                }
            }
            if (selected) {
                try {
                    session->layer_devices.emplace(layer, selected->physical);
                } catch (...) {
                    return nullptr;
                }
            }
        }

        const uint64_t hash = has_layer
            ? ((uint64_t)(uint32_t)layer + 1) * 0x9e3779b97f4a7c15ULL
            : moe_cache_name_hash(name);
        if (!selected) {
            const uint64_t target = hash % total_weight;
            uint64_t cumulative = 0;
            for (const auto & device_ptr : session->devices) {
                moe_cache_device & candidate = *device_ptr;
                const size_t weight = route_weight(candidate);
                if (weight == 0) {
                    continue;
                }
                cumulative = weight > UINT64_MAX - cumulative
                    ? UINT64_MAX : cumulative + weight;
                if (target < cumulative) {
                    selected = &candidate;
                    break;
                }
            }
            if (!selected) {
                return nullptr;
            }
            try {
                if (tensor_override) {
                    session->tensor_devices.emplace(host_base, selected->physical);
                } else {
                    session->layer_devices.emplace(layer, selected->physical);
                }
            } catch (...) {
                return nullptr;
            }
        }

        const size_t selected_scratch = moe_cache_scratch_total(
                selected->scratch_reserve, &scratch_requirements);
        if (selected_scratch >= selected->budget_limit ||
            minimum_pool > selected->budget_limit - selected_scratch) {
            return nullptr;
        }
        selected->scratch_reserve.input = std::max(
                selected->scratch_reserve.input, scratch_requirements.input);
        selected->scratch_reserve.q8 = std::max(
                selected->scratch_reserve.q8, scratch_requirements.q8);
        selected->scratch_reserve.out = std::max(
                selected->scratch_reserve.out, scratch_requirements.out);

        try {
            pool_index = moe_cache_discover_pool(
                    *session, *selected, host_base, tensor_size, expert_size, wtype, n_expert);
        } catch (...) {
            MOE_CACHE_LOG("[moe-cache] disabled one node after host allocation failure\n");
            return nullptr;
        }
        if (pool_index < 0 || pool_index >= (int)selected->pools.size() ||
            !selected->pools[pool_index]->slab) {
            return nullptr;
        }
        pool = selected->pools[pool_index].get();
        moe_cache_session::active_source * source = nullptr;
        try {
            source = &session->active_sources[host_base];
        } catch (...) {
            return nullptr;
        }
        source->bytes = std::max(source->bytes, tensor_size);
        source->references++;
        session->active_nodes++;
    }

    moe_cache_device & device = *selected;
    std::unique_lock<std::mutex> dispatch_lock;
    try {
        dispatch_lock = std::unique_lock<std::mutex>(
                device.dispatch_mu, std::try_to_lock);
    } catch (...) {
        std::lock_guard<std::mutex> lock(session->mu);
        auto source = session->active_sources.find(host_base);
        if (source != session->active_sources.end() && --source->second.references == 0) {
            session->active_sources.erase(source);
        }
        session->active_nodes--;
        session->idle_cv.notify_all();
        return nullptr;
    }
    if (!dispatch_lock.owns_lock() || device.dead.load()) {
        std::lock_guard<std::mutex> lock(session->mu);
        auto source = session->active_sources.find(host_base);
        if (source != session->active_sources.end() && --source->second.references == 0) {
            session->active_sources.erase(source);
        }
        session->active_nodes--;
        session->idle_cv.notify_all();
        return nullptr;
    }

    std::unique_ptr<moe_cache_node> node(new (std::nothrow) moe_cache_node());
    if (!node) {
        std::lock_guard<std::mutex> lock(session->mu);
        auto source = session->active_sources.find(host_base);
        if (source != session->active_sources.end() && --source->second.references == 0) {
            session->active_sources.erase(source);
        }
        session->active_nodes--;
        session->idle_cv.notify_all();
        return nullptr;
    }
    node->session = session;
    node->device = &device;
    node->pool = pool;
    node->pool_index = pool_index;
    node->host_base = host_base;
    node->expert_size = expert_size;
    node->n_in = n_in;
    node->n_out = n_out;
    node->n_expert = n_expert;
    node->n_tokens = n_tokens;
    node->wtype = wtype;
    node->dispatch_lock = std::move(dispatch_lock);
    return node.release();
}

static int moe_cache_overlap_rows(const moe_cache_node & node, int n_ids) {
    const int configured = node.session->config.overlap_cpu_rows;
    if (configured >= 0) {
        return std::min(configured, std::max(0, n_ids - 1));
    }
    if (n_ids <= 1 || node.n_tokens <= 0 || n_ids % node.n_tokens != 0) {
        return 0;
    }

    const int n_tokens = (int) node.n_tokens;
    const int top_k = n_ids / n_tokens;
    // Bound CPU work by expert bytes, token count, and one quarter of the node.
    const size_t work_budget = moe_cache_overlap_bytes_per_token * (size_t) n_tokens;
    int rows = (int) std::min<size_t>(work_budget / node.expert_size, INT_MAX);
    rows = std::max(rows, 1);
    if (top_k >= 8) {
        rows = std::max(rows, 2);
    }

    rows = std::min(rows, std::max(1, n_ids / 4));
    rows = std::min(rows, std::max(2, n_tokens));
    return std::min(rows, n_ids - 1);
}

static int moe_cache_lookup_or_queue_locked(
        moe_cache_session & session, moe_cache_device & device,
        moe_cache_pool & pool, int pool_index, const void * host_base,
        int32_t expert, size_t expert_size, int & inserts_left,
        bool & wake_worker) {
    const moe_cache_key key{host_base, expert};
    auto found = pool.map.find(key);
    if (found != pool.map.end()) {
        return pool.slots[found->second].state == moe_cache_slot_state::valid
            ? found->second : -1;
    }

    moe_cache_demand * demand = nullptr;
    try {
        demand = &device.demand_count[key];
    } catch (...) {
        device.insert_skips++;
        return -1;
    }
    demand->expert_size = expert_size;
    if (demand->count < std::numeric_limits<uint16_t>::max()) {
        demand->count++;
    }
    const int initial_admit_after =
        pool.covers_all_entries && !session.config.admit_after_explicit
            ? 1 : session.config.admit_after;
    const int admit_after = pool.free_slots.empty()
        ? std::max(initial_admit_after, session.config.readmit_after)
        : initial_admit_after;
    if (demand->count < admit_after) {
        device.admission_skips++;
        return -1;
    }
    const size_t queue_limit = session.config.queue_mb << 20;
    if (inserts_left <= 0 || (int)device.queue.size() >= session.config.queue_max ||
        expert_size > queue_limit - std::min(queue_limit, device.queued_bytes)) {
        device.insert_skips++;
        return -1;
    }

    int slot_index = -1;
    if (!pool.free_slots.empty()) {
        slot_index = pool.free_slots.back();
        pool.free_slots.pop_back();
    } else {
        // V1.5 GPU-ID routing may publish direct pointers for the whole
        // current scheduler scope. Do not recycle any valid cache slot until
        // the scope's CUDA completion event has retired.
        if (device.scheduler_gpu_ids_guards.load() > 0) {
            device.insert_skips++;
            return -1;
        }
        int candidate = pool.lru_head;
        while (candidate >= 0 && pool.slots[candidate].readers > 0) {
            candidate = pool.slots[candidate].next;
        }
        if (candidate < 0) {
            device.insert_skips++;
            return -1;
        }
        slot_index = candidate;
        moe_cache_slot_reset(pool, slot_index, false);
        device.evictions++;
    }

    moe_cache_slot & slot = pool.slots[slot_index];
    slot.key = key;
    slot.generation++;
    slot.readers = 0;
    slot.state = moe_cache_slot_state::copying;
    try {
        const auto inserted = pool.map.emplace(key, slot_index);
        if (!inserted.second) {
            moe_cache_slot_reset(pool, slot_index, true);
            device.insert_skips++;
            return -1;
        }

        const void * source =
            (const char *)host_base + (size_t)expert * expert_size;
        device.queue.push_back({
                pool_index, slot_index, slot.generation,
                key, source, expert_size});
        device.queued_bytes += expert_size;
    } catch (...) {
        moe_cache_slot_reset(pool, slot_index, true);
        device.insert_skips++;
        return -1;
    }
    device.inserts++;
    inserts_left--;
    wake_worker = true;
    return -1;
}

static void moe_cache_scheduler_release_pending_locked(
        moe_cache_session & session,
        moe_cache_scheduler_pending & pending) {
    for (int index = 0; index < pending.n_pins; index++) {
        moe_cache_scheduler_pin & pin = pending.pins[index];
        if (pin.pool && pin.slot >= 0 && pin.slot < pin.pool->n_slots) {
            moe_cache_slot & slot = pin.pool->slots[pin.slot];
            if (slot.readers > 0) {
                slot.readers--;
            }
        }
    }
    pending.n_pins = 0;
    if (pending.host_base) {
        auto source = session.active_sources.find(pending.host_base);
        if (source != session.active_sources.end() &&
            --source->second.references == 0) {
            session.active_sources.erase(source);
        }
        pending.host_base = nullptr;
    }
    session.idle_cv.notify_all();
}

static void moe_cache_scheduler_release_pending(
        moe_cache_session & session, moe_cache_device & device,
        moe_cache_scheduler_pending & pending) {
    (void)device;
    std::lock_guard<std::mutex> lock(session.mu);
    moe_cache_scheduler_release_pending_locked(session, pending);
}

static bool moe_cache_direct_reap_locked(
        moe_cache_session & session, moe_cache_device & device, bool wait);

static bool moe_cache_scheduler_reap_locked(
        moe_cache_session & session, moe_cache_device & device, bool wait) {
    ggml_cuda_set_device(device.logical);
    if (!moe_cache_direct_reap_locked(session, device, wait)) {
        return false;
    }
    while (!device.scheduler_pending.empty()) {
        moe_cache_scheduler_pending & pending = device.scheduler_pending.front();
        cudaError_t status = wait
            ? cudaEventSynchronize(pending.event)
            : cudaEventQuery(pending.event);
        if (!wait && status == cudaErrorNotReady) {
            break;
        }
        if (status != cudaSuccess) {
            (void)cudaGetLastError();
            device.dead.store(true);
            return false;
        }
        moe_cache_scheduler_release_pending(session, device, pending);
        device.scheduler_free_events.push_back(pending.event);
        device.scheduler_pending.pop_front();
    }
    return true;
}

static void moe_cache_direct_release_pending_locked(
        moe_cache_session & session, moe_cache_device & device,
        moe_cache_direct_pending & pending) {
    for (moe_cache_scheduler_pin & pin : pending.pins) {
        if (pin.pool && pin.slot >= 0 && pin.slot < pin.pool->n_slots) {
            moe_cache_slot & slot = pin.pool->slots[pin.slot];
            if (slot.readers > 0) {
                slot.readers--;
            }
        }
    }
    pending.pins.clear();
    for (const void * host_base : pending.host_bases) {
        auto source = session.active_sources.find(host_base);
        if (source != session.active_sources.end() &&
            --source->second.references == 0) {
            session.active_sources.erase(source);
        }
    }
    pending.host_bases.clear();
    if (pending.gpu_ids_guard && device.scheduler_gpu_ids_guards.load() > 0) {
        device.scheduler_gpu_ids_guards--;
        pending.gpu_ids_guard = false;
    }
    session.idle_cv.notify_all();
}

static bool moe_cache_direct_reap_locked(
        moe_cache_session & session, moe_cache_device & device, bool wait) {
    ggml_cuda_set_device(device.logical);
    while (!device.scheduler_direct_pending.empty()) {
        moe_cache_direct_pending & pending =
            device.scheduler_direct_pending.front();
        cudaError_t status = wait
            ? cudaEventSynchronize(pending.event)
            : cudaEventQuery(pending.event);
        if (!wait && status == cudaErrorNotReady) {
            break;
        }
        if (status != cudaSuccess) {
            (void)cudaGetLastError();
            device.dead.store(true);
            return false;
        }
        {
            std::lock_guard<std::mutex> lock(session.mu);
            moe_cache_direct_release_pending_locked(session, device, pending);
        }
        device.scheduler_free_events.push_back(pending.event);
        device.scheduler_direct_pending.pop_front();
    }
    return true;
}

static void moe_cache_direct_rollback_epoch_locked(
        moe_cache_session & session, moe_cache_device & device,
        size_t pin_begin, size_t source_begin) {
    while (device.scheduler_direct_epoch.pins.size() > pin_begin) {
        moe_cache_scheduler_pin pin = device.scheduler_direct_epoch.pins.back();
        device.scheduler_direct_epoch.pins.pop_back();
        if (pin.pool && pin.slot >= 0 && pin.slot < pin.pool->n_slots &&
            pin.pool->slots[pin.slot].readers > 0) {
            pin.pool->slots[pin.slot].readers--;
        }
    }
    while (device.scheduler_direct_epoch.host_bases.size() > source_begin) {
        const void * host_base = device.scheduler_direct_epoch.host_bases.back();
        device.scheduler_direct_epoch.host_bases.pop_back();
        auto source = session.active_sources.find(host_base);
        if (source != session.active_sources.end() &&
            --source->second.references == 0) {
            session.active_sources.erase(source);
        }
    }
}

static bool moe_cache_direct_finalize_device(
        moe_cache_session & session, moe_cache_device & device) {
    if (device.scheduler_direct_epoch.pins.empty() &&
        device.scheduler_direct_epoch.host_bases.empty() &&
        !device.scheduler_direct_epoch.gpu_ids_guard) {
        device.scheduler_direct_epoch.stream = nullptr;
        return true;
    }
    cudaStream_t stream = device.scheduler_direct_epoch.stream;
    if (!stream) {
        return false;
    }

    moe_cache_direct_pending pending;
    pending.pins.swap(device.scheduler_direct_epoch.pins);
    pending.host_bases.swap(device.scheduler_direct_epoch.host_bases);
    pending.gpu_ids_guard = device.scheduler_direct_epoch.gpu_ids_guard;
    device.scheduler_direct_epoch.gpu_ids_guard = false;
    device.scheduler_direct_epoch.stream = nullptr;

    ggml_cuda_set_device(device.logical);
    if (!device.scheduler_free_events.empty()) {
        pending.event = device.scheduler_free_events.back();
        device.scheduler_free_events.pop_back();
    } else if (!moe_cache_cuda_ok(
            device, cudaEventCreateWithFlags(
                &pending.event, cudaEventDisableTiming),
            "scheduler direct event create", false)) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::lock_guard<std::mutex> lock(session.mu);
        moe_cache_direct_release_pending_locked(session, device, pending);
        return false;
    }

    if (!moe_cache_cuda_ok(
            device, cudaEventRecord(pending.event, stream),
            "scheduler direct event record", false)) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        {
            std::lock_guard<std::mutex> lock(session.mu);
            moe_cache_direct_release_pending_locked(session, device, pending);
        }
        device.scheduler_free_events.push_back(pending.event);
        return false;
    }
    try {
        device.scheduler_direct_pending.push_back(std::move(pending));
    } catch (...) {
        CUDA_CHECK(cudaEventSynchronize(pending.event));
        {
            std::lock_guard<std::mutex> lock(session.mu);
            moe_cache_direct_release_pending_locked(session, device, pending);
        }
        device.scheduler_free_events.push_back(pending.event);
        return false;
    }
    return true;
}

static void moe_cache_direct_finalize_session(moe_cache_session & session) {
    for (const auto & device_ptr : session.devices) {
        moe_cache_device & device = *device_ptr;
        if (device.scheduler_direct_epoch.pins.empty() &&
            device.scheduler_direct_epoch.host_bases.empty() &&
            !device.scheduler_direct_epoch.gpu_ids_guard) {
            continue;
        }
        std::unique_lock<std::mutex> dispatch_lock(device.dispatch_mu);
        (void)moe_cache_direct_finalize_device(session, device);
    }
}


extern "C" int ggml_cuda_moe_cache_materialize_enabled(void) {
    if (g_session_suppressed > 0 || g_session_stack.empty()) {
        return 0;
    }
    moe_cache_session * session = g_session_stack.back().active;
    return session && !session->stopping && !session->dormant &&
        session->config.scheduler_copy;
}

extern "C" int ggml_cuda_moe_cache_materialize(
        void * opaque_backend, const char * tensor_name,
        const void * host_base, size_t tensor_size, size_t expert_size,
        int wtype, int64_t n_expert, const int32_t * ids, int n_ids,
        void * dst_base) {
    if (g_session_suppressed > 0 || g_session_stack.empty() ||
        !opaque_backend || !tensor_name || !host_base || !dst_base ||
        !ids || n_ids < 1 || n_ids > moe_cache_node_rows_max ||
        n_expert <= 0 || expert_size == 0 ||
        (uint64_t)n_expert > SIZE_MAX / expert_size ||
        tensor_size < (size_t)n_expert * expert_size ||
        !moe_cache_tensor_name_supported(tensor_name) ||
        !moe_cache_type_supported((ggml_type)wtype)) {
        return 0;
    }

    moe_cache_session * session = g_session_stack.back().active;
    if (!session || session->stopping || session->dormant ||
        !session->config.scheduler_copy ||
        expert_size < session->config.min_expert_bytes) {
        return 0;
    }
    moe_cache_log_configuration(*session);

    ggml_backend_t backend = (ggml_backend_t)opaque_backend;
    if (!ggml_backend_is_cuda(backend)) {
        return 0;
    }
    ggml_backend_cuda_context * cuda_ctx =
        (ggml_backend_cuda_context *)backend->context;
    if (!cuda_ctx) {
        return 0;
    }

    const int logical = cuda_ctx->device;
    moe_cache_device * selected = nullptr;
    for (const auto & device_ptr : session->devices) {
        if (device_ptr->logical == logical) {
            selected = device_ptr.get();
            break;
        }
    }
    if (!selected || selected->dead.load()) {
        return 0;
    }

    std::unique_lock<std::mutex> dispatch_lock(
        selected->dispatch_mu, std::try_to_lock);
    if (!dispatch_lock.owns_lock() || selected->dead.load()) {
        return 0;
    }

    ggml_cuda_set_device(logical);
    if (!moe_cache_scheduler_reap_locked(*session, *selected, false)) {
        return 0;
    }

    const char * mapped_host =
        moe_cache_map_host_source(*selected, host_base, tensor_size);
    if (!mapped_host) {
        return 0;
    }

    moe_cache_pool * pool = nullptr;
    int pool_index = -1;
    moe_cache_scheduler_pending pending = {};
    moe_cache_copy_batch batch = {};
    int inserts_left = session->config.inserts_per_plan;
    bool wake_worker = false;

    {
        std::lock_guard<std::mutex> lock(session->mu);
        if (session->stopping || session->dormant || selected->dead.load()) {
            return 0;
        }

        try {
            pool_index = moe_cache_discover_pool(
                *session, *selected, host_base, tensor_size,
                expert_size, wtype, n_expert);
        } catch (...) {
            return 0;
        }
        if (pool_index < 0 || pool_index >= (int)selected->pools.size()) {
            return 0;
        }
        pool = selected->pools[pool_index].get();
        if (!pool || !pool->slab) {
            return 0;
        }

        try {
            moe_cache_session::active_source & source =
                session->active_sources[host_base];
            source.bytes = std::max(source.bytes, tensor_size);
            source.references++;
            pending.host_base = host_base;
        } catch (...) {
            return 0;
        }

        int32_t previous = -1;
        for (int index = 0; index < n_ids; index++) {
            const int32_t expert = ids[index];
            if (expert < 0 || expert >= n_expert || expert <= previous) {
                moe_cache_scheduler_release_pending_locked(*session, pending);
                return 0;
            }
            previous = expert;

            const int slot_index = moe_cache_lookup_or_queue_locked(
                *session, *selected, *pool, pool_index,
                host_base, expert, expert_size,
                inserts_left, wake_worker);

            const char * source = nullptr;
            if (slot_index >= 0) {
                moe_cache_slot & slot = pool->slots[slot_index];
                slot.readers++;
                moe_cache_lru_remove(*pool, slot_index);
                moe_cache_lru_push_back(*pool, slot_index);
                pending.pins[pending.n_pins++] = {pool, slot_index};
                source = pool->slab + (size_t)slot_index * expert_size;
                selected->hits++;
                selected->scheduler_copy_hits++;
            } else {
                source = mapped_host + (size_t)expert * expert_size;
                selected->misses++;
                selected->scheduler_copy_misses++;
            }

            if (batch.count >= moe_cache_copy_desc_max) {
                moe_cache_scheduler_release_pending_locked(*session, pending);
                return 0;
            }
            batch.desc[batch.count++] = {
                source,
                (char *)dst_base + (size_t)expert * expert_size,
                expert_size};

            // Match the scheduler's stock selective-copy contract: MMQ may
            // inspect a small amount of valid data beyond the last used expert
            // in a consecutive run. Cache slabs do not contain that padding,
            // so always source it from the mapped host tensor.
            const bool run_end =
                index + 1 == n_ids || ids[index + 1] != expert + 1;
            if (run_end && expert < n_expert - 1) {
                const size_t padding = std::min<size_t>(expert_size, 512);
                if (batch.count >= moe_cache_copy_desc_max) {
                    moe_cache_scheduler_release_pending_locked(*session, pending);
                    return 0;
                }
                batch.desc[batch.count++] = {
                    mapped_host + (size_t)(expert + 1) * expert_size,
                    (char *)dst_base + (size_t)(expert + 1) * expert_size,
                    padding};
            }
        }
    }

    if (wake_worker) {
        session->cv.notify_one();
    }

    if (batch.count <= 0) {
        moe_cache_scheduler_release_pending(*session, *selected, pending);
        return 0;
    }

    const dim3 grid(
        moe_cache_copy_blocks_per_desc,
        (unsigned int)batch.count,
        1);
    moe_cache_copy_batch_kernel<<<grid, 256, 0, cuda_ctx->stream()>>>(batch);
    if (!moe_cache_cuda_ok(
            *selected, cudaPeekAtLastError(),
            "scheduler copy kernel", true)) {
        moe_cache_scheduler_release_pending(*session, *selected, pending);
        return 0;
    }

    if (!selected->scheduler_free_events.empty()) {
        pending.event = selected->scheduler_free_events.back();
        selected->scheduler_free_events.pop_back();
    } else if (!moe_cache_cuda_ok(
            *selected, cudaEventCreateWithFlags(
                &pending.event, cudaEventDisableTiming),
            "scheduler copy event create", false)) {
        CUDA_CHECK(cudaStreamSynchronize(cuda_ctx->stream()));
        moe_cache_scheduler_release_pending(*session, *selected, pending);
        return 0;
    }

    if (!moe_cache_cuda_ok(
            *selected, cudaEventRecord(pending.event, cuda_ctx->stream()),
            "scheduler copy event record", false)) {
        CUDA_CHECK(cudaStreamSynchronize(cuda_ctx->stream()));
        moe_cache_scheduler_release_pending(*session, *selected, pending);
        selected->scheduler_free_events.push_back(pending.event);
        return 0;
    }

    try {
        selected->scheduler_pending.push_back(pending);
    } catch (...) {
        CUDA_CHECK(cudaEventSynchronize(pending.event));
        moe_cache_scheduler_release_pending(*session, *selected, pending);
        selected->scheduler_free_events.push_back(pending.event);
        return 0;
    }
    selected->scheduler_copy_calls++;
    selected->scheduler_copy_bytes +=
        (unsigned long long)n_ids * expert_size;

    if (session->config.stats_every > 0 &&
        selected->scheduler_copy_calls % session->config.stats_every == 0) {
        moe_cache_log_stats(*selected);
    }
    return 1;
}


static void moe_cache_gpu_ids_resolve_pointer_table_locked(
        moe_cache_session & session, moe_cache_device & device,
        moe_cache_direct_route & route, const void * host_base,
        size_t tensor_size, size_t expert_size, int wtype, int64_t n_expert,
        void * dst_base) {
    const moe_cache_source_residency_key source_key = {
        host_base, n_expert, expert_size, wtype};

    try {
        auto inserted = device.scheduler_gpu_ids_source_cache.try_emplace(source_key);
        moe_cache_source_residency & cached = inserted.first->second;

        int pool_index = cached.pool_index;
        if (pool_index < 0 || pool_index >= (int)device.pools.size()) {
            pool_index = moe_cache_discover_pool(
                session, device, host_base, tensor_size,
                expert_size, wtype, n_expert);
        }
        moe_cache_pool * pool =
            pool_index >= 0 && pool_index < (int)device.pools.size()
                ? device.pools[pool_index].get() : nullptr;

        const bool rebuild =
            cached.pool_index != pool_index ||
            cached.resident.size() != (size_t)n_expert;
        if (rebuild) {
            cached.pool_index = pool_index;
            cached.resident.assign((size_t)n_expert, nullptr);
            if (pool && pool->slab) {
                for (int32_t expert = 0; expert < n_expert; ++expert) {
                    const moe_cache_key key{host_base, expert};
                    auto found = pool->map.find(key);
                    if (found == pool->map.end()) {
                        continue;
                    }
                    const int slot_index = found->second;
                    if (slot_index >= 0 && slot_index < pool->n_slots &&
                        pool->slots[slot_index].state == moe_cache_slot_state::valid) {
                        cached.resident[(size_t)expert] =
                            pool->slab + (size_t)slot_index * expert_size;
                    }
                }
            }
        }

        for (int32_t expert = 0; expert < n_expert; ++expert) {
            const void * resident = cached.resident[(size_t)expert];
            route.host_pointers[expert] = resident
                ? resident
                : (char *)dst_base + (size_t)expert * expert_size;
        }
        return;
    } catch (...) {
        // Allocation failure in the optimization must not break inference.
        // Fall back to a direct map scan for this prepare.
    }

    const int pool_index = moe_cache_discover_pool(
        session, device, host_base, tensor_size,
        expert_size, wtype, n_expert);
    moe_cache_pool * pool =
        pool_index >= 0 && pool_index < (int)device.pools.size()
            ? device.pools[pool_index].get() : nullptr;
    for (int32_t expert = 0; expert < n_expert; ++expert) {
        const void * pointer =
            (char *)dst_base + (size_t)expert * expert_size;
        if (pool && pool->slab) {
            const moe_cache_key key{host_base, expert};
            auto found = pool->map.find(key);
            if (found != pool->map.end()) {
                const int slot_index = found->second;
                if (slot_index >= 0 && slot_index < pool->n_slots &&
                    pool->slots[slot_index].state == moe_cache_slot_state::valid) {
                    pointer = pool->slab + (size_t)slot_index * expert_size;
                }
            }
        }
        route.host_pointers[expert] = pointer;
    }
}

static bool moe_cache_gpu_ids_process_pending_locked(
        moe_cache_session & session, moe_cache_device & device) {
    if (device.scheduler_gpu_ids_guards.load() != 0) {
        return true;
    }

    bool wake_worker = false;
    for (auto & snapshot : device.scheduler_gpu_ids_pending) {
        if (!snapshot.valid || !snapshot.event || !snapshot.ids ||
            snapshot.ids_count <= 0) {
            continue;
        }

        cudaError_t status = cudaEventQuery(snapshot.event);
        if (status == cudaErrorNotReady) {
            continue;
        }
        if (status != cudaSuccess) {
            (void)cudaGetLastError();
            device.dead.store(true);
            return false;
        }

        const int ids_count = snapshot.ids_count;
        for (const auto & source : snapshot.sources) {
            const void * host_base = source.host_base;
            const size_t tensor_size = source.tensor_size;
            const size_t expert_size = source.expert_size;
            const int wtype = source.wtype;
            const int64_t n_expert = source.n_expert;
            if (!host_base || expert_size == 0 ||
                n_expert <= 0 || n_expert > INT_MAX) {
                continue;
            }

            int pool_index = moe_cache_discover_pool(
                session, device, host_base, tensor_size,
                expert_size, wtype, n_expert);
            moe_cache_pool * pool =
                pool_index >= 0 && pool_index < (int)device.pools.size()
                    ? device.pools[pool_index].get() : nullptr;

            std::vector<uint8_t> seen((size_t)n_expert, 0);
            int inserts_left = session.config.inserts_per_plan;
            for (int index = 0; index < ids_count; ++index) {
                const int32_t expert = snapshot.ids[index];
                if (expert < 0 || expert >= n_expert || seen[(size_t)expert]) {
                    continue;
                }
                seen[(size_t)expert] = 1;

                int slot_index = -1;
                if (pool && pool->slab) {
                    slot_index = moe_cache_lookup_or_queue_locked(
                        session, device, *pool, pool_index,
                        host_base, expert, expert_size,
                        inserts_left, wake_worker);
                }
                if (slot_index >= 0) {
                    moe_cache_lru_remove(*pool, slot_index);
                    moe_cache_lru_push_back(*pool, slot_index);
                    device.hits++;
                    device.scheduler_direct_hits++;
                } else {
                    device.misses++;
                    device.scheduler_direct_misses++;
                    device.scheduler_direct_miss_bytes += expert_size;
                }
            }
            device.scheduler_gpu_ids_metadata++;
        }

        snapshot.valid = false;
        snapshot.ids_count = 0;
        snapshot.ids_group = nullptr;
        snapshot.sources.clear();
    }

    if (wake_worker) {
        session.cv.notify_one();
    }
    return true;
}

extern "C" int ggml_cuda_moe_cache_direct_materialize_gpu_ids(
        void * opaque_backend, const char * tensor_name,
        const void * host_base, size_t tensor_size, size_t expert_size,
        int wtype, int64_t n_expert, const int32_t * device_ids,
        int64_t ids_ne0, int64_t ids_ne1, size_t ids_nb1,
        const void * ids_group, void * dst_base) {
    if (g_session_suppressed > 0 || g_session_stack.empty() ||
        !opaque_backend || !tensor_name || !host_base || !dst_base ||
        !device_ids || !ids_group || ids_ne0 <= 0 || ids_ne1 <= 0 ||
        ids_ne0 > moe_cache_node_rows_max || ids_ne1 > moe_cache_node_rows_max ||
        ids_ne0 > moe_cache_node_rows_max / ids_ne1 ||
        ids_nb1 < (size_t)ids_ne0 * sizeof(int32_t) ||
        n_expert <= 0 || expert_size == 0 ||
        (uint64_t)n_expert > SIZE_MAX / expert_size ||
        tensor_size < (size_t)n_expert * expert_size ||
        !moe_cache_tensor_name_supported(tensor_name) ||
        !moe_cache_type_supported((ggml_type)wtype) ||
        !moe_cache_scheduler_direct_type_supported((ggml_type)wtype)) {
        return 0;
    }

    moe_cache_session * session = g_session_stack.back().active;
    if (!session || session->stopping || session->dormant ||
        !session->config.scheduler_copy || !session->config.scheduler_direct ||
        !session->config.scheduler_gpu_ids ||
        expert_size < session->config.min_expert_bytes) {
        return 0;
    }
    moe_cache_log_configuration(*session);

    ggml_backend_t backend = (ggml_backend_t)opaque_backend;
    if (!ggml_backend_is_cuda(backend)) {
        return 0;
    }
    ggml_backend_cuda_context * cuda_ctx =
        (ggml_backend_cuda_context *)backend->context;
    if (!cuda_ctx) {
        return 0;
    }

    moe_cache_device * selected = nullptr;
    for (const auto & device_ptr : session->devices) {
        if (device_ptr->logical == cuda_ctx->device) {
            selected = device_ptr.get();
            break;
        }
    }
    if (!selected || selected->dead.load()) {
        return 0;
    }

    std::unique_lock<std::mutex> dispatch_lock(selected->dispatch_mu);
    ggml_cuda_set_device(selected->logical);
    if (!moe_cache_direct_reap_locked(*session, *selected, false)) {
        return 0;
    }

    if (!selected->scheduler_direct_epoch.gpu_ids_guard) {
        // New scheduler scope. IDs tensor objects are stable group identities
        // only within this scope; allocator/device pointers may be reused later.
        selected->scheduler_gpu_ids_current_group = nullptr;
        selected->scheduler_gpu_ids_current_snapshot = nullptr;
        if (selected->scheduler_gpu_ids_guards.load() == 0) {
            std::lock_guard<std::mutex> lock(session->mu);
            if (!moe_cache_gpu_ids_process_pending_locked(*session, *selected)) {
                return 0;
            }
        }
        selected->scheduler_gpu_ids_guards++;
        selected->scheduler_direct_epoch.gpu_ids_guard = true;
    }

    if (selected->scheduler_direct_epoch.stream &&
        selected->scheduler_direct_epoch.stream != cuda_ctx->stream()) {
        GGML_ABORT("scheduler gpu-ids: CUDA stream changed within one scheduler scope");
    }
    selected->scheduler_direct_epoch.stream = cuda_ctx->stream();

    const char * mapped_host =
        moe_cache_map_host_source(*selected, host_base, tensor_size);
    if (!mapped_host) {
        GGML_ABORT("scheduler gpu-ids: failed to map host expert tensor %s", tensor_name);
    }

    const moe_cache_direct_route_key route_key = {
        dst_base, n_expert, expert_size, wtype};
    moe_cache_direct_route * route = nullptr;
    auto route_it = selected->scheduler_direct_routes.find(route_key);
    if (route_it == selected->scheduler_direct_routes.end()) {
        moe_cache_direct_route created;
        if (cudaMalloc((void **)&created.device_pointers,
                       (size_t)n_expert * sizeof(void *)) != cudaSuccess ||
            !created.device_pointers ||
            cudaMallocHost((void **)&created.host_pointers,
                           (size_t)n_expert * sizeof(void *)) != cudaSuccess ||
            !created.host_pointers) {
            (void)cudaGetLastError();
            if (created.device_pointers) cudaFree((void *)created.device_pointers);
            if (created.host_pointers) cudaFreeHost((void *)created.host_pointers);
            GGML_ABORT("scheduler gpu-ids: route allocation failed for %s", tensor_name);
        }
        try {
            route_it = selected->scheduler_direct_routes.emplace(
                route_key, created).first;
        } catch (...) {
            cudaFree((void *)created.device_pointers);
            cudaFreeHost((void *)created.host_pointers);
            GGML_ABORT("scheduler gpu-ids: route insertion failed for %s", tensor_name);
        }
    }
    route = &route_it->second;

    // Routes may have been created by the v1.4 CPU-ID path before GPU-ID
    // routing was enabled. Lazily add the pinned host mirrors in that case.
    if (!route->host_pointers) {
        CUDA_CHECK(cudaMallocHost(
            (void **)&route->host_pointers, (size_t)n_expert * sizeof(void *)));
    }

    {
        std::lock_guard<std::mutex> lock(session->mu);
        try {
            moe_cache_session::active_source & source =
                session->active_sources[host_base];
            source.bytes = std::max(source.bytes, tensor_size);
            source.references++;
            selected->scheduler_direct_epoch.host_bases.push_back(host_base);
        } catch (...) {
            GGML_ABORT("scheduler gpu-ids: active-source bookkeeping failed");
        }

        moe_cache_gpu_ids_resolve_pointer_table_locked(
            *session, *selected, *route, host_base, tensor_size,
            expert_size, wtype, n_expert, dst_base);
    }

    cudaStream_t stream = cuda_ctx->stream();
    CUDA_CHECK(cudaMemcpyAsync(
        (void *)route->device_pointers, route->host_pointers,
        (size_t)n_expert * sizeof(void *), cudaMemcpyHostToDevice, stream));

    moe_cache_gpu_ids_batch batch = {};
    batch.pointer_table = route->device_pointers;
    batch.mapped_host = mapped_host;
    batch.dst_base = (char *)dst_base;
    batch.ids = device_ids;
    batch.expert_size = expert_size;
    batch.ids_stride = ids_nb1 / sizeof(int32_t);
    batch.n_expert = (int)n_expert;
    batch.top_k = (int)ids_ne0;
    batch.n_tokens = (int)ids_ne1;

    const dim3 grid(
        moe_cache_copy_blocks_per_desc,
        (unsigned int)n_expert, 1);
    moe_cache_gpu_ids_prepare_kernel<<<grid, 256, 0, stream>>>(batch);
    if (!moe_cache_cuda_ok(
            *selected, cudaPeekAtLastError(),
            "scheduler gpu-ids prepare kernel", true)) {
        GGML_ABORT("scheduler gpu-ids: prepare kernel launch failed");
    }

    if (selected->scheduler_gpu_ids_pending.empty()) {
        try {
            selected->scheduler_gpu_ids_pending.resize(
                moe_cache_gpu_ids_snapshot_count);
        } catch (...) {
            GGML_ABORT("scheduler gpu-ids: snapshot ring allocation failed");
        }
        for (auto & candidate : selected->scheduler_gpu_ids_pending) {
            if (cudaMallocHost(
                    (void **)&candidate.ids,
                    moe_cache_node_rows_max * sizeof(int32_t)) != cudaSuccess ||
                !candidate.ids ||
                cudaEventCreateWithFlags(
                    &candidate.event, cudaEventDisableTiming) != cudaSuccess ||
                !candidate.event) {
                (void)cudaGetLastError();
                GGML_ABORT("scheduler gpu-ids: snapshot ring CUDA allocation failed");
            }
        }
    }

    // The same routing IDs tensor normally feeds multiple expert-weight
    // tensors (e.g. gate/up/down). Snapshot the IDs only once per tensor object
    // per scheduler scope, then attach every source tensor to that shared
    // snapshot. This preserves the old scheduler's prev_ids_tensor reuse while
    // keeping the current execution path fully GPU-resident.
    moe_cache_gpu_ids_snapshot * snapshot = nullptr;
    if (selected->scheduler_gpu_ids_current_group == ids_group &&
        selected->scheduler_gpu_ids_current_snapshot &&
        selected->scheduler_gpu_ids_current_snapshot->valid) {
        snapshot = selected->scheduler_gpu_ids_current_snapshot;
    } else {
        selected->scheduler_gpu_ids_current_group = ids_group;
        selected->scheduler_gpu_ids_current_snapshot = nullptr;
        for (auto & candidate : selected->scheduler_gpu_ids_pending) {
            if (!candidate.valid) {
                snapshot = &candidate;
                break;
            }
        }
        if (snapshot) {
            const size_t row_bytes = (size_t)ids_ne0 * sizeof(int32_t);
            snapshot->ids_count = (int)(ids_ne0 * ids_ne1);
            snapshot->ids_group = ids_group;
            snapshot->sources.clear();
            snapshot->valid = true;
            CUDA_CHECK(cudaMemcpy2DAsync(
                snapshot->ids, row_bytes,
                device_ids, ids_nb1,
                row_bytes, (size_t)ids_ne1,
                cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaEventRecord(snapshot->event, stream));
            selected->scheduler_gpu_ids_current_snapshot = snapshot;
            selected->scheduler_gpu_ids_snapshots++;
        }
    }

    if (snapshot) {
        bool have_source = false;
        for (const auto & source : snapshot->sources) {
            if (source.host_base == host_base &&
                source.tensor_size == tensor_size &&
                source.expert_size == expert_size &&
                source.wtype == wtype &&
                source.n_expert == n_expert) {
                have_source = true;
                break;
            }
        }
        if (!have_source) {
            try {
                snapshot->sources.push_back({
                    host_base, tensor_size, expert_size, wtype, n_expert});
            } catch (...) {
                // Metadata is advisory; inference correctness does not depend
                // on recording every source if host allocation fails.
            }
        }
    }

    selected->scheduler_gpu_ids_prepares++;
    if (session->config.stats_every > 0 &&
        selected->scheduler_gpu_ids_prepares % session->config.stats_every == 0) {
        moe_cache_log_stats(*selected);
    }
    return 1;
}

extern "C" int ggml_cuda_moe_cache_direct_materialize(
        void * opaque_backend, const char * tensor_name,
        const void * host_base, size_t tensor_size, size_t expert_size,
        int wtype, int64_t n_expert, const int32_t * ids, int n_ids,
        void * dst_base) {
    if (g_session_suppressed > 0 || g_session_stack.empty() ||
        !opaque_backend || !tensor_name || !host_base || !dst_base ||
        !ids || n_ids < 1 || n_ids > moe_cache_node_rows_max ||
        n_expert <= 0 || expert_size == 0 ||
        (uint64_t)n_expert > SIZE_MAX / expert_size ||
        tensor_size < (size_t)n_expert * expert_size ||
        !moe_cache_tensor_name_supported(tensor_name) ||
        !moe_cache_type_supported((ggml_type)wtype) ||
        !moe_cache_scheduler_direct_type_supported((ggml_type)wtype)) {
        return 0;
    }

    moe_cache_session * session = g_session_stack.back().active;
    if (!session || session->stopping || session->dormant ||
        !session->config.scheduler_copy || !session->config.scheduler_direct ||
        expert_size < session->config.min_expert_bytes) {
        return 0;
    }
    moe_cache_log_configuration(*session);

    ggml_backend_t backend = (ggml_backend_t)opaque_backend;
    if (!ggml_backend_is_cuda(backend)) {
        return 0;
    }
    ggml_backend_cuda_context * cuda_ctx =
        (ggml_backend_cuda_context *)backend->context;
    if (!cuda_ctx) {
        return 0;
    }

    moe_cache_device * selected = nullptr;
    for (const auto & device_ptr : session->devices) {
        if (device_ptr->logical == cuda_ctx->device) {
            selected = device_ptr.get();
            break;
        }
    }
    if (!selected || selected->dead.load()) {
        return 0;
    }

    std::unique_lock<std::mutex> dispatch_lock(selected->dispatch_mu);
    if (selected->dead.load()) {
        return 0;
    }

    ggml_cuda_set_device(selected->logical);
    if (!moe_cache_scheduler_reap_locked(*session, *selected, false)) {
        return 0;
    }

    const char * mapped_host =
        moe_cache_map_host_source(*selected, host_base, tensor_size);
    if (!mapped_host) {
        GGML_ABORT("scheduler direct: failed to map host expert tensor %s", tensor_name);
    }

    const moe_cache_direct_route_key route_key = {
        dst_base, n_expert, expert_size, wtype};
    moe_cache_direct_route * route = nullptr;
    auto route_it = selected->scheduler_direct_routes.find(route_key);
    if (route_it == selected->scheduler_direct_routes.end()) {
        moe_cache_direct_route created;
        if (cudaMalloc((void **)&created.device_pointers,
                       (size_t)n_expert * sizeof(void *)) != cudaSuccess ||
            !created.device_pointers) {
            (void)cudaGetLastError();
            GGML_ABORT("scheduler direct: pointer-table allocation failed for %s", tensor_name);
        }
        try {
            route_it = selected->scheduler_direct_routes.emplace(
                route_key, created).first;
        } catch (...) {
            cudaFree((void *)created.device_pointers);
            GGML_ABORT("scheduler direct: route allocation failed for %s", tensor_name);
        }
    }
    route = &route_it->second;
    if (!route->device_pointers) {
        GGML_ABORT("scheduler direct: missing pointer table for %s", tensor_name);
    }

    if (selected->scheduler_direct_epoch.stream &&
        selected->scheduler_direct_epoch.stream != cuda_ctx->stream()) {
        // A route already captured into a CUDA Graph must never silently
        // switch back to the normal work-buffer kernel on another stream.
        GGML_ABORT("scheduler direct: CUDA stream changed within one scheduler scope");
    }
    selected->scheduler_direct_epoch.stream = cuda_ctx->stream();

    moe_cache_direct_batch batch = {};
    batch.pointer_table = route->device_pointers;
    int inserts_left = session->config.inserts_per_plan;
    bool wake_worker = false;
    size_t pin_begin = 0;
    size_t source_begin = 0;

    {
        std::lock_guard<std::mutex> lock(session->mu);
        if (session->stopping || session->dormant || selected->dead.load()) {
            GGML_ABORT("scheduler direct: cache session became unavailable during materialization");
        }
        pin_begin = selected->scheduler_direct_epoch.pins.size();
        source_begin = selected->scheduler_direct_epoch.host_bases.size();

        try {
            moe_cache_session::active_source & source =
                session->active_sources[host_base];
            source.bytes = std::max(source.bytes, tensor_size);
            source.references++;
            selected->scheduler_direct_epoch.host_bases.push_back(host_base);
        } catch (...) {
            moe_cache_direct_rollback_epoch_locked(
                *session, *selected, pin_begin, source_begin);
            GGML_ABORT("scheduler direct: active-source bookkeeping failed");
        }

        int pool_index = -1;
        try {
            pool_index = moe_cache_discover_pool(
                *session, *selected, host_base, tensor_size,
                expert_size, wtype, n_expert);
        } catch (...) {
            moe_cache_direct_rollback_epoch_locked(
                *session, *selected, pin_begin, source_begin);
            GGML_ABORT("scheduler direct: pool discovery failed for %s", tensor_name);
        }
        moe_cache_pool * pool =
            pool_index >= 0 && pool_index < (int)selected->pools.size()
                ? selected->pools[pool_index].get() : nullptr;

        int32_t previous = -1;
        for (int index = 0; index < n_ids; ++index) {
            const int32_t expert = ids[index];
            if (expert < 0 || expert >= n_expert || expert <= previous ||
                batch.count >= moe_cache_node_rows_max) {
                moe_cache_direct_rollback_epoch_locked(
                    *session, *selected, pin_begin, source_begin);
                GGML_ABORT("scheduler direct: invalid or unstable expert-id set for %s", tensor_name);
            }
            previous = expert;

            int slot_index = -1;
            if (pool && pool->slab) {
                slot_index = moe_cache_lookup_or_queue_locked(
                    *session, *selected, *pool, pool_index,
                    host_base, expert, expert_size,
                    inserts_left, wake_worker);
            }

            const char * host_source =
                mapped_host + (size_t)expert * expert_size;
            char * work_destination =
                (char *)dst_base + (size_t)expert * expert_size;

            moe_cache_direct_desc & desc = batch.desc[batch.count++];
            desc.expert = expert;
            desc.bytes = expert_size;
            if (slot_index >= 0) {
                moe_cache_slot & slot = pool->slots[slot_index];
                slot.readers++;
                moe_cache_lru_remove(*pool, slot_index);
                moe_cache_lru_push_back(*pool, slot_index);
                try {
                    selected->scheduler_direct_epoch.pins.push_back(
                        {pool, slot_index});
                } catch (...) {
                    slot.readers--;
                    moe_cache_direct_rollback_epoch_locked(
                        *session, *selected, pin_begin, source_begin);
                    GGML_ABORT("scheduler direct: pin bookkeeping failed");
                }
                desc.weight_pointer =
                    pool->slab + (size_t)slot_index * expert_size;
                desc.copy_weight = 0;
                selected->hits++;
                selected->scheduler_direct_hits++;
            } else {
                desc.source = host_source;
                desc.destination = work_destination;
                desc.weight_pointer = work_destination;
                desc.copy_weight = 1;
                selected->misses++;
                selected->scheduler_direct_misses++;
                selected->scheduler_direct_miss_bytes += expert_size;
            }
        }
    }

    if (wake_worker) {
        session->cv.notify_one();
    }
    if (batch.count <= 0) {
        std::lock_guard<std::mutex> lock(session->mu);
        moe_cache_direct_rollback_epoch_locked(
            *session, *selected, pin_begin, source_begin);
        GGML_ABORT("scheduler direct: empty materialization batch");
    }

    const dim3 grid(
        moe_cache_copy_blocks_per_desc,
        (unsigned int)batch.count, 1);
    moe_cache_direct_prepare_kernel<<<grid, 256, 0, cuda_ctx->stream()>>>(batch);
    if (!moe_cache_cuda_ok(
            *selected, cudaPeekAtLastError(),
            "scheduler direct prepare kernel", true)) {
        CUDA_CHECK(cudaStreamSynchronize(cuda_ctx->stream()));
        std::lock_guard<std::mutex> lock(session->mu);
        moe_cache_direct_rollback_epoch_locked(
            *session, *selected, pin_begin, source_begin);
        GGML_ABORT("scheduler direct: prepare kernel launch failed");
    }

    selected->scheduler_direct_prepares++;
    if (session->config.stats_every > 0 &&
        selected->scheduler_direct_prepares % session->config.stats_every == 0) {
        moe_cache_log_stats(*selected);
    }
    return 1;
}




int ggml_cuda_moe_cache_direct_mmv(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0,
        const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst) {
    if (g_session_suppressed > 0 || g_session_stack.empty() ||
        !src0 || !src1 || !ids || !dst ||
        src1->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32 ||
        dst->type != GGML_TYPE_F32 || !ggml_is_quantized(src0->type) ||
        dst->ne[2] < 1 || dst->ne[2] > MMVQ_MAX_BATCH_SIZE ||
        dst->ne[3] != 1 || src1->ne[2] != dst->ne[2] ||
        src1->ne[3] != 1 || src1->ne[1] < 1 || dst->ne[1] < 1 ||
        dst->ne[1] > moe_cache_node_rows_max ||
        ids->ne[0] != dst->ne[1] || ids->ne[1] != dst->ne[2] ||
        ids->nb[0] != sizeof(int32_t)) {
        return 0;
    }

    moe_cache_session * session = g_session_stack.back().active;
    if (!session || session->stopping || session->dormant ||
        !session->config.scheduler_copy || !session->config.scheduler_direct) {
        return 0;
    }

    moe_cache_device * selected = nullptr;
    for (const auto & device_ptr : session->devices) {
        if (device_ptr->logical == ctx.device) {
            selected = device_ptr.get();
            break;
        }
    }
    if (!selected || selected->dead.load()) {
        return 0;
    }

    const moe_cache_direct_route_key route_key = {
        src0->data, src0->ne[2], src0->nb[2], (int)src0->type};
    auto found = selected->scheduler_direct_routes.find(route_key);
    if (found == selected->scheduler_direct_routes.end() ||
        !found->second.device_pointers) {
        return 0;
    }
    const moe_cache_direct_route & route = found->second;

    cudaStream_t stream = ctx.stream();
    const int64_t ne10 = src1->ne[0];
    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    const size_t q8_bytes =
        (size_t)src1->ne[3] * (size_t)src1->ne[2] *
        (size_t)src1->ne[1] * (size_t)ne10_padded *
        sizeof(block_q8_1) / QK8_1;
    ggml_cuda_pool_alloc<char> act_q8(ctx.pool(), q8_bytes);

    const size_t ts_src1 = ggml_type_size(src1->type);
    const int64_t s11 = src1->nb[1] / ts_src1;
    const int64_t s12 = src1->nb[2] / ts_src1;
    const int64_t s13 = src1->nb[3] / ts_src1;
    quantize_row_q8_1_cuda(
        (const float *)src1->data, nullptr, act_q8.get(), src0->type,
        ne10, s11, s12, s13, ne10_padded,
        src1->ne[1], src1->ne[2], src1->ne[3], stream);

    GGML_ASSERT(dst->nb[1] / sizeof(float) == (size_t)dst->ne[0]);
    if (dst->ne[2] == 1) {
        moe_cache_direct_mmv_ptrs(
            route.device_pointers, src0->type, act_q8.get(),
            (const int32_t *)ids->data, (float *)dst->data,
            src0->ne[0], src0->ne[1], src0->ne[2],
            dst->ne[1], src1->ne[1], stream);
    } else {
        moe_cache_direct_mmv_moe_ptrs(
            route.device_pointers, src0->type, act_q8.get(),
            (const int32_t *)ids->data, (float *)dst->data,
            src0->ne[0], src0->ne[1], src0->ne[2],
            dst->ne[1], dst->ne[2], src1->ne[1],
            ids->nb[1] / sizeof(int32_t),
            dst->nb[1] / sizeof(float),
            dst->nb[2] / sizeof(float), stream);
    }

    if (!moe_cache_cuda_ok(
            *selected, cudaPeekAtLastError(),
            "scheduler direct expert matvec", true)) {
        GGML_ABORT("scheduler direct expert matvec launch failed");
    }
    selected->scheduler_direct_executes++;
    return 1;
}

static int moe_cache_plan(
        void * opaque, const int32_t * ids, int n_ids, int32_t * slot_indices) {
    moe_cache_node * node = (moe_cache_node *)opaque;
    if (!node || !ids || !slot_indices || n_ids < 0 ||
        n_ids > moe_cache_node_rows_max || node->planned) {
        return 0;
    }
    node->planned = true;
    for (int index = 0; index < n_ids; index++) {
        slot_indices[index] = -1;
    }

    moe_cache_session & session = *node->session;
    moe_cache_device & device = *node->device;
    moe_cache_pool & pool = *node->pool;
    int hits = 0;
    int miss_indices[moe_cache_node_rows_max];
    int n_misses = 0;
    int inserts_left = session.config.inserts_per_plan;
    bool wake_worker = false;

    std::unique_lock<std::mutex> lock(session.mu);
    if (session.stopping) {
        return 0;
    }
    for (int index = 0; index < n_ids; index++) {
        const int32_t expert = ids[index];
        if (expert < 0 || expert >= node->n_expert || device.dead.load()) {
            continue;
        }

        const int slot_index = moe_cache_lookup_or_queue_locked(
                session, device, pool, node->pool_index,
                node->host_base, expert, node->expert_size,
                inserts_left, wake_worker);
        if (slot_index < 0) {
            device.misses++;
            if (n_misses < moe_cache_node_rows_max) {
                miss_indices[n_misses++] = index;
            }
            continue;
        }

        moe_cache_slot & slot = pool.slots[slot_index];
        slot.readers++;
        moe_cache_lru_remove(pool, slot_index);
        moe_cache_lru_push_back(pool, slot_index);
        node->pins[node->n_pins++] = {&pool, slot_index};
        slot_indices[index] = slot_index;
        device.hits++;
        hits++;
    }
    if (hits == n_ids) {
        int rows = moe_cache_overlap_rows(*node, n_ids);
        for (int index = n_ids - 1; index >= 0 && rows > 0; index--) {
            if (slot_indices[index] < 0 || node->n_pins <= 0) {
                continue;
            }
            const moe_cache_pin & pin = node->pins[node->n_pins - 1];
            if (pin.slot != slot_indices[index]) {
                continue;
            }
            moe_cache_slot & slot = pin.pool->slots[pin.slot];
            if (slot.readers > 0) {
                slot.readers--;
            }
            node->n_pins--;
            slot_indices[index] = -1;
            hits--;
            rows--;
            device.overlap_rows++;
        }
    }

    // Preserve stock overlap semantics: only an all-resident batch may hand
    // resident rows back to CPU. After that, move a configurable percentage
    // of genuine misses to transient GPU execution.
    if (session.config.gpu_miss_percent > 0 && n_misses > 0 &&
        pool.n_slots <= INT_MAX - node->n_expert) {
        int gpu_rows =
            (n_misses * session.config.gpu_miss_percent + 99) / 100;
        gpu_rows = std::min(gpu_rows, n_misses);

        for (int miss = 0; miss < gpu_rows; miss++) {
            const int index = miss_indices[miss];
            const int32_t expert = ids[index];
            if (expert < 0 || expert >= node->n_expert) {
                continue;
            }

            // Synthetic slot range: [pool.n_slots, pool.n_slots+n_expert).
            // The CPU hook treats it as GPU work; dispatch() recognizes it.
            slot_indices[index] = pool.n_slots + expert;
            node->gpu_miss_rows++;
            device.gpu_miss_rows++;
            device.gpu_miss_bytes += node->expert_size;
            hits++;
        }
    }

    device.nodes++;
    lock.unlock();
    if (wake_worker) {
        session.cv.notify_all();
    }
    return hits;
}

static int moe_cache_dispatch_internal(
        void * opaque, int wtype, int64_t n_in, int64_t n_out, int n_hits,
        const int32_t * slot_indices, const float * const * act_rows,
        moe_cache_pool * gate_pool, const int32_t * gate_slot_indices,
        int glu_op, float up_min, float up_max,
        float gate_min, float gate_max) {
    moe_cache_node * node = (moe_cache_node *)opaque;
    const bool fused = gate_pool != nullptr;
    if (!node || !node->planned || node->dispatched || n_hits <= 0 ||
        n_hits > moe_cache_node_rows_max ||
        (n_hits - node->gpu_miss_rows) * (fused ? 2 : 1) != node->n_pins ||
        !slot_indices || !act_rows ||
        (fused && (!gate_slot_indices ||
                   gate_pool->wtype != wtype ||
                   gate_pool->expert_size != node->expert_size ||
                   glu_op != GGML_GLU_OP_SWIGLU ||
                   std::isnan(up_min) || std::isnan(up_max) ||
                   std::isnan(gate_min) || std::isnan(gate_max) ||
                   up_min > up_max || gate_min > gate_max)) ||
        wtype != node->wtype || n_in != node->n_in || n_out != node->n_out ||
        n_in > INT_MAX || n_out > INT_MAX ||
        n_in > INT64_MAX - (MATRIX_ROW_PADDING - 1)) {
        return 0;
    }

    moe_cache_session & session = *node->session;
    moe_cache_device & device = *node->device;
    moe_cache_pool & pool = *node->pool;

    if (node->gpu_miss_rows < 0 || node->gpu_miss_rows > n_hits) {
        return 0;
    }

    if (device.dead.load() || moe_cache_fail(session, "dispatch")) {
        std::lock_guard<std::mutex> lock(session.mu);
        device.dispatch_failures++;
        return 0;
    }

    ggml_cuda_set_device(device.logical);
    if (!device.compute_stream) {
        if (!moe_cache_cuda_ok(device,
                    cudaStreamCreateWithFlags(&device.compute_stream, cudaStreamNonBlocking),
                    "compute stream creation", true)) {
            std::lock_guard<std::mutex> lock(session.mu);
            device.dispatch_failures++;
            return 0;
        }
    }

    const float * unique_act_rows[moe_cache_node_rows_max];
    int32_t activation_indices[moe_cache_node_rows_max];
    bool transient_rows[moe_cache_node_rows_max] = {};
    int activation_rows = 0;
    int transient_count = 0;
    bool modulo_activation = true;

    for (int index = 0; index < n_hits; index++) {
        const int32_t encoded = slot_indices[index];
        const int32_t gate_encoded =
            fused ? gate_slot_indices[index] : -1;

        const bool up_transient =
            encoded >= pool.n_slots &&
            encoded < pool.n_slots + node->n_expert;
        const bool gate_transient =
            fused &&
            gate_encoded >= gate_pool->n_slots &&
            gate_encoded < gate_pool->n_slots + node->n_expert;
        const bool transient =
            up_transient && (!fused || gate_transient);

        if (encoded < 0 ||
            (!up_transient && encoded >= pool.n_slots) ||
            (fused &&
             (gate_encoded < 0 ||
              (!gate_transient && gate_encoded >= gate_pool->n_slots) ||
              up_transient != gate_transient ||
              (transient &&
               encoded - pool.n_slots !=
                   gate_encoded - gate_pool->n_slots))) ||
            !act_rows[index]) {
            return 0;
        }

        transient_rows[index] = transient;
        transient_count += transient ? 1 : 0;

        int activation = 0;
        while (activation < activation_rows &&
               unique_act_rows[activation] != act_rows[index]) {
            activation++;
        }
        if (activation == activation_rows) {
            unique_act_rows[activation_rows++] = act_rows[index];
        }
        activation_indices[index] = activation;
        modulo_activation &= activation == index % activation_rows;
    }

    if (transient_count != node->gpu_miss_rows) {
        return 0;
    }

    const int64_t padded_n_in =
        ((n_in + MATRIX_ROW_PADDING - 1) / MATRIX_ROW_PADDING) * MATRIX_ROW_PADDING;
    const size_t type_size = ggml_type_size((ggml_type)wtype);
    if (type_size == 0 || node->expert_size % type_size != 0 ||
        ggml_row_size((ggml_type)wtype, n_in) % type_size != 0 ||
        node->expert_size / type_size > INT_MAX ||
        ggml_row_size((ggml_type)wtype, n_in) / type_size > INT_MAX ||
        padded_n_in / QK8_1 > INT_MAX ||
        (uint64_t)activation_rows * (padded_n_in / QK8_1) > INT_MAX ||
        (uint64_t)n_out * n_hits > INT_MAX ||
        (uint64_t)(node->expert_size / type_size) * pool.n_slots > INT_MAX) {
        return 0;
    }

    if (n_in > INT64_MAX / activation_rows ||
        (uint64_t)n_in * activation_rows > SIZE_MAX / sizeof(float) ||
        n_out > INT64_MAX / n_hits ||
        (uint64_t)n_out * n_hits > SIZE_MAX / sizeof(float) ||
        (uint64_t)activation_rows * (padded_n_in / QK8_1) >
            SIZE_MAX / sizeof(block_q8_1)) {
        return 0;
    }
    const bool use_activation_map =
        session.config.force_dedicated_mmv ||
        transient_count > 0 ||
        !modulo_activation;
    const size_t id_arrays = 1 + (fused ? 1 : 0) +
        (use_activation_map ? 1 : 0);
    const size_t ids_bytes = (size_t)n_hits * sizeof(int32_t) * id_arrays;
    const size_t act_bytes = (size_t)activation_rows * n_in * sizeof(float);
    if (ids_bytes > SIZE_MAX - act_bytes) {
        return 0;
    }
    const size_t input_bytes = ids_bytes + act_bytes;
    const size_t q8_bytes =
        (size_t)activation_rows * (padded_n_in / QK8_1) * sizeof(block_q8_1);
    const size_t out_bytes = (size_t)n_hits * n_out * sizeof(float);

    const size_t desired_caps[] = {
        moe_cache_growth_capacity(device.d_input_cap, input_bytes),
        moe_cache_growth_capacity(device.act_q8_cap, q8_bytes),
        moe_cache_growth_capacity(device.d_out_cap, out_bytes),
    };
    size_t scratch_bytes = 0;
    for (size_t capacity : desired_caps) {
        if (capacity == 0 || capacity > SIZE_MAX - scratch_bytes) {
            std::lock_guard<std::mutex> lock(session.mu);
            device.dispatch_failures++;
            return 0;
        }
        scratch_bytes += capacity;
    }
    {
        std::lock_guard<std::mutex> lock(session.mu);
        if (scratch_bytes > device.budget_limit ||
            device.allocated_bytes > device.budget_limit - scratch_bytes) {
            device.dispatch_failures++;
            return 0;
        }
    }

    if (!moe_cache_grow_host(device, (void **)&device.h_input, device.h_input_cap,
                             input_bytes, "input host allocation") ||
        !moe_cache_grow_device(device, (void **)&device.d_input, device.d_input_cap,
                               input_bytes, "input device allocation") ||
        !moe_cache_grow_device(device, &device.d_act_q8, device.act_q8_cap,
                               q8_bytes, "q8 activation allocation") ||
        !moe_cache_grow_device(device, (void **)&device.d_out, device.d_out_cap,
                               out_bytes, "output device allocation") ||
        !moe_cache_grow_host(device, (void **)&device.h_out, device.h_out_cap,
                             out_bytes, "output host allocation")) {
        std::lock_guard<std::mutex> lock(session.mu);
        device.dispatch_failures++;
        device.dead.store(true);
        return 0;
    }

    const bool tensor_bytes_valid =
        transient_count > 0 &&
        node->expert_size > 0 &&
        (uint64_t)node->n_expert <= SIZE_MAX / node->expert_size;
    const size_t tensor_bytes =
        tensor_bytes_valid
            ? (size_t)node->n_expert * node->expert_size
            : 0;

    const char * mapped_up =
        session.config.gpu_miss_direct && tensor_bytes_valid
            ? moe_cache_map_host_source(
                device, node->host_base, tensor_bytes)
            : nullptr;
    const char * mapped_gate =
        session.config.gpu_miss_direct &&
        fused && tensor_bytes_valid
            ? moe_cache_map_host_source(
                device, node->host_base2, tensor_bytes)
            : nullptr;

    const bool use_direct_host =
        transient_count > 0 &&
        session.config.gpu_miss_direct &&
        mapped_up && (!fused || mapped_gate);

    if (transient_count > 0 && !use_direct_host) {
        const size_t slabs_per_row = fused ? 2 : 1;
        if (node->expert_size > SIZE_MAX / (size_t)transient_count ||
            (size_t)transient_count * node->expert_size >
                SIZE_MAX / slabs_per_row) {
            return 0;
        }
        const size_t required =
            (size_t)transient_count * node->expert_size * slabs_per_row;

        if (required > device.d_gpu_miss_cap) {
            char * fresh = nullptr;
            if (!moe_cache_cuda_ok(
                    device, cudaMalloc((void **)&fresh, required),
                    "GPU miss transient allocation", false)) {
                std::lock_guard<std::mutex> lock(session.mu);
                device.dispatch_failures++;
                return 0;
            }
            if (device.d_gpu_miss) {
                cudaFree(device.d_gpu_miss);
            }
            device.d_gpu_miss = fresh;
            device.d_gpu_miss_cap = required;
        }
    }

    int32_t * h_ids = (int32_t *)device.h_input;
    float * h_act = (float *)(device.h_input + ids_bytes);
    int32_t * d_ids = (int32_t *)device.d_input;
    float * d_act = (float *)(device.d_input + ids_bytes);
    int resident_rows = 0;
    int transient_pos = 0;
    int grouped = 0;
    int32_t transient_experts[moe_cache_node_rows_max] = {};

    for (int pass = 0; pass < 2; pass++) {
        for (int index = 0; index < n_hits; index++) {
            const bool transient = transient_rows[index];
            if ((pass == 0 && transient) || (pass == 1 && !transient)) {
                continue;
            }

            node->output_map[grouped] = index;

            if (transient) {
                const int32_t expert = slot_indices[index] - pool.n_slots;
                const int32_t gate_expert =
                    fused ? gate_slot_indices[index] - gate_pool->n_slots
                          : expert;
                if (expert < 0 || expert >= node->n_expert ||
                    gate_expert != expert) {
                    return 0;
                }
                transient_experts[transient_pos] = expert;
                h_ids[grouped] =
                    use_direct_host ? expert : transient_pos;
                if (fused) {
                    h_ids[n_hits + grouped] =
                        use_direct_host ? expert : transient_pos;
                }
                transient_pos++;
            } else {
                h_ids[grouped] = slot_indices[index];
                if (fused) {
                    h_ids[n_hits + grouped] = gate_slot_indices[index];
                }
                resident_rows++;
            }
            if (use_activation_map) {
                h_ids[(fused ? 2 : 1) * n_hits + grouped] =
                    activation_indices[index];
            }
            grouped++;
        }
    }

    if (grouped != n_hits ||
        resident_rows + transient_count != n_hits ||
        transient_pos != transient_count) {
        return 0;
    }
    for (int index = 0; index < activation_rows; index++) {
        memcpy(h_act + (size_t)index * n_in,
               unique_act_rows[index], n_in * sizeof(float));
    }

    (void)cudaGetLastError();
    bool ok =
        moe_cache_cuda_ok(device, cudaMemcpyAsync(
                device.d_input, device.h_input, input_bytes,
                cudaMemcpyHostToDevice, device.compute_stream), "input upload", true);

    if (ok && transient_count > 0 && !use_direct_host) {
        char * transient_gate =
            fused
                ? device.d_gpu_miss +
                    (size_t)transient_count * node->expert_size
                : nullptr;

        const char * copy_mapped_up =
            session.config.gpu_miss_copy_kernel && tensor_bytes_valid
                ? moe_cache_map_host_source(
                    device, node->host_base, tensor_bytes)
                : nullptr;
        const char * copy_mapped_gate =
            session.config.gpu_miss_copy_kernel &&
            fused && tensor_bytes_valid
                ? moe_cache_map_host_source(
                    device, node->host_base2, tensor_bytes)
                : nullptr;

        const bool use_copy_kernel =
            session.config.gpu_miss_copy_kernel &&
            copy_mapped_up && (!fused || copy_mapped_gate);

        if (use_copy_kernel) {
            moe_cache_copy_batch batch = {};

            for (int miss = 0; miss < transient_count; miss++) {
                const int32_t expert = transient_experts[miss];

                if (batch.count >= moe_cache_copy_desc_max) {
                    ok = false;
                    break;
                }

                batch.desc[batch.count++] = {
                    copy_mapped_up + (size_t)expert * node->expert_size,
                    device.d_gpu_miss +
                        (size_t)miss * node->expert_size,
                    node->expert_size};

                if (fused) {
                    if (batch.count >= moe_cache_copy_desc_max) {
                        ok = false;
                        break;
                    }

                    batch.desc[batch.count++] = {
                        copy_mapped_gate +
                            (size_t)expert * node->expert_size,
                        transient_gate +
                            (size_t)miss * node->expert_size,
                        node->expert_size};
                }
            }

            if (ok && batch.count > 0) {
                const dim3 grid(
                    moe_cache_copy_blocks_per_desc,
                    (unsigned int)batch.count,
                    1);
                moe_cache_copy_batch_kernel<<<grid, 256, 0, device.compute_stream>>>(
                    batch);
                ok = moe_cache_cuda_ok(
                        device, cudaPeekAtLastError(),
                        "GPU miss copy kernel", true);

                if (ok) {
                    device.gpu_miss_copy_kernel_calls++;
                    device.gpu_miss_copy_kernel_descs += batch.count;
                }
            }
        } else {
            device.gpu_miss_copy_fallbacks++;

            for (int miss = 0; miss < transient_count; miss++) {
                const int32_t expert = transient_experts[miss];
                const void * source =
                    (const char *)node->host_base +
                    (size_t)expert * node->expert_size;
                void * destination =
                    device.d_gpu_miss +
                    (size_t)miss * node->expert_size;

                if (!moe_cache_cuda_ok(
                        device, cudaMemcpyAsync(
                            destination, source, node->expert_size,
                            cudaMemcpyHostToDevice, device.compute_stream),
                        "GPU miss transient upload fallback", true)) {
                    ok = false;
                    break;
                }

                if (fused) {
                    const void * gate_source =
                        (const char *)node->host_base2 +
                        (size_t)expert * node->expert_size;
                    void * gate_destination =
                        transient_gate +
                        (size_t)miss * node->expert_size;

                    if (!moe_cache_cuda_ok(
                            device, cudaMemcpyAsync(
                                gate_destination, gate_source,
                                node->expert_size,
                                cudaMemcpyHostToDevice,
                                device.compute_stream),
                            "GPU miss fused gate upload fallback", true)) {
                        ok = false;
                        break;
                    }
                }
            }
        }
    }

    if (ok && use_direct_host) {
        device.gpu_miss_direct_calls++;
        device.gpu_miss_direct_rows += transient_count;
        if (fused) {
            device.gpu_miss_direct_fused_rows += transient_count;
        }
    }

    if (ok) {
        quantize_row_q8_1_cuda(
                d_act, nullptr, device.d_act_q8, (ggml_type)wtype,
                n_in, n_in, (int64_t)activation_rows * n_in,
                (int64_t)activation_rows * n_in, padded_n_in,
                activation_rows, 1, 1, device.compute_stream);
        ok = moe_cache_cuda_ok(
                device, cudaPeekAtLastError(), "activation quantization", true);
    }
    if (ok) {
        if (fused) {
            int32_t * d_activation_map =
                use_activation_map ? d_ids + 2 * n_hits : nullptr;

            if (resident_rows > 0) {
                ggml_cuda_moe_cache_mmv_fused(
                        pool.slab, gate_pool->slab, (ggml_type)wtype,
                        (const char *)device.d_act_q8,
                        d_ids, d_ids + n_hits,
                        d_activation_map,
                        device.d_out, n_in, n_out,
                        (int64_t)pool.expert_size,
                        resident_rows, activation_rows,
                        up_min, up_max, gate_min, gate_max,
                        device.compute_stream);
            }

            if (transient_count > 0) {
                const char * transient_up =
                    use_direct_host ? mapped_up : device.d_gpu_miss;
                const char * transient_gate =
                    use_direct_host
                        ? mapped_gate
                        : device.d_gpu_miss +
                            (size_t)transient_count * node->expert_size;
                ggml_cuda_moe_cache_mmv_fused(
                        transient_up, transient_gate,
                        (ggml_type)wtype,
                        (const char *)device.d_act_q8,
                        d_ids + resident_rows,
                        d_ids + n_hits + resident_rows,
                        d_activation_map
                            ? d_activation_map + resident_rows
                            : nullptr,
                        device.d_out + (size_t)resident_rows * n_out,
                        n_in, n_out,
                        (int64_t)pool.expert_size,
                        transient_count, activation_rows,
                        up_min, up_max, gate_min, gate_max,
                        device.compute_stream);
            }
        } else {
            int32_t * d_activation_map =
                use_activation_map ? d_ids + n_hits : nullptr;

            if (resident_rows > 0) {
                ggml_cuda_moe_cache_mmv(
                        pool.slab, (ggml_type)wtype,
                        (const char *)device.d_act_q8,
                        d_ids,
                        d_activation_map,
                        device.d_out,
                        n_in, n_out, pool.n_slots,
                        (int64_t)pool.expert_size,
                        resident_rows, activation_rows,
                        device.compute_stream);
            }

            if (transient_count > 0) {
                ggml_cuda_moe_cache_mmv(
                        use_direct_host ? mapped_up : device.d_gpu_miss,
                        (ggml_type)wtype,
                        (const char *)device.d_act_q8,
                        d_ids + resident_rows,
                        d_activation_map ? d_activation_map + resident_rows : nullptr,
                        device.d_out + (size_t)resident_rows * n_out,
                        n_in, n_out,
                        use_direct_host ? node->n_expert : transient_count,
                        (int64_t)pool.expert_size,
                        transient_count, activation_rows,
                        device.compute_stream);
            }
        }
        ok = moe_cache_cuda_ok(
                device, cudaPeekAtLastError(), "expert matvec launch", true);
    }

    if (!ok) {
        cudaStreamSynchronize(device.compute_stream);
        std::lock_guard<std::mutex> lock(session.mu);
        device.dispatch_failures++;
        return 0;
    }

    device.activation_dedup += n_hits - activation_rows;
    node->dispatched = true;
    return 1;
}

static int moe_cache_dispatch(
        void * opaque, int wtype, int64_t n_in, int64_t n_out, int n_hits,
        const int32_t * slot_indices, const float * const * act_rows) {
    return moe_cache_dispatch_internal(
            opaque, wtype, n_in, n_out, n_hits,
            slot_indices, act_rows, nullptr, nullptr, -1,
            0.0f, 0.0f, 0.0f, 0.0f);
}

static int moe_cache_collect(
        void * opaque, int n_hits, float * const * dst_rows, int64_t n_out) {
    moe_cache_node * node = (moe_cache_node *)opaque;
    const int pins_per_hit = node && node->host_base2 ? 2 : 1;
    if (!node || !node->dispatched || n_hits <= 0 ||
        n_hits > moe_cache_node_rows_max ||
        (n_hits - node->gpu_miss_rows) * pins_per_hit != node->n_pins ||
        !dst_rows || n_out != node->n_out) {
        return 0;
    }
    for (int index = 0; index < n_hits; index++) {
        if (!dst_rows[index]) {
            return 0;
        }
    }

    moe_cache_session & session = *node->session;
    moe_cache_device & device = *node->device;
    ggml_cuda_set_device(device.logical);

    bool ok = !device.dead.load();
    if (moe_cache_fail(session, "collect")) {
        ok = false;
    }
    const size_t bytes = (size_t)n_hits * n_out * sizeof(float);
    if (ok) {
        ok = moe_cache_cuda_ok(device, cudaMemcpyAsync(
                device.h_out, device.d_out, bytes,
                cudaMemcpyDeviceToHost, device.compute_stream), "output download", true);
    }
    if (ok) {
        ok = moe_cache_cuda_ok(
                device, cudaStreamSynchronize(device.compute_stream),
                "output synchronization", true);
    } else {
        cudaStreamSynchronize(device.compute_stream);
    }
    node->dispatched = false;

    if (ok) {
        if (node->gpu_miss_rows == 0) {
            for (int index = 0; index < n_hits; index++) {
                memcpy(dst_rows[index], device.h_out + (size_t)index * n_out,
                       n_out * sizeof(float));
            }
        } else {
            for (int index = 0; index < n_hits; index++) {
                const int original = node->output_map[index];
                if (original < 0 || original >= n_hits) {
                    ok = false;
                    break;
                }
                memcpy(dst_rows[original],
                       device.h_out + (size_t)index * n_out,
                       n_out * sizeof(float));
            }
        }
    }

    {
        std::lock_guard<std::mutex> lock(session.mu);
        if (!ok) {
            device.collect_failures++;
        }
        device.collect_calls++;
        if (session.config.stats_every > 0 &&
            device.collect_calls % session.config.stats_every == 0) {
            moe_cache_log_stats(device);
        }
    }
    return ok ? 1 : 0;
}

static void moe_cache_end(void * opaque) {
    std::unique_ptr<moe_cache_node> node((moe_cache_node *)opaque);
    if (!node) {
        return;
    }

    if (node->dispatched) {
        ggml_cuda_set_device(node->device->logical);
        moe_cache_cuda_ok(
                *node->device, cudaStreamSynchronize(node->device->compute_stream),
                "end synchronization", true);
        node->dispatched = false;
    }

    moe_cache_session & session = *node->session;
    const bool trim = node->device->dead.load();
    {
        std::lock_guard<std::mutex> lock(session.mu);
        for (int index = 0; index < node->n_pins; index++) {
            const moe_cache_pin & pin = node->pins[index];
            if (pin.pool && pin.slot >= 0 && pin.slot < pin.pool->n_slots) {
                moe_cache_slot & slot = pin.pool->slots[pin.slot];
                if (slot.readers > 0) {
                    slot.readers--;
                }
            }
        }
        for (const void * base : {node->host_base, node->host_base2}) {
            if (!base) {
                continue;
            }
            auto source = session.active_sources.find(base);
            if (source != session.active_sources.end() &&
                --source->second.references == 0) {
                session.active_sources.erase(source);
            }
        }
        session.active_nodes--;
        session.idle_cv.notify_all();
    }
    if (trim && node->dispatch_lock.owns_lock()) {
        node->dispatch_lock.unlock();
        moe_cache_trim_session(session, node->device->physical);
    }
}

static void * moe_cache_fused_begin(
        const ggml_moe_cache_tensor_desc * up,
        const ggml_moe_cache_tensor_desc * gate,
        int glu_op, float up_min, float up_max,
        float gate_min, float gate_max,
        const int32_t * ids, int n_ids, int64_t n_tokens,
        const float * const * act_rows, uint64_t * hit_mask) {
    if (hit_mask) {
        *hit_mask = 0;
    }
    if (g_session_suppressed > 0 || g_session_stack.empty() ||
        !up || !gate || !ids || !act_rows || !hit_mask ||
        !up->name || !gate->name || !up->data || !gate->data ||
        up->data == gate->data ||
        !moe_cache_tensor_name_supported(up->name) ||
        !moe_cache_tensor_name_supported(gate->name) ||
        glu_op != GGML_GLU_OP_SWIGLU ||
        std::isnan(up_min) || std::isnan(up_max) ||
        std::isnan(gate_min) || std::isnan(gate_max) ||
        up_min > up_max || gate_min > gate_max ||
        n_ids < 1 || n_ids > moe_cache_node_rows_max ||
        n_tokens < 1 || n_tokens > n_ids || n_ids % n_tokens != 0 ||
        up->expert_size != gate->expert_size ||
        up->n_in != gate->n_in || up->n_out != gate->n_out ||
        up->n_expert != gate->n_expert || up->type != gate->type ||
        up->n_in <= 0 || up->n_out <= 0 ||
        up->n_expert <= 0 ||
        up->type == GGML_TYPE_NVFP4 ||
        !moe_cache_type_supported((ggml_type)up->type)) {
        return nullptr;
    }
    for (int index = 0; index < n_ids; index++) {
        if (!act_rows[index]) {
            return nullptr;
        }
    }

    moe_cache_session * session = g_session_stack.back().active;
    if (!session || session->stopping || session->dormant) {
        return nullptr;
    }
    moe_cache_log_configuration(*session);
    if (up->expert_size < session->config.min_expert_bytes) {
        return nullptr;
    }
    if (n_tokens > session->config.max_batch) {
        bool expected = false;
        if (session->batch_bypass_announced.compare_exchange_strong(expected, true)) {
            MOE_CACHE_LOG("[moe-cache] bypassing %lld-token fused node above max-batch=%d; increase the limit only when decode or speculative batches benefit on this hardware\n",
                    (long long)n_tokens, session->config.max_batch);
        }
        return nullptr;
    }

    const size_t row_size = ggml_row_size((ggml_type)up->type, up->n_in);
    if (row_size == 0 || (uint64_t)up->n_out > SIZE_MAX / row_size ||
        up->expert_size != (size_t)up->n_out * row_size ||
        (uint64_t)up->n_expert > SIZE_MAX / up->expert_size) {
        return nullptr;
    }
    const size_t tensor_size = (size_t)up->n_expert * up->expert_size;

    int up_layer = -1;
    int gate_layer = -1;
    if (!moe_cache_layer_number(up->name, up_layer) ||
        !moe_cache_layer_number(gate->name, gate_layer) ||
        up_layer != gate_layer) {
        return nullptr;
    }

    std::unique_ptr<moe_cache_node> node(new (std::nothrow) moe_cache_node());
    if (!node) {
        return nullptr;
    }

    moe_cache_device * selected = nullptr;
    moe_cache_pool * pool = nullptr;
    int pool_index = -1;
    {
        std::lock_guard<std::mutex> lock(session->mu);
        if (session->stopping || session->dormant) {
            return nullptr;
        }

        auto route_for = [&](const void * tensor) {
            const auto tensor_route = session->tensor_devices.find(tensor);
            if (tensor_route != session->tensor_devices.end()) {
                return tensor_route->second;
            }
            const auto layer_route = session->layer_devices.find(up_layer);
            return layer_route != session->layer_devices.end()
                ? layer_route->second : -1;
        };
        const int up_device = route_for(up->data);
        const int gate_device = route_for(gate->data);
        if (up_device < 0 || up_device != gate_device) {
            return nullptr;
        }
        for (const auto & device_ptr : session->devices) {
            if (device_ptr->physical == up_device) {
                selected = device_ptr.get();
                break;
            }
        }
        if (!selected || selected->dead.load()) {
            return nullptr;
        }
        pool_index = moe_cache_find_pool(
                *selected, up->expert_size, up->type);
        if (pool_index < 0 || pool_index >= (int)selected->pools.size()) {
            return nullptr;
        }
        pool = selected->pools[pool_index].get();
        if (!pool || !pool->slab) {
            return nullptr;
        }

        auto acquire_source = [&](const void * base) {
            try {
                moe_cache_session::active_source & source =
                    session->active_sources[base];
                source.bytes = std::max(source.bytes, tensor_size);
                source.references++;
                return true;
            } catch (...) {
                return false;
            }
        };
        if (!acquire_source(up->data)) {
            return nullptr;
        }
        if (!acquire_source(gate->data)) {
            auto source = session->active_sources.find(up->data);
            if (source != session->active_sources.end() &&
                --source->second.references == 0) {
                session->active_sources.erase(source);
            }
            return nullptr;
        }
        session->active_nodes++;
    }

    auto release_active = [&] {
        std::lock_guard<std::mutex> lock(session->mu);
        for (const void * base : {up->data, gate->data}) {
            auto source = session->active_sources.find(base);
            if (source != session->active_sources.end() &&
                --source->second.references == 0) {
                session->active_sources.erase(source);
            }
        }
        session->active_nodes--;
        session->idle_cv.notify_all();
    };

    std::unique_lock<std::mutex> dispatch_lock;
    try {
        dispatch_lock = std::unique_lock<std::mutex>(
                selected->dispatch_mu, std::try_to_lock);
    } catch (...) {
        release_active();
        return nullptr;
    }
    if (!dispatch_lock.owns_lock() || selected->dead.load()) {
        release_active();
        return nullptr;
    }

    node->session = session;
    node->device = selected;
    node->pool = pool;
    node->pool_index = pool_index;
    node->host_base = up->data;
    node->host_base2 = gate->data;
    node->expert_size = up->expert_size;
    node->n_in = up->n_in;
    node->n_out = up->n_out;
    node->n_expert = up->n_expert;
    node->n_tokens = n_tokens;
    node->wtype = up->type;
    node->dispatch_lock = std::move(dispatch_lock);
    node->planned = true;

    int32_t up_slots[moe_cache_node_rows_max];
    int32_t gate_slots[moe_cache_node_rows_max];
    int32_t resident_up[moe_cache_node_rows_max];
    int32_t resident_gate[moe_cache_node_rows_max];
    const float * hit_acts[moe_cache_node_rows_max];

    // Compact fused rows must remain in the same ascending original-row
    // order used by the CPU hook when it constructs state->rows[] from
    // hit_mask. V5 appended transient misses after all residents and could
    // therefore swap expert outputs between routed rows.
    int hit_indices[moe_cache_node_rows_max];

    uint64_t mask = 0;
    int hits = 0;
    int fused_miss_indices[moe_cache_node_rows_max];
    int n_fused_misses = 0;
    int inserts_left = session->config.inserts_per_plan;
    bool wake_worker = false;
    {
        std::unique_lock<std::mutex> lock(session->mu);
        if (!session->stopping && !selected->dead.load()) {
            for (int index = 0; index < n_ids; index++) {
                resident_up[index] = -1;
                resident_gate[index] = -1;
                const int32_t expert = ids[index];
                if (expert < 0 || expert >= up->n_expert) {
                    continue;
                }
                const auto up_found = pool->map.find({up->data, expert});
                const auto gate_found = pool->map.find({gate->data, expert});
                if (up_found != pool->map.end() &&
                    pool->slots[up_found->second].state == moe_cache_slot_state::valid) {
                    resident_up[index] = up_found->second;
                }
                if (gate_found != pool->map.end() &&
                    pool->slots[gate_found->second].state == moe_cache_slot_state::valid) {
                    resident_gate[index] = gate_found->second;
                }
                if (resident_up[index] >= 0 && resident_gate[index] >= 0) {
                    selected->pair_both++;
                } else if (resident_up[index] >= 0) {
                    selected->pair_up_only++;
                } else if (resident_gate[index] >= 0) {
                    selected->pair_gate_only++;
                } else {
                    selected->pair_neither++;
                }
            }

            for (int index = 0; index < n_ids; index++) {
                if (resident_up[index] < 0 || resident_gate[index] < 0) {
                    continue;
                }
                moe_cache_slot & up_slot = pool->slots[resident_up[index]];
                moe_cache_slot & gate_slot = pool->slots[resident_gate[index]];
                up_slot.readers++;
                gate_slot.readers++;
                moe_cache_lru_remove(*pool, resident_up[index]);
                moe_cache_lru_push_back(*pool, resident_up[index]);
                moe_cache_lru_remove(*pool, resident_gate[index]);
                moe_cache_lru_push_back(*pool, resident_gate[index]);
                node->pins[node->n_pins++] = {pool, resident_up[index]};
                node->pins[node->n_pins++] = {pool, resident_gate[index]};
                up_slots[hits] = resident_up[index];
                gate_slots[hits] = resident_gate[index];
                hit_acts[hits] = act_rows[index];
                hit_indices[hits] = index;
                mask |= UINT64_C(1) << index;
                selected->hits += 2;
                hits++;
            }

            for (int index = 0; index < n_ids && hits > 0; index++) {
                if (mask & (UINT64_C(1) << index)) {
                    continue;
                }
                const int32_t expert = ids[index];
                if (expert < 0 || expert >= up->n_expert) {
                    selected->misses += 2;
                    continue;
                }

                const int up_slot = moe_cache_lookup_or_queue_locked(
                        *session, *selected, *pool, pool_index,
                        up->data, expert, up->expert_size,
                        inserts_left, wake_worker);
                if (up_slot >= 0) {
                    moe_cache_slot & slot = pool->slots[up_slot];
                    slot.readers++;
                    moe_cache_lru_remove(*pool, up_slot);
                    moe_cache_lru_push_back(*pool, up_slot);
                    selected->hits++;
                } else {
                    selected->misses++;
                }

                const int gate_slot = moe_cache_lookup_or_queue_locked(
                        *session, *selected, *pool, pool_index,
                        gate->data, expert, gate->expert_size,
                        inserts_left, wake_worker);
                if (gate_slot >= 0) {
                    moe_cache_slot & slot = pool->slots[gate_slot];
                    slot.readers++;
                    moe_cache_lru_remove(*pool, gate_slot);
                    moe_cache_lru_push_back(*pool, gate_slot);
                    selected->hits++;
                } else {
                    selected->misses++;
                }

                if (up_slot >= 0) {
                    pool->slots[up_slot].readers--;
                }
                if (gate_slot >= 0) {
                    pool->slots[gate_slot].readers--;
                }

                if (up_slot < 0 && gate_slot < 0 &&
                    n_fused_misses < moe_cache_node_rows_max) {
                    fused_miss_indices[n_fused_misses++] = index;
                }
            }

            if (session->config.gpu_miss_percent > 0 &&
                n_fused_misses > 0 &&
                pool->n_slots <= INT_MAX - up->n_expert) {
                int gpu_rows =
                    (n_fused_misses * session->config.gpu_miss_percent + 99) /
                    100;
                gpu_rows = std::min(gpu_rows, n_fused_misses);

                for (int miss = 0; miss < gpu_rows; miss++) {
                    const int index = fused_miss_indices[miss];
                    const int32_t expert = ids[index];
                    if (expert < 0 || expert >= up->n_expert) {
                        continue;
                    }

                    up_slots[hits] = pool->n_slots + expert;
                    gate_slots[hits] = pool->n_slots + expert;
                    hit_acts[hits] = act_rows[index];
                    hit_indices[hits] = index;
                    mask |= UINT64_C(1) << index;
                    node->gpu_miss_rows++;
                    selected->gpu_miss_rows++;
                    selected->gpu_miss_fused_rows++;
                    selected->gpu_miss_bytes +=
                        2 * (unsigned long long)up->expert_size;
                    hits++;
                }
            }

            if (hits > 1 && node->gpu_miss_rows > 0) {
                for (int i = 1; i < hits; i++) {
                    const int key_index = hit_indices[i];
                    const int32_t key_up = up_slots[i];
                    const int32_t key_gate = gate_slots[i];
                    const float * key_act = hit_acts[i];

                    int j = i - 1;
                    while (j >= 0 && hit_indices[j] > key_index) {
                        hit_indices[j + 1] = hit_indices[j];
                        up_slots[j + 1] = up_slots[j];
                        gate_slots[j + 1] = gate_slots[j];
                        hit_acts[j + 1] = hit_acts[j];
                        j--;
                    }

                    hit_indices[j + 1] = key_index;
                    up_slots[j + 1] = key_up;
                    gate_slots[j + 1] = key_gate;
                    hit_acts[j + 1] = key_act;
                }
            }

            if (hits > 0) {
                selected->nodes++;
                selected->fused_attempts++;
            }
        }
    }
    if (wake_worker) {
        session->cv.notify_all();
    }

    if (hits == 0 || !moe_cache_dispatch_internal(
            node.get(), up->type, up->n_in, up->n_out, hits,
            up_slots, hit_acts, pool, gate_slots, glu_op,
            up_min, up_max, gate_min, gate_max)) {
        moe_cache_end(node.release());
        return nullptr;
    }

    {
        std::lock_guard<std::mutex> lock(session->mu);
        selected->fused_rows += hits;
        selected->fused_candidates += n_ids;
        selected->fused_nodes++;
    }

    *hit_mask = mask;
    return node.release();
}

static void moe_cache_invalidate_session(
        moe_cache_session & session, const void * base, size_t size) {
    std::unique_lock<std::mutex> lock(session.mu);
    session.tensor_devices.erase(base);
    for (auto & device_ptr : session.devices) {
        moe_cache_cancel_queue_locked(*device_ptr, base, size, false);
    }

    session.idle_cv.wait(lock, [&] {
        for (const auto & active : session.active_sources) {
            if (active.second.references > 0 &&
                moe_cache_ranges_overlap(active.first, active.second.bytes, base, size)) {
                return false;
            }
        }
        for (const auto & device_ptr : session.devices) {
            if (device_ptr->inflight &&
                moe_cache_ranges_overlap(
                    device_ptr->inflight_source, device_ptr->inflight_bytes, base, size)) {
                return false;
            }
        }
        return true;
    });

    for (auto & device_ptr : session.devices) {
        moe_cache_cancel_queue_locked(*device_ptr, base, size, false);
        moe_cache_device & device = *device_ptr;
        for (auto & pool_ptr : device.pools) {
            moe_cache_pool & pool = *pool_ptr;
            for (int index = 0; index < pool.n_slots; index++) {
                moe_cache_slot & slot = pool.slots[index];
                const void * source = slot.key.expert >= 0
                    ? (const char *)slot.key.tensor +
                        (size_t)slot.key.expert * pool.expert_size
                    : nullptr;
                if (slot.state != moe_cache_slot_state::free &&
                    moe_cache_ranges_overlap(source, pool.expert_size, base, size)) {
                    moe_cache_slot_reset(pool, index, true);
                }
            }
        }
        for (auto it = device.scheduler_gpu_ids_source_cache.begin();
             it != device.scheduler_gpu_ids_source_cache.end();) {
            if (moe_cache_ranges_overlap(
                    it->first.host_base,
                    (size_t)it->first.n_expert * it->first.expert_size,
                    base, size)) {
                it = device.scheduler_gpu_ids_source_cache.erase(it);
            } else {
                ++it;
            }
        }
        for (auto it = device.seen_tensors.begin(); it != device.seen_tensors.end();) {
            if (moe_cache_ranges_overlap(it->first, it->second.bytes, base, size)) {
                session.tensor_devices.erase(it->first);
                for (moe_cache_shape & shape : device.shapes) {
                    if (shape.expert_size == it->second.expert_size &&
                        shape.wtype == it->second.wtype) {
                        const uint64_t entries = (uint64_t)it->second.n_expert;
                        shape.n_entries = entries <= shape.n_entries
                            ? shape.n_entries - entries : 0;
                        shape.n_tensors = std::max<int64_t>(shape.n_tensors - 1, 0);
                        if (shape.pool >= 0 && shape.pool < (int)device.pools.size()) {
                            device.pools[shape.pool]->covers_all_entries =
                                (uint64_t)device.pools[shape.pool]->n_slots >= shape.n_entries;
                        }
                        if (shape.n_tensors == 0 && shape.pool < 0) {
                            shape.finished = false;
                        }
                        break;
                    }
                }
                it = device.seen_tensors.erase(it);
                device.visits_since_new_tensor = 0;
            } else {
                ++it;
            }
        }
        for (auto it = device.demand_count.begin(); it != device.demand_count.end();) {
            const void * source = it->first.expert >= 0
                ? (const char *)it->first.tensor +
                    (size_t)it->first.expert * it->second.expert_size
                : nullptr;
            if (moe_cache_ranges_overlap(
                    source, it->second.expert_size, base, size)) {
                it = device.demand_count.erase(it);
            } else {
                ++it;
            }
        }
    }
}

static void moe_cache_invalidate(const void * base, size_t size) {
    if (!base || size == 0 ||
        g_session_count.load(std::memory_order_acquire) == 0) {
        return;
    }
    std::lock_guard<std::mutex> registry_lock(g_registry_mu);
    for (moe_cache_session * session : g_sessions) {
        moe_cache_invalidate_session(*session, base, size);
    }
}

static size_t moe_cache_trim_session(
        moe_cache_session & session, int physical_device) {
    moe_cache_device * selected = nullptr;
    for (auto & device_ptr : session.devices) {
        if (device_ptr->physical == physical_device) {
            selected = device_ptr.get();
            break;
        }
    }
    if (!selected) {
        return 0;
    }

    std::unique_lock<std::mutex> dispatch_lock(selected->dispatch_mu);
    if (!moe_cache_scheduler_reap_locked(session, *selected, true)) {
        return 0;
    }
    std::unique_lock<std::mutex> lock(session.mu);
    selected->dead.store(true);
    moe_cache_cancel_queue_locked(*selected, nullptr, 0, true);
    session.cv.notify_all();
    session.idle_cv.wait(lock, [&] {
        return !selected->inflight;
    });

    size_t freed = 0;
    for (const auto & pool_ptr : selected->pools) {
        if (pool_ptr->slab) {
            freed += (size_t)pool_ptr->n_slots * pool_ptr->expert_size;
        }
    }
    freed += selected->d_out_cap + selected->d_input_cap +
             selected->act_q8_cap;
    lock.unlock();
    moe_cache_free_device(*selected);
    moe_cache_budget_unregister(*selected);

    if (freed > 0) {
        MOE_CACHE_LOG("[moe-cache] CUDA%d trimmed %zu MiB after a cache failure or allocator pressure\n",
                physical_device, freed >> 20);
    }
    return freed;
}

extern "C" size_t ggml_moe_cache_trim(int device) {
    if (g_session_count.load(std::memory_order_acquire) == 0) {
        return 0;
    }
    const ggml_cuda_device_info & info = ggml_cuda_info();
    if (device < 0 || device >= info.device_count) {
        return 0;
    }
    const int physical_device = info.devices[device].physical_device;
    size_t freed = 0;
    std::lock_guard<std::mutex> registry_lock(g_registry_mu);
    for (moe_cache_session * session : g_sessions) {
        freed += moe_cache_trim_session(*session, physical_device);
    }
    return freed;
}

void ggml_moe_cache_register(const void * owner) {
    if (ggml_moe_cache.owner && ggml_moe_cache.owner != owner) {
        return;
    }
    ggml_moe_cache.owner = owner;
    ggml_moe_cache.query_config = moe_cache_query_config;
    ggml_moe_cache.query_device = moe_cache_query_device;
    ggml_moe_cache.query_shape = moe_cache_query_shape;
    ggml_moe_cache.session_create = moe_cache_session_create;
    ggml_moe_cache.session_destroy = moe_cache_session_destroy;
    ggml_moe_cache.session_enter = moe_cache_session_enter;
    ggml_moe_cache.session_leave = moe_cache_session_leave;
    ggml_moe_cache.begin = moe_cache_begin;
    ggml_moe_cache.plan = moe_cache_plan;
    ggml_moe_cache.dispatch = moe_cache_dispatch;
    ggml_moe_cache.collect = moe_cache_collect;
    ggml_moe_cache.end = moe_cache_end;
    ggml_moe_cache.fused_begin = moe_cache_fused_begin;
    ggml_moe_cache.invalidate = moe_cache_invalidate;
}

#endif

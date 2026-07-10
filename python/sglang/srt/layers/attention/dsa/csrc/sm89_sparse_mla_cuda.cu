#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <mma.h>
#include <torch/extension.h>

#include <cmath>
#include <cstdint>
#include <limits>

namespace {

constexpr int kDimNope = 512;
constexpr int kDimRope = 64;
constexpr int kPackedWidth = 656;
constexpr int kGroupSize = 128;
constexpr int kThreads = 256;

__device__ __forceinline__ float load_fp8_scaled(
    const uint8_t* __restrict__ kv,
    int64_t token,
    int64_t dim,
    int64_t kv_stride0) {
  const auto* fp8_ptr = reinterpret_cast<const __nv_fp8_e4m3*>(kv + token * kv_stride0 + dim);
  const int64_t group_id = dim / kGroupSize;
  const auto* scale_ptr = reinterpret_cast<const float*>(kv + token * kv_stride0 + kDimNope + group_id * 4);
  return static_cast<float>(*fp8_ptr) * (*scale_ptr);
}

__device__ __forceinline__ float load_rope_bf16(
    const uint8_t* __restrict__ kv,
    int64_t token,
    int64_t dim,
    int64_t kv_stride0) {
  const auto* rope_ptr = reinterpret_cast<const __nv_bfloat16*>(kv + token * kv_stride0 + kDimNope + 16);
  return __bfloat162float(rope_ptr[dim]);
}

__device__ __forceinline__ float block_reduce_sum(float value, float* shared) {
  const int tid = threadIdx.x;
  shared[tid] = value;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }
  return shared[0];
}

__device__ __forceinline__ float block_reduce_max(float value, float* shared) {
  const int tid = threadIdx.x;
  shared[tid] = value;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
    }
    __syncthreads();
  }
  return shared[0];
}

__global__ void sm89_sparse_mla_prefill_kernel(
    const __nv_bfloat16* __restrict__ q_nope,
    const __nv_bfloat16* __restrict__ q_rope,
    const uint8_t* __restrict__ kv_cache,
    const int32_t* __restrict__ page_table,
    const int32_t* __restrict__ cache_seqlens,
    __nv_bfloat16* __restrict__ out,
    int64_t q_nope_stride0,
    int64_t q_nope_stride1,
    int64_t q_nope_stride2,
    int64_t q_rope_stride0,
    int64_t q_rope_stride1,
    int64_t q_rope_stride2,
    int64_t kv_stride0,
    int64_t page_table_stride0,
    int64_t page_table_stride1,
    int64_t out_stride0,
    int64_t out_stride1,
    int64_t out_stride2,
    int32_t topk,
    float sm_scale,
    float logit_cap) {
  const int row = blockIdx.x;
  const int head = blockIdx.y;
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane_id = tid & 31;

  __shared__ float tile_probs[8];
  __shared__ int32_t tile_tokens[8];
  __shared__ float softmax_state[4];  // m, l, alpha, unused padding

  const int32_t raw_row_len = cache_seqlens[row];
  const int32_t row_len = max(0, min(raw_row_len, topk));

  float acc0 = 0.0f;
  float acc1 = 0.0f;
  const int out_d0 = tid;
  const int out_d1 = tid + blockDim.x;

  if (tid == 0) {
    softmax_state[0] = -INFINITY;
    softmax_state[1] = 0.0f;
    softmax_state[2] = 1.0f;
  }
  __syncthreads();

  for (int n_start = 0; n_start < topk; n_start += 8) {
    const int n = n_start + warp_id;
    int32_t token = -1;
    bool valid = false;
    float score = -INFINITY;

    if (n < topk) {
      token = page_table[row * page_table_stride0 + n * page_table_stride1];
      valid = n < row_len && token >= 0;
    }

    if (valid) {
      float partial = 0.0f;
      for (int d = lane_id; d < kDimNope; d += 32) {
        const float q = __bfloat162float(
            q_nope[row * q_nope_stride0 + head * q_nope_stride1 + d * q_nope_stride2]);
        const float k = load_fp8_scaled(kv_cache, token, d, kv_stride0);
        partial += q * k;
      }
      for (int d = lane_id; d < kDimRope; d += 32) {
        const float q = __bfloat162float(
            q_rope[row * q_rope_stride0 + head * q_rope_stride1 + d * q_rope_stride2]);
        const float k = load_rope_bf16(kv_cache, token, d, kv_stride0);
        partial += q * k;
      }

#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        partial += __shfl_down_sync(0xffffffff, partial, offset);
      }
      if (lane_id == 0) {
        score = partial * sm_scale;
        if (logit_cap > 0.0f) {
          score = logit_cap * tanhf(score / logit_cap);
        }
      }
    }

    if (lane_id == 0) {
      tile_probs[warp_id] = score;
      tile_tokens[warp_id] = token;
    }
    __syncthreads();

    if (tid == 0) {
      float tile_max = -INFINITY;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        tile_max = fmaxf(tile_max, tile_probs[i]);
      }

      const float old_m = softmax_state[0];
      const float old_l = softmax_state[1];
      float alpha = 1.0f;
      float new_m = old_m;
      float new_l = old_l;
      if (isfinite(tile_max)) {
        new_m = fmaxf(old_m, tile_max);
        alpha = isfinite(old_m) ? __expf(old_m - new_m) : 0.0f;
        float tile_l = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const float p = isfinite(tile_probs[i]) ? __expf(tile_probs[i] - new_m) : 0.0f;
          tile_probs[i] = p;
          tile_l += p;
        }
        new_l = old_l * alpha + tile_l;
      } else {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          tile_probs[i] = 0.0f;
        }
      }
      softmax_state[0] = new_m;
      softmax_state[1] = new_l;
      softmax_state[2] = alpha;
    }
    __syncthreads();

    const float alpha = softmax_state[2];
    acc0 *= alpha;
    acc1 *= alpha;

#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float p = tile_probs[i];
      const int32_t v_token = tile_tokens[i];
      if (p != 0.0f) {
        if (out_d0 < kDimNope) {
          acc0 += p * load_fp8_scaled(kv_cache, v_token, out_d0, kv_stride0);
        }
        if (out_d1 < kDimNope) {
          acc1 += p * load_fp8_scaled(kv_cache, v_token, out_d1, kv_stride0);
        }
      }
    }
    __syncthreads();
  }

  const float denom = softmax_state[1];
  if (out_d0 < kDimNope) {
    const float value = denom > 0.0f ? acc0 / denom : 0.0f;
    out[row * out_stride0 + head * out_stride1 + out_d0 * out_stride2] = __float2bfloat16(value);
  }
  if (out_d1 < kDimNope) {
    const float value = denom > 0.0f ? acc1 / denom : 0.0f;
    out[row * out_stride0 + head * out_stride1 + out_d1 * out_stride2] = __float2bfloat16(value);
  }
}

constexpr int kHeadTile = 16;
constexpr int kTokenTile = 16;
constexpr int kWarpsPerBlock = 4;
constexpr int kTokensPerQkBlock = kWarpsPerBlock * kTokenTile;
constexpr int kDimsPerPvBlock = kWarpsPerBlock * kTokenTile;

__device__ __forceinline__ float warp_reduce_max(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
  }
  return value;
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

__global__ void sm89_sparse_mla_qk_tensorcore_kernel(
    const __nv_bfloat16* __restrict__ q_nope,
    const __nv_bfloat16* __restrict__ q_rope,
    const uint8_t* __restrict__ kv_cache,
    const int32_t* __restrict__ page_table,
    const int32_t* __restrict__ cache_seqlens,
    float* __restrict__ logits,
    int64_t q_nope_stride0,
    int64_t q_nope_stride1,
    int64_t q_nope_stride2,
    int64_t q_rope_stride0,
    int64_t q_rope_stride1,
    int64_t q_rope_stride2,
    int64_t kv_stride0,
    int64_t page_table_stride0,
    int64_t page_table_stride1,
    int32_t kv_tokens,
    int32_t num_heads,
    int32_t topk,
    float sm_scale,
    float logit_cap) {
  namespace wmma = nvcuda::wmma;
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int row = blockIdx.z;
  const int head_start = blockIdx.y * kHeadTile;
  const int token_start = blockIdx.x * kTokensPerQkBlock;
  const int32_t row_len = max(0, min(cache_seqlens[row], topk));

  __shared__ __align__(16) __nv_bfloat16 q_tile[kHeadTile * kTokenTile];
  __shared__ __align__(16) __nv_bfloat16 k_tiles[kWarpsPerBlock][kTokenTile * kTokenTile];
  __shared__ __align__(16) float logits_tiles[kWarpsPerBlock][kHeadTile * kTokenTile];
  __shared__ int32_t token_ids[kTokensPerQkBlock];

  if (tid < kTokensPerQkBlock) {
    const int n = token_start + tid;
    int32_t token = n < topk
                        ? page_table[row * page_table_stride0 + n * page_table_stride1]
                        : -1;
    if (!(n < row_len && token >= 0 && token < kv_tokens)) {
      token = -1;
    }
    token_ids[tid] = token;
  }
  __syncthreads();

  wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> q_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> k_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;
  wmma::fill_fragment(acc_frag, 0.0f);

  for (int k_start = 0; k_start < kDimNope + kDimRope; k_start += kTokenTile) {
    for (int index = tid; index < kHeadTile * kTokenTile; index += blockDim.x) {
      const int head_local = index / kTokenTile;
      const int k_local = index % kTokenTile;
      const int head = head_start + head_local;
      const int dim = k_start + k_local;
      __nv_bfloat16 value = __float2bfloat16(0.0f);
      if (head < num_heads) {
        if (dim < kDimNope) {
          value = q_nope[row * q_nope_stride0 + head * q_nope_stride1 + dim * q_nope_stride2];
        } else {
          const int rope_dim = dim - kDimNope;
          value = q_rope[row * q_rope_stride0 + head * q_rope_stride1 + rope_dim * q_rope_stride2];
        }
      }
      q_tile[index] = value;
    }

    for (int index = tid; index < kWarpsPerBlock * kTokenTile * kTokenTile; index += blockDim.x) {
      const int target_warp = index / (kTokenTile * kTokenTile);
      const int tile_index = index % (kTokenTile * kTokenTile);
      const int token_local = tile_index / kTokenTile;
      const int k_local = tile_index % kTokenTile;
      const int token = token_ids[target_warp * kTokenTile + token_local];
      const int dim = k_start + k_local;
      __nv_bfloat16 value = __float2bfloat16(0.0f);
      if (token >= 0) {
        value = dim < kDimNope
                    ? __float2bfloat16(load_fp8_scaled(kv_cache, token, dim, kv_stride0))
                    : *reinterpret_cast<const __nv_bfloat16*>(
                          kv_cache + token * kv_stride0 + kDimNope + 16 +
                          (dim - kDimNope) * static_cast<int>(sizeof(__nv_bfloat16)));
      }
      // WMMA matrix B is column-major: [K, token] at token * 16 + K.
      k_tiles[target_warp][token_local * kTokenTile + k_local] = value;
    }
    __syncthreads();

    wmma::load_matrix_sync(q_frag, q_tile, kTokenTile);
    wmma::load_matrix_sync(k_frag, k_tiles[warp_id], kTokenTile);
    wmma::mma_sync(acc_frag, q_frag, k_frag, acc_frag);
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < acc_frag.num_elements; ++i) {
    float value = acc_frag.x[i] * sm_scale;
    if (logit_cap > 0.0f) {
      value = logit_cap * tanhf(value / logit_cap);
    }
    acc_frag.x[i] = value;
  }
  wmma::store_matrix_sync(logits_tiles[warp_id], acc_frag, kTokenTile, wmma::mem_row_major);
  __syncthreads();

  for (int index = tid; index < kWarpsPerBlock * kHeadTile * kTokenTile; index += blockDim.x) {
    const int source_warp = index / (kHeadTile * kTokenTile);
    const int tile_index = index % (kHeadTile * kTokenTile);
    const int head_local = tile_index / kTokenTile;
    const int token_local = tile_index % kTokenTile;
    const int head = head_start + head_local;
    const int n = token_start + source_warp * kTokenTile + token_local;
    if (head < num_heads && n < topk) {
      const int token = token_ids[source_warp * kTokenTile + token_local];
      logits[(static_cast<int64_t>(row) * num_heads + head) * topk + n] =
          n < row_len && token >= 0 ? logits_tiles[source_warp][tile_index] : -INFINITY;
    }
  }
}

__global__ void sm89_sparse_mla_lse_kernel(
    const float* __restrict__ logits,
    float* __restrict__ lse,
    int32_t num_heads,
    int32_t topk) {
  const int row = blockIdx.x;
  const int head_start = blockIdx.y * kHeadTile;
  const int warp_id = threadIdx.x >> 5;
  const int lane_id = threadIdx.x & 31;

#pragma unroll
  for (int pass = 0; pass < 2; ++pass) {
    const int head = head_start + warp_id + pass * 8;
    if (head >= num_heads) {
      continue;
    }
    const float* row_logits = logits + (static_cast<int64_t>(row) * num_heads + head) * topk;
    float local_max = -INFINITY;
    for (int n = lane_id; n < topk; n += 32) {
      local_max = fmaxf(local_max, row_logits[n]);
    }
    const float row_max = __shfl_sync(0xffffffff, warp_reduce_max(local_max), 0);
    float local_sum = 0.0f;
    if (isfinite(row_max)) {
      for (int n = lane_id; n < topk; n += 32) {
        local_sum += __expf(row_logits[n] - row_max);
      }
    }
    const float row_sum = warp_reduce_sum(local_sum);
    if (lane_id == 0) {
      lse[static_cast<int64_t>(row) * num_heads + head] =
          row_sum > 0.0f ? row_max + __logf(row_sum) : -INFINITY;
    }
  }
}

__global__ void sm89_sparse_mla_pv_tensorcore_kernel(
    const float* __restrict__ logits,
    const float* __restrict__ lse,
    const uint8_t* __restrict__ kv_cache,
    const int32_t* __restrict__ page_table,
    const int32_t* __restrict__ cache_seqlens,
    __nv_bfloat16* __restrict__ out,
    int64_t kv_stride0,
    int64_t page_table_stride0,
    int64_t page_table_stride1,
    int64_t out_stride0,
    int64_t out_stride1,
    int64_t out_stride2,
    int32_t kv_tokens,
    int32_t num_heads,
    int32_t topk) {
  namespace wmma = nvcuda::wmma;
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int row = blockIdx.z;
  const int head_start = blockIdx.y * kHeadTile;
  const int dim_start = blockIdx.x * kDimsPerPvBlock;
  const int32_t row_len = max(0, min(cache_seqlens[row], topk));

  __shared__ __align__(16) __nv_bfloat16 prob_tile[kHeadTile * kTokenTile];
  __shared__ __align__(16) __nv_bfloat16 value_tiles[kWarpsPerBlock][kTokenTile * kTokenTile];
  __shared__ __align__(16) float out_tiles[kWarpsPerBlock][kHeadTile * kTokenTile];
  __shared__ int32_t token_ids[kTokenTile];

  wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> prob_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> value_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;
  wmma::fill_fragment(acc_frag, 0.0f);

  for (int token_start = 0; token_start < topk; token_start += kTokenTile) {
    if (tid < kTokenTile) {
      const int n = token_start + tid;
      int32_t token = n < topk
                          ? page_table[row * page_table_stride0 + n * page_table_stride1]
                          : -1;
      if (!(n < row_len && token >= 0 && token < kv_tokens)) {
        token = -1;
      }
      token_ids[tid] = token;
    }
    __syncthreads();

    for (int index = tid; index < kHeadTile * kTokenTile; index += blockDim.x) {
      const int head_local = index / kTokenTile;
      const int token_local = index % kTokenTile;
      const int head = head_start + head_local;
      const int n = token_start + token_local;
      float probability = 0.0f;
      if (head < num_heads && n < row_len && token_ids[token_local] >= 0) {
        const float row_lse = lse[static_cast<int64_t>(row) * num_heads + head];
        if (isfinite(row_lse)) {
          const float logit = logits[(static_cast<int64_t>(row) * num_heads + head) * topk + n];
          probability = __expf(logit - row_lse);
        }
      }
      prob_tile[index] = __float2bfloat16(probability);
    }

    for (int index = tid; index < kWarpsPerBlock * kTokenTile * kTokenTile; index += blockDim.x) {
      const int target_warp = index / (kTokenTile * kTokenTile);
      const int tile_index = index % (kTokenTile * kTokenTile);
      const int token_local = tile_index / kTokenTile;
      const int dim_local = tile_index % kTokenTile;
      const int token = token_ids[token_local];
      const int dim = dim_start + target_warp * kTokenTile + dim_local;
      const float value = token >= 0 && dim < kDimNope
                              ? load_fp8_scaled(kv_cache, token, dim, kv_stride0)
                              : 0.0f;
      value_tiles[target_warp][tile_index] = __float2bfloat16(value);
    }
    __syncthreads();

    wmma::load_matrix_sync(prob_frag, prob_tile, kTokenTile);
    wmma::load_matrix_sync(value_frag, value_tiles[warp_id], kTokenTile);
    wmma::mma_sync(acc_frag, prob_frag, value_frag, acc_frag);
    __syncthreads();
  }

  wmma::store_matrix_sync(out_tiles[warp_id], acc_frag, kTokenTile, wmma::mem_row_major);
  __syncthreads();
  for (int index = tid; index < kWarpsPerBlock * kHeadTile * kTokenTile; index += blockDim.x) {
    const int source_warp = index / (kHeadTile * kTokenTile);
    const int tile_index = index % (kHeadTile * kTokenTile);
    const int head_local = tile_index / kTokenTile;
    const int dim_local = tile_index % kTokenTile;
    const int head = head_start + head_local;
    const int dim = dim_start + source_warp * kTokenTile + dim_local;
    if (head < num_heads && dim < kDimNope) {
      out[row * out_stride0 + head * out_stride1 + dim * out_stride2] =
          __float2bfloat16(out_tiles[source_warp][tile_index]);
    }
  }
}

}  // namespace

torch::Tensor sm89_sparse_mla_prefill_cuda(
    const torch::Tensor& q_nope,
    const torch::Tensor& q_rope,
    const torch::Tensor& kv_cache,
    const torch::Tensor& page_table,
    const torch::Tensor& cache_seqlens,
    double sm_scale,
    double logit_cap,
    int64_t v_head_dim) {
  TORCH_CHECK(q_nope.is_cuda(), "q_nope must be CUDA");
  TORCH_CHECK(q_rope.is_cuda(), "q_rope must be CUDA");
  TORCH_CHECK(kv_cache.is_cuda(), "kv_cache must be CUDA");
  TORCH_CHECK(page_table.is_cuda(), "page_table must be CUDA");
  TORCH_CHECK(cache_seqlens.is_cuda(), "cache_seqlens must be CUDA");
  TORCH_CHECK(q_nope.scalar_type() == at::kBFloat16, "q_nope must be bfloat16");
  TORCH_CHECK(q_rope.scalar_type() == at::kBFloat16, "q_rope must be bfloat16");
  TORCH_CHECK(kv_cache.scalar_type() == at::kFloat8_e4m3fn, "kv_cache must be float8_e4m3fn");
  TORCH_CHECK(page_table.scalar_type() == at::kInt, "page_table must be int32");
  TORCH_CHECK(cache_seqlens.scalar_type() == at::kInt, "cache_seqlens must be int32");
  TORCH_CHECK(q_nope.dim() == 3, "q_nope must have shape [total_q, heads, 512]");
  TORCH_CHECK(q_rope.dim() == 3, "q_rope must have shape [total_q, heads, 64]");
  TORCH_CHECK(kv_cache.dim() == 3, "kv_cache must have shape [tokens, 1, 656]");
  TORCH_CHECK(page_table.dim() == 2, "page_table must have shape [total_q, topk]");
  TORCH_CHECK(cache_seqlens.dim() == 1, "cache_seqlens must have shape [total_q]");
  TORCH_CHECK(v_head_dim == kDimNope, "v_head_dim must be 512");
  TORCH_CHECK(q_nope.size(2) == kDimNope, "q_nope dim must be 512");
  TORCH_CHECK(q_rope.size(2) == kDimRope, "q_rope dim must be 64");
  TORCH_CHECK(kv_cache.size(1) == 1, "kv_cache middle dim must be 1");
  TORCH_CHECK(kv_cache.size(2) == kPackedWidth, "kv_cache packed width must be 656");
  TORCH_CHECK(kv_cache.stride(2) == 1, "kv_cache last stride must be 1");
  TORCH_CHECK(page_table.size(0) == q_nope.size(0), "page_table rows must match q_nope");
  TORCH_CHECK(cache_seqlens.size(0) == q_nope.size(0), "cache_seqlens rows must match q_nope");
  TORCH_CHECK(q_rope.size(0) == q_nope.size(0), "q_rope rows must match q_nope");
  TORCH_CHECK(q_rope.size(1) == q_nope.size(1), "q_rope heads must match q_nope");
  TORCH_CHECK(page_table.size(1) <= 4096, "prototype CUDA backend supports topk <= 4096");

  auto out = torch::empty_like(q_nope);
  if (q_nope.numel() == 0) {
    return out;
  }

  const c10::cuda::OptionalCUDAGuard device_guard(q_nope.device());
  const auto stream = at::cuda::getCurrentCUDAStream(q_nope.get_device());
  const dim3 grid(q_nope.size(0), q_nope.size(1));
  const dim3 block(kThreads);

  sm89_sparse_mla_prefill_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q_nope.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(q_rope.data_ptr()),
      reinterpret_cast<const uint8_t*>(kv_cache.data_ptr()),
      reinterpret_cast<const int32_t*>(page_table.data_ptr()),
      reinterpret_cast<const int32_t*>(cache_seqlens.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
      q_nope.stride(0),
      q_nope.stride(1),
      q_nope.stride(2),
      q_rope.stride(0),
      q_rope.stride(1),
      q_rope.stride(2),
      kv_cache.stride(0),
      page_table.stride(0),
      page_table.stride(1),
      out.stride(0),
      out.stride(1),
      out.stride(2),
      static_cast<int32_t>(page_table.size(1)),
      static_cast<float>(sm_scale),
      static_cast<float>(logit_cap));

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor sm89_sparse_mla_prefill_cuda_tensorcore(
    const torch::Tensor& q_nope,
    const torch::Tensor& q_rope,
    const torch::Tensor& kv_cache,
    const torch::Tensor& page_table,
    const torch::Tensor& cache_seqlens,
    double sm_scale,
    double logit_cap,
    int64_t v_head_dim) {
  TORCH_CHECK(q_nope.is_cuda() && q_rope.is_cuda() && kv_cache.is_cuda(),
              "q_nope, q_rope, and kv_cache must be CUDA tensors");
  TORCH_CHECK(page_table.is_cuda() && cache_seqlens.is_cuda(),
              "page_table and cache_seqlens must be CUDA tensors");
  TORCH_CHECK(q_nope.scalar_type() == at::kBFloat16, "q_nope must be bfloat16");
  TORCH_CHECK(q_rope.scalar_type() == at::kBFloat16, "q_rope must be bfloat16");
  TORCH_CHECK(kv_cache.scalar_type() == at::kFloat8_e4m3fn, "kv_cache must be float8_e4m3fn");
  TORCH_CHECK(page_table.scalar_type() == at::kInt, "page_table must be int32");
  TORCH_CHECK(cache_seqlens.scalar_type() == at::kInt, "cache_seqlens must be int32");
  TORCH_CHECK(q_nope.dim() == 3 && q_nope.size(2) == kDimNope,
              "q_nope must have shape [total_q, heads, 512]");
  TORCH_CHECK(q_rope.dim() == 3 && q_rope.size(2) == kDimRope,
              "q_rope must have shape [total_q, heads, 64]");
  TORCH_CHECK(kv_cache.dim() == 3 && kv_cache.size(1) == 1 && kv_cache.size(2) == kPackedWidth,
              "kv_cache must have shape [tokens, 1, 656]");
  TORCH_CHECK(page_table.dim() == 2 && page_table.size(0) == q_nope.size(0),
              "page_table must have shape [total_q, topk]");
  TORCH_CHECK(cache_seqlens.dim() == 1 && cache_seqlens.size(0) == q_nope.size(0),
              "cache_seqlens must have shape [total_q]");
  TORCH_CHECK(q_rope.size(0) == q_nope.size(0) && q_rope.size(1) == q_nope.size(1),
              "q_rope rows and heads must match q_nope");
  const auto device = q_nope.device();
  TORCH_CHECK(q_rope.device() == device && kv_cache.device() == device && page_table.device() == device &&
                  cache_seqlens.device() == device,
              "all Tensor-Core inputs must be on the same CUDA device");
  TORCH_CHECK(kv_cache.stride(2) == 1 && kv_cache.stride(0) >= kPackedWidth && kv_cache.stride(0) % 4 == 0,
              "kv_cache must have aligned packed rows");
  TORCH_CHECK(cache_seqlens.stride(0) == 1, "cache_seqlens must be contiguous");
  TORCH_CHECK(v_head_dim == kDimNope, "v_head_dim must be 512");
  TORCH_CHECK(page_table.size(1) > 0 && page_table.size(1) <= 4096,
              "Tensor-Core CUDA backend supports 0 < topk <= 4096");
  TORCH_CHECK(q_nope.size(0) <= 4096, "total_q <= 4096 is required to bound logits workspace");
  TORCH_CHECK(q_nope.size(1) <= 64, "Tensor-Core CUDA backend supports at most 64 query heads");

  auto out = torch::empty_like(q_nope);
  if (q_nope.numel() == 0) {
    return out;
  }
  auto logits = torch::empty(
      {q_nope.size(0), q_nope.size(1), page_table.size(1)},
      q_nope.options().dtype(torch::kFloat32));
  auto lse = torch::empty(
      {q_nope.size(0), q_nope.size(1)},
      q_nope.options().dtype(torch::kFloat32));

  const c10::cuda::OptionalCUDAGuard device_guard(q_nope.device());
  const auto stream = at::cuda::getCurrentCUDAStream(q_nope.get_device());
  const int32_t total_q = static_cast<int32_t>(q_nope.size(0));
  const int32_t num_heads = static_cast<int32_t>(q_nope.size(1));
  const int32_t topk = static_cast<int32_t>(page_table.size(1));
  const int32_t head_tiles = (num_heads + kHeadTile - 1) / kHeadTile;

  const dim3 qk_grid((topk + kTokensPerQkBlock - 1) / kTokensPerQkBlock, head_tiles, total_q);
  sm89_sparse_mla_qk_tensorcore_kernel<<<qk_grid, 128, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q_nope.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(q_rope.data_ptr()),
      reinterpret_cast<const uint8_t*>(kv_cache.data_ptr()),
      reinterpret_cast<const int32_t*>(page_table.data_ptr()),
      reinterpret_cast<const int32_t*>(cache_seqlens.data_ptr()),
      logits.data_ptr<float>(),
      q_nope.stride(0),
      q_nope.stride(1),
      q_nope.stride(2),
      q_rope.stride(0),
      q_rope.stride(1),
      q_rope.stride(2),
      kv_cache.stride(0),
      page_table.stride(0),
      page_table.stride(1),
      static_cast<int32_t>(kv_cache.size(0)),
      num_heads,
      topk,
      static_cast<float>(sm_scale),
      static_cast<float>(logit_cap));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const dim3 lse_grid(total_q, head_tiles);
  sm89_sparse_mla_lse_kernel<<<lse_grid, 256, 0, stream>>>(
      logits.data_ptr<float>(), lse.data_ptr<float>(), num_heads, topk);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const dim3 pv_grid((kDimNope + kDimsPerPvBlock - 1) / kDimsPerPvBlock, head_tiles, total_q);
  sm89_sparse_mla_pv_tensorcore_kernel<<<pv_grid, 128, 0, stream>>>(
      logits.data_ptr<float>(),
      lse.data_ptr<float>(),
      reinterpret_cast<const uint8_t*>(kv_cache.data_ptr()),
      reinterpret_cast<const int32_t*>(page_table.data_ptr()),
      reinterpret_cast<const int32_t*>(cache_seqlens.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
      kv_cache.stride(0),
      page_table.stride(0),
      page_table.stride(1),
      out.stride(0),
      out.stride(1),
      out.stride(2),
      static_cast<int32_t>(kv_cache.size(0)),
      num_heads,
      topk);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sm89_sparse_mla_prefill_cuda", &sm89_sparse_mla_prefill_cuda, "SM89 GLM DSA sparse MLA prefill CUDA prototype");
  m.def("sm89_sparse_mla_prefill_cuda_tensorcore", &sm89_sparse_mla_prefill_cuda_tensorcore,
        "SM89 GLM DSA cross-head BF16 Tensor-Core prefill prototype");
}

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#if defined(__CUDACC__) && defined(BFLOAT16)
#include <cuda_bf16.h>  // bfloat16 is float32 compatible with less mantissa bits
#endif

#define WARP_SIZE 32

template <typename T>
__device__ __forceinline__ T shfl_xor(T v, int i) {
#if __CUDA_ARCH__ >= 300
#if CUDA_VERSION >= 9000
  return __shfl_xor_sync(0xffffffff, v, i, WARP_SIZE);
#else
  return __shfl_xor(v, i, WARP_SIZE);
#endif
#else
  assert(false);
#endif
}

template <typename T, bool broadcast = true>
__device__ __forceinline__ void reduce_sum_wrap(T &val) {
// #if __CUDA_ARCH__ >= 800
//   val = __reduce_add_sync(0xffffffff, val); // only for unsigned; int
// #else
#pragma unroll
  for (int i = 16; i >= 1; i /= 2) val += shfl_xor<T>(val, i);
  if (broadcast) val = __shfl_sync(0xffffffff, val, 0);
  // #endif
}

// If you use two or more reudce_sum_block(), please be carefully! Add __syncthreads() between them.
template <typename T, bool broadcast = true>
__device__ void reduce_sum_block(T &val) {
  int tid     = threadIdx.x;
  int lane_id = tid % WARP_SIZE;
  volatile __shared__ T smem[WARP_SIZE];
#pragma unroll
  for (int i = 16; i >= 1; i /= 2) val += shfl_xor<T>(val, i);
  if (lane_id == 0) smem[tid / WARP_SIZE] = val;
  if (tid < WARP_SIZE && tid >= (blockDim.x / WARP_SIZE)) smem[tid] = 0;
  __syncthreads();
  if (tid < WARP_SIZE) {
    val = smem[tid];
#pragma unroll
    for (int i = 16; i >= 1; i /= 2) val += shfl_xor<T>(val, i);
  }
  if (broadcast) {
    if (tid == 0) smem[0] = val;
    __syncthreads();
    val = smem[0];
  }
}

template <typename T, bool broadcast = true>
__device__ __forceinline__ void reduce_max_wrap(T &val) {
#if __CUDA_ARCH__ >= 800
  if constexpr (std::is_same<T, unsigned>::value) {
    val = __reduce_max_sync(0xffffffff, val);
  } else if constexpr (std::is_same<T, int>::value) {
    val = __reduce_max_sync(0xffffffff, val);
    //   } else if constexpr (std::is_same<T, float>::value) {
    // val = __int_as_float(__reduce_max_sync(0xffffffff, __float_as_int(val))); // only for all value >= 0
  } else {
#pragma unroll
    for (int i = 16; i >= 1; i /= 2) val = max(shfl_xor<T>(val, i), val);
    if (broadcast) val = __shfl_sync(0xffffffff, val, 0);
  }
#else
#pragma unroll
  for (int i = 16; i >= 1; i /= 2) val = max(shfl_xor<T>(val, i), val);
  if (broadcast) val = __shfl_sync(0xffffffff, val, 0);
#endif
}

template <typename T, bool broadcast = true>
__device__ void reduce_max_block(T &val) {
  int tid     = threadIdx.x;
  int lane_id = tid % WARP_SIZE;
  volatile __shared__ T smem[WARP_SIZE];
#pragma unroll
  for (int i = WARP_SIZE / 2; i >= 1; i >>= 1) val = max(val, shfl_xor<T>(val, i));
  if (lane_id == 0) smem[tid / WARP_SIZE] = val;
  if (tid < WARP_SIZE && tid >= blockDim.x / WARP_SIZE) smem[tid] = val;  // avoid undefined data in smem
  __syncthreads();
  if (tid < WARP_SIZE) {
    val = smem[tid];
#pragma unroll
    for (int i = WARP_SIZE / 2; i >= 1; i >>= 1) val = max(val, shfl_xor<T>(val, i));
  }
  if (broadcast) {
    if (tid == 0) smem[0] = val;
    __syncthreads();
    val = smem[0];
  }
}

template <typename T, bool broadcast = true>
__device__ __forceinline__ void reduce_min_wrap(T &val) {
#if __CUDA_ARCH__ >= 800
  if constexpr (std::is_same<T, unsigned>::value) {
    val = __reduce_min_sync(0xffffffff, val);
  } else if constexpr (std::is_same<T, int>::value) {
    val = __reduce_min_sync(0xffffffff, val);
    //   } else if constexpr (std::is_same<T, float>::value) {
    // val = __int_as_float(__reduce_min_sync(0xffffffff, __float_as_int(val))); // only for all value >= 0
  } else {
#pragma unroll
    for (int i = 16; i >= 1; i /= 2) val = min(shfl_xor<T>(val, i), val);
    if (broadcast) val = __shfl_sync(0xffffffff, val, 0);
  }
#else
#pragma unroll
  for (int i = 16; i >= 1; i /= 2) val = min(shfl_xor<T>(val, i), val);
  if (broadcast) val = __shfl_sync(0xffffffff, val, 0);
#endif
}

template <typename T, bool broadcast = true>
__device__ void reduce_min_block(T &val) {
  int tid     = threadIdx.x;
  int lane_id = tid % WARP_SIZE;
  volatile __shared__ T smem[WARP_SIZE];
#pragma unroll
  for (int i = WARP_SIZE / 2; i >= 1; i >>= 1) val = min(val, shfl_xor<T>(val, i));
  if (lane_id == 0) smem[tid / WARP_SIZE] = val;
  if (tid < WARP_SIZE && tid >= blockDim.x / WARP_SIZE) smem[tid] = val;  // avoid undefined data in smem
  __syncthreads();
  if (tid < WARP_SIZE) {
    val = smem[tid];
#pragma unroll
    for (int i = WARP_SIZE / 2; i >= 1; i >>= 1) val = min(val, shfl_xor<T>(val, i));
  }
  if (broadcast) {
    if (tid == 0) smem[0] = val;
    __syncthreads();
    val = smem[0];
  }
}

/*
template <typename T>
__device__ __forceinline__ T reduce2(T val) {
  int tid = threadIdx.x;
  volatile __shared__ T smem[CUDA_NUM_THREADS - WARP_SIZE];
  if (tid >= WARP_SIZE) smem[tid - WARP_SIZE] = val;
  __syncthreads();
  if (tid < WARP_SIZE) {
    for (int k = tid; k < CUDA_NUM_THREADS - WARP_SIZE; k += WARP_SIZE) {
      val += smem[k];
    }
    smem[tid] = val;
  }
  if (tid < 16) smem[tid] += smem[tid + 16];
  if (tid < 8) smem[tid] += smem[tid + 8];
  if (tid < 4) smem[tid] += smem[tid + 4];
  if (tid < 2) smem[tid] += smem[tid + 2];
  if (tid < 1) smem[tid] += smem[tid + 1];
  __syncthreads();
  return smem[0];
}
*/
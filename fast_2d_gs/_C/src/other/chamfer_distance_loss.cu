#include "util.cuh"

template <typename T, bool sphere>
void __global__ chamfer_distance_forward_kernal(int B, int M, int N, int D, const T* __restrict__ x,
    const T* __restrict__ y, T* __restrict__ min_value, int32_t* __restrict__ min_index) {
  const int b   = blockIdx.y;
  const int m   = blockIdx.x;
  const int tid = threadIdx.x;
  const int lid = threadIdx.x % WARP_SIZE;
  const int wid = threadIdx.x / WARP_SIZE;

  constexpr T INF         = std::numeric_limits<T>::max();
  constexpr T EPS         = 1e-6;
  constexpr unsigned MASK = -1u;

  extern __shared__ T X[];

#if __CUDA_ARCH__ >= 800
  __shared__ int temp[WARP_SIZE];
#endif

  y = y + b * N * D;
  if (tid < D) X[tid] = x[(b * M + m) * D + tid];
  __syncthreads();
  T min_dist  = INF;
  int32_t idx = 0;
  for (int i = tid; i < N; i += blockDim.x) {
    T dist = 0;
    if constexpr (sphere) {
      T t  = sin(X[0]) * sin(y[i * 2 + 0]) * cos(X[1] - y[i * 2 + 1]) + cos(X[0]) * cos(y[i * 2 + 0]);
      t    = clamp(t, T(-1 + EPS), T(1) - EPS);
      dist = acos(t);
    } else {
      for (int d = 0; d < D; ++d) {
        T t  = X[d] - y[i * D + d];
        dist = dist + t * t;
      }
    }
    if (dist < min_dist) {
      min_dist = dist;
      idx      = i;
    }
  }
#if __CUDA_ARCH__ >= 800
  int dist   = __float_as_int(min_dist);
  auto value = __reduce_min_sync(MASK, dist);
  if (lid == 0) temp[wid] = value;
  if (tid < WARP_SIZE && tid >= blockDim.x / WARP_SIZE) temp[tid] = dist;  // avoid undefined data in smem
  __syncthreads();
  if (wid == 0) value = __reduce_min_sync(MASK, temp[lid]);
  if (tid == 0) temp[0] = value;
  __syncthreads();
  value = temp[0];
#else
  T value = min_dist, dist = min_dist;
  reduce_min_block<T, true>(value);
#endif
  if (dist == value) {
    min_value[b * M + m] = min_dist;
    min_index[b * M + m] = idx;
  }
}

template <typename T, bool sphere>
void __global__ chamfer_distance_backward_kernal(int B, int M, int N, int D, const T* __restrict__ x,
    const T* __restrict__ y, const int32_t* __restrict__ index1, const int32_t* __restrict__ index2,
    const T* __restrict__ g1, const T* __restrict__ g2, T* __restrict__ gx, T* __restrict__ gy) {
  const int idx   = blockIdx.x * blockDim.x + threadIdx.x;
  constexpr T EPS = 1e-6;
  if (idx < B * M) {
    int k = index1[idx];
    T g   = g1[idx];
    if constexpr (sphere) {
      T theta1 = x[idx * 2 + 0];
      T theta2 = y[k * 2 + 0];
      T phi1   = x[idx * 2 + 1];
      T phi2   = y[k * 2 + 1];
      T s1     = sin(theta1);
      T s2     = sin(theta2);
      T c1     = cos(theta1);
      T c2     = cos(theta2);
      T cp     = cos(phi1 - phi2);
      T sp     = sin(phi1 - phi2);
      T t      = s1 * s2 * cp + c1 * c2;
      t        = clamp(t, T(-1 + EPS), T(1 - EPS));
      g        = -rsqrt(T(1) - t * t) * g;

      atomicAdd(gx + idx * 2 + 0, g * (c1 * s2 * cp - s1 * c2));
      atomicAdd(gx + idx * 2 + 1, g * (s1 * s2 * -sp));
      atomicAdd(gy + k * 2 + 0, g * (s1 * c2 * cp - c1 * s2));
      atomicAdd(gy + k * 2 + 1, g * (s1 * s2 * sp));
    } else {
      for (int d = 0; d < D; ++d) {
        T t = T(2.) * (x[idx * D + d] - y[k * D + d]);
        atomicAdd(gx + idx * D + d, t * g);
        atomicAdd(gy + k * D + d, -t * g);
      }
    }
  }
  if (idx < B * N) {
    int k = index2[idx];
    T g   = g2[idx];
    if constexpr (sphere) {
      T theta1 = x[k * 2 + 0];
      T theta2 = y[idx * 2 + 0];
      T phi1   = x[k * 2 + 1];
      T phi2   = y[idx * 2 + 1];
      T s1     = sin(theta1);
      T s2     = sin(theta2);
      T c1     = cos(theta1);
      T c2     = cos(theta2);
      T cp     = cos(phi1 - phi2);
      T sp     = sin(phi1 - phi2);
      T t      = s1 * s2 * cp + c1 * c2;
      t        = clamp(t, T(-1 + EPS), T(1 - EPS));
      g        = -rsqrt(T(1) - t * t) * g;

      atomicAdd(gx + k * 2 + 0, g * (c1 * s2 * cp - s1 * c2));
      atomicAdd(gx + k * 2 + 1, g * (-s1 * s2 * sp));
      atomicAdd(gy + idx * 2 + 0, g * (s1 * c2 * cp - c1 * s2));
      atomicAdd(gy + idx * 2 + 1, g * (s1 * s2 * sp));
    } else {
      for (int d = 0; d < D; ++d) {
        T t = T(2.) * (x[k * D + d] - y[idx * D + d]);
        atomicAdd(gx + k * D + d, t * g);
        atomicAdd(gy + idx * D + d, -t * g);
      }
    }
  }
}

vector<Tensor> chamfer_distance_forward(Tensor p1, Tensor p2, bool sphere) {
  CHECK_INPUT(p1);
  CHECK_INPUT(p2);
  BCNN_ASSERT(p1.ndimension() == p2.ndimension() && p1.size(-1) == p2.size(-1), "Error shape");
  unsigned D = p1.size(-1);
  unsigned M = p1.size(-2);
  unsigned N = p2.size(-2);
  BCNN_ASSERT(p1.numel() / M == p2.numel() / N, "Error shape");
  if (sphere) BCNN_ASSERT(D == 2, "Shape must be [..., 2] when sphere=True");
  unsigned B = p1.numel() / M / D;
  auto size1 = p1.sizes().vec();
  auto size2 = p2.sizes().vec();
  size1.pop_back();
  size2.pop_back();
  Tensor index1 = torch::zeros(size1, p1.options().dtype(torch::kInt32));
  Tensor index2 = torch::zeros(size2, p1.options().dtype(torch::kInt32));
  Tensor dist1  = torch::zeros(size1, p1.options());
  Tensor dist2  = torch::zeros(size2, p1.options());

  //   AT_DISPATCH_FLOATING_TYPES(p1.scalar_type(), "chamfer_distance_forward_kernal", [&] {
  using scalar_t = float;
  switch (sphere) {
    case true:
      chamfer_distance_forward_kernal<scalar_t, true> KERNEL_ARG(dim3(M, B, 1), 1024, D * sizeof(scalar_t))(B, M, N, D,
          p1.data_ptr<scalar_t>(), p2.data_ptr<scalar_t>(), dist1.data_ptr<scalar_t>(), index1.data_ptr<int32_t>());
      chamfer_distance_forward_kernal<scalar_t, true> KERNEL_ARG(dim3(N, B, 1), 1024, D * sizeof(scalar_t))(B, N, M, D,
          p2.data_ptr<scalar_t>(), p1.data_ptr<scalar_t>(), dist2.data_ptr<scalar_t>(), index2.data_ptr<int32_t>());
      break;
    default:
      chamfer_distance_forward_kernal<scalar_t, false> KERNEL_ARG(dim3(M, B, 1), 1024, D * sizeof(scalar_t))(B, M, N, D,
          p1.data_ptr<scalar_t>(), p2.data_ptr<scalar_t>(), dist1.data_ptr<scalar_t>(), index1.data_ptr<int32_t>());
      chamfer_distance_forward_kernal<scalar_t, false> KERNEL_ARG(dim3(N, B, 1), 1024, D * sizeof(scalar_t))(B, N, M, D,
          p2.data_ptr<scalar_t>(), p1.data_ptr<scalar_t>(), dist2.data_ptr<scalar_t>(), index2.data_ptr<int32_t>());
      break;
  }
  //   cudaDeviceSynchronize();
  //   CHECK_CUDA_ERROR("chamfer_distance_forward_kernal");
  //   });
  return {index1, index2, dist1, dist2};
}

vector<Tensor> chamfer_distance_backward(
    Tensor p1, Tensor p2, bool sphere, Tensor index1, Tensor index2, Tensor grad1, Tensor grad2) {
  CHECK_INPUT(grad1);
  CHECK_INPUT(grad2);
  unsigned D = p1.size(-1);
  unsigned M = p1.size(-2);
  unsigned N = p2.size(-2);
  unsigned B = p1.numel() / M / D;
  int K      = B * max(M, N);

  Tensor g1 = torch::zeros_like(p1);
  Tensor g2 = torch::zeros_like(p2);
  //   AT_DISPATCH_FLOATING_TYPES(p1.scalar_type(), "chamfer_distance_backward_kernal", [&] {
  using scalar_t = float;
  switch (sphere) {
    case true:
      chamfer_distance_backward_kernal<scalar_t, true> KERNEL_ARG(div_round_up(K, 256), 256)(B, M, N, D,
          p1.data_ptr<scalar_t>(), p2.data_ptr<scalar_t>(), index1.data_ptr<int32_t>(), index2.data_ptr<int32_t>(),
          grad1.data_ptr<scalar_t>(), grad2.data_ptr<scalar_t>(), g1.data_ptr<scalar_t>(), g2.data<scalar_t>());
      break;
    default:
      chamfer_distance_backward_kernal<scalar_t, false> KERNEL_ARG(div_round_up(K, 256), 256)(B, M, N, D,
          p1.data_ptr<scalar_t>(), p2.data_ptr<scalar_t>(), index1.data_ptr<int32_t>(), index2.data_ptr<int32_t>(),
          grad1.data_ptr<scalar_t>(), grad2.data_ptr<scalar_t>(), g1.data_ptr<scalar_t>(), g2.data<scalar_t>());
      break;
  }
  //   cudaDeviceSynchronize();
  //   CHECK_CUDA_ERROR("chamfer_distance_backward_kernal");
  //   });
  return {g1, g2};
}
REGIST_PYTORCH_EXTENSION(chamfer_distance, {
  m.def("chamfer_distance_forward", &chamfer_distance_forward, "chamfer_distance_forward (CUDA)");
  m.def("chamfer_distance_backward", &chamfer_distance_backward, "chamfer_distance_backward (CUDA)");
})
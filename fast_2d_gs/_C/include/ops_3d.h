#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

#include "ops_3d_types.h"
namespace OPS_3D {
template <typename T>
using T2 = typename TypeSelecotr<T>::T2;

template <typename T>
using T3 = typename TypeSelecotr<T>::T3;

template <typename T>
using T4 = typename TypeSelecotr<T>::T4;

template <typename T, int M = 3, int N = 3>
__forceinline__ __device__ void zero_mat(T* A) {
#pragma unroll
  for (int i = 0; i < M * N; ++i) A[i] = 0;
}

template <typename T, int M = 3, int N = 3>
__forceinline__ __device__ void eye_mat(T* A) {
#pragma unroll
  for (int i = 0; i < M; ++i)
#pragma unroll
    for (int j = 0; j < N; ++j) A[i * N + j] = i == j;
}

template <typename T, int M = 3, int N = 3, int K = 3>
__forceinline__ __device__ void matmul(const T* A, const T* B, T* C) {  // A [M, K] @ B: [K, N] = C: [M, N]
#pragma unroll
  for (int i = 0; i < M; ++i) {
#pragma unroll
    for (int j = 0; j < N; ++j) {
#pragma unroll
      for (int k = 0; k < K; ++k) {
        C[i * N + j] += A[i * K + k] * B[k * N + j];
      }
    }
  }
}

template <typename T, int M = 3, int N = 3, int K = 3>
__forceinline__ __device__ void matmul_tn(const T* At, const T* B, T* C) {
// At [K, M], B: [K, N], C: [M, N]
#pragma unroll
  for (int i = 0; i < M; ++i) {
#pragma unroll
    for (int j = 0; j < N; ++j) {
#pragma unroll
      for (int k = 0; k < K; ++k) {
        C[i * N + j] += At[k * M + i] * B[k * N + j];
      }
    }
  }
}

template <typename T, int M = 3, int N = 3, int K = 3>
__forceinline__ __device__ void matmul_nt(const T* A, const T* Bt, T* C) {
// A [M, K], Bt: [N, K], C: [M, N]
#pragma unroll
  for (int i = 0; i < M; ++i) {
#pragma unroll
    for (int j = 0; j < N; ++j) {
#pragma unroll
      for (int k = 0; k < K; ++k) {
        C[i * N + j] += A[i * K + k] * Bt[j * K + k];
      }
    }
  }
}
template <typename T>
__forceinline__ __device__ void matmul_4x4x4(const T* A, const T* B, T* C) {
#pragma unroll
  for (int i = 0; i < 4; ++i) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        C[i * 4 + j] += A[i * 4 + k] * B[k * 4 + j];
      }
    }
  }
}
template <typename T = float>
__forceinline__ __device__ vec4<T> quaternion_normalize(const vec4<T>& q) {
  T norm = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
  norm   = T(1.) / max(sqrt(norm), T(1e-12));
  return q * norm;
}

template <typename T = float>
__forceinline__ __device__ vec4<T> quaternion_normalize(const T* q) {
  T norm = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
  norm   = T(1.) / max(sqrt(norm), T(1e-12));
  return vec4<T>(q[0] * norm, q[1] * norm, q[2] * norm, q[3] * norm);
}

template <typename T = float>
__forceinline__ __device__ void quaternion_to_R(const vec4<T>& q, T* R) {
  R[0] = 1 - 2 * (q.y * q.y + q.z * q.z);
  R[1] = 2 * (q.x * q.y - q.z * q.w);
  R[2] = 2 * (q.y * q.w + q.x * q.z);
  R[3] = 2 * (q.x * q.y + q.z * q.w);
  R[4] = 1 - 2 * (q.x * q.x + q.z * q.z);
  R[5] = 2 * (q.y * q.z - q.x * q.w);
  R[6] = 2 * (q.x * q.z - q.y * q.w);
  R[7] = 2 * (q.x * q.w + q.y * q.z);
  R[8] = 1 - 2 * (q.x * q.x + q.y * q.y);
}

template <typename T = float>
__forceinline__ __device__ void quaternion_to_R(const T* q, T* R) {
  R[0] = 1 - 2 * (q[1] * q[1] + q[2] * q[2]);
  R[1] = 2 * (q[0] * q[1] - q[2] * q[3]);
  R[2] = 2 * (q[1] * q[3] + q[0] * q[2]);
  R[3] = 2 * (q[0] * q[1] + q[2] * q[3]);
  R[4] = 1 - 2 * (q[0] * q[0] + q[2] * q[2]);
  R[5] = 2 * (q[1] * q[2] - q[0] * q[3]);
  R[6] = 2 * (q[0] * q[2] - q[1] * q[3]);
  R[7] = 2 * (q[0] * q[3] + q[1] * q[2]);
  R[8] = 1 - 2 * (q[0] * q[0] + q[1] * q[1]);
}

template <typename T = float>
__forceinline__ __device__ vec4<T> dL_quaternion_to_R(const vec4<T>& q, const T* dR) {
  vec4<T> dq;
  dq.x = 2 * (-2 * q.x * (dR[4] + dR[8]) + q.y * (dR[1] + dR[3]) + q.z * (dR[2] + dR[6]) + q.w * (dR[7] - dR[5]));
  dq.y = 2 * (q.x * (dR[1] + dR[3]) - 2 * q.y * (dR[0] + dR[8]) + q.z * (dR[5] + dR[7]) + q.w * (dR[2] - dR[6]));
  dq.z = 2 * (q.x * (dR[2] + dR[6]) + q.y * (dR[5] + dR[7]) - 2 * q.z * (dR[0] + dR[4]) + q.w * (dR[3] - dR[1]));
  dq.w = 2 * (q.x * (dR[7] - dR[5]) + q.y * (dR[2] - dR[6]) + q.z * (dR[3] - dR[1]));
  return dq;
}

template <typename T = float>
__forceinline__ __host__ __device__ void dL_quaternion_to_R(const T* q, const T* dR, T* dq) {
  dq[0] = 2 * (-2 * q[0] * (dR[4] + dR[8]) + q[1] * (dR[1] + dR[3]) + q[2] * (dR[2] + dR[6]) + q[3] * (dR[7] - dR[5]));
  dq[1] = 2 * (q[0] * (dR[1] + dR[3]) - 2 * q[1] * (dR[0] + dR[8]) + q[2] * (dR[5] + dR[7]) + q[3] * (dR[2] - dR[6]));
  dq[2] = 2 * (q[0] * (dR[2] + dR[6]) + q[1] * (dR[5] + dR[7]) - 2 * q[2] * (dR[0] + dR[4]) + q[3] * (dR[3] - dR[1]));
  dq[3] = 2 * (q[0] * (dR[7] - dR[5]) + q[1] * (dR[2] - dR[6]) + q[2] * (dR[3] - dR[1]));
}

template <typename T>
__forceinline__ __device__ vec3<T> xfm_p_3x3(const vec3<T>& p, const T* matrix) {
  vec3<T> transformed = {
      matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z,
      matrix[3] * p.x + matrix[4] * p.y + matrix[5] * p.z,
      matrix[6] * p.x + matrix[7] * p.y + matrix[8] * p.z,
  };
  return transformed;
}

template <typename T>
__forceinline__ __device__ vec3<T> xfm_p_3x3(const vec3<T>& p, const mat3<T>& matrix) {
  auto m              = matrix._data;
  vec3<T> transformed = {
      m[0] * p.x + m[1] * p.y + m[2] * p.z,
      m[3] * p.x + m[4] * p.y + m[5] * p.z,
      m[6] * p.x + m[7] * p.y + m[8] * p.z,
  };
  return transformed;
}

template <typename T>
__forceinline__ __device__ vec3<T> xfm_p_4x3(const vec3<T>& p, const T* matrix) {
  vec3<T> transformed = {
      matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z + matrix[3],
      matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z + matrix[7],
      matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z + matrix[11],
  };
  return transformed;
}

template <typename T>
__forceinline__ __device__ vec3<T> xfm_p_4x3(const vec3<T>& p, const mat4<T>& matrix) {
  auto m              = matrix._data;
  vec3<T> transformed = {
      m[0] * p.x + m[1] * p.y + m[2] * p.z + m[3],
      m[4] * p.x + m[5] * p.y + m[6] * p.z + m[7],
      m[8] * p.x + m[9] * p.y + m[10] * p.z + m[11],
  };
  return transformed;
}

template <typename T>
__forceinline__ __device__ vec3<T> xfm_inv_p_4x3(const vec3<T>& p, const T* matrix) {
  vec3<T> pp          = {p.x - matrix[3], p.y - matrix[7], p.z - matrix[11]};
  vec3<T> transformed = {
      matrix[0] * pp.x + matrix[4] * pp.y + matrix[8] * pp.z,
      matrix[1] * pp.x + matrix[5] * pp.y + matrix[9] * pp.z,
      matrix[2] * pp.x + matrix[6] * pp.y + matrix[10] * pp.z,
  };
  return transformed;
}

template <typename T>
__forceinline__ __device__ vec3<T> xfm_inv_p_4x3(const vec3<T>& p, const mat4<T>& matrix) {
  auto m              = matrix._data;
  vec3<T> pp          = {p.x - m[3], p.y - m[7], p.z - m[11]};
  vec3<T> transformed = {
      m[0] * pp.x + m[4] * pp.y + m[8] * pp.z,
      m[1] * pp.x + m[5] * pp.y + m[9] * pp.z,
      m[2] * pp.x + m[6] * pp.y + m[10] * pp.z,
  };
  return transformed;
}

template <typename T>
__forceinline__ __device__ vec4<T> xfm_p_4x4(const vec3<T>& p, const T* matrix) {
  vec4<T> transformed = {matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z + matrix[3],
      matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z + matrix[7],
      matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z + matrix[11],
      matrix[12] * p.x + matrix[13] * p.y + matrix[14] * p.z + matrix[15]};
  return transformed;
}

template <typename T>
__forceinline__ __device__ vec4<T> xfm_p_4x4(const vec4<T>& p, const T* matrix) {
  vec4<T> transformed = {matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z + p.w * matrix[3],
      matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z + p.w * matrix[7],
      matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z + p.w * matrix[11],
      matrix[12] * p.x + matrix[13] * p.y + matrix[14] * p.z + p.w * matrix[15]};
  return transformed;
}

template <typename T>
__forceinline__ __device__ vec3<T> xfm_v_4x3(const vec3<T>& p, const T* matrix) {
  vec3<T> transformed = {
      matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z,
      matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z,
      matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z,
  };
  return transformed;
}

template <typename T>
__forceinline__ __device__ vec3<T> xfm_v_4x3_T(const vec3<T>& p, const T* matrix) {
  vec3<T> transformed = {
      matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z,
      matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z,
      matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z,
  };
  return transformed;
}

template <typename T>
__forceinline__ __device__ vec3<T> xfm_3x3_T(const vec3<T>& p, const T* matrix) {
  vec3<T> transformed = {
      matrix[0] * p.x + matrix[3] * p.y + matrix[6] * p.z,
      matrix[1] * p.x + matrix[4] * p.y + matrix[7] * p.z,
      matrix[2] * p.x + matrix[5] * p.y + matrix[8] * p.z,
  };
  return transformed;
}

template <typename T>
__forceinline__ __device__ vec3<T> xfm_3x3_T(const vec3<T>& p, const mat3<T>& matrix) {
  auto m              = matrix._data;
  vec3<T> transformed = {
      m[0] * p.x + m[3] * p.y + m[6] * p.z,
      m[1] * p.x + m[4] * p.y + m[7] * p.z,
      m[2] * p.x + m[5] * p.y + m[8] * p.z,
  };
  return transformed;
}

}  // namespace OPS_3D
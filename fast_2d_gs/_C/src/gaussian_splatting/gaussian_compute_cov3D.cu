#if __INTELLISENSE__
#define __CUDACC__
#endif
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <stdio.h>

#include "ops_3d.h"
#include "util.cuh"

namespace cg = cooperative_groups;

namespace GaussianRasterizer {
using namespace OPS_3D;

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care of quaternion normalization.
template <typename T>
__device__ void computeCov3D(const vec3<T>& s, const vec4<T>& rot, T* cov3D) {
  // Create scaling matrix
  T R[9] = {0};
  quaternion_to_R(rot, R);  // Normalize quaternion to get valid rotation

  T sx2 = s.x * s.x;
  T sy2 = s.y * s.y;
  T sz2 = s.z * s.z;

  // Covariance is symmetric, only store upper right
  cov3D[0] = R[0] * R[0] * sx2 + R[1] * R[1] * sy2 + R[2] * R[2] * sz2;
  cov3D[1] = R[0] * R[3] * sx2 + R[1] * R[4] * sy2 + R[2] * R[5] * sz2;
  cov3D[2] = R[0] * R[6] * sx2 + R[1] * R[7] * sy2 + R[2] * R[8] * sz2;
  cov3D[3] = R[3] * R[3] * sx2 + R[4] * R[4] * sy2 + R[5] * R[5] * sz2;
  cov3D[4] = R[3] * R[6] * sx2 + R[4] * R[7] * sy2 + R[5] * R[8] * sz2;
  cov3D[5] = R[6] * R[6] * sx2 + R[7] * R[7] * sy2 + R[8] * R[8] * sz2;
}

// Backward pass for the conversion of scale and rotation to a 3D covariance matrix for each Gaussian.
template <typename T>
__device__ void computeCov3D_backward(
    const vec3<T>& scale, const vec4<T>& rot, const T* dL_dcov3Ds, vec3<T>& dL_dscales, vec4<T>& dL_drots) {
  T R[9] = {0};
  quaternion_to_R(rot, R);  // Recompute (intermediate) results for the 3D covariance computation.

  const T* dL_dcov3D = dL_dcov3Ds;
  vec3<T> gs;
  gs.x = R[0] * R[0] * dL_dcov3D[0] + R[0] * R[3] * dL_dcov3D[1] + R[0] * R[6] * dL_dcov3D[2] +
         R[3] * R[3] * dL_dcov3D[3] + R[3] * R[6] * dL_dcov3D[4] + R[6] * R[6] * dL_dcov3D[5];
  gs.y = R[1] * R[1] * dL_dcov3D[0] + R[1] * R[4] * dL_dcov3D[1] + R[1] * R[7] * dL_dcov3D[2] +
         R[4] * R[4] * dL_dcov3D[3] + R[4] * R[7] * dL_dcov3D[4] + R[7] * R[7] * dL_dcov3D[5];
  gs.z = R[2] * R[2] * dL_dcov3D[0] + R[2] * R[5] * dL_dcov3D[1] + R[2] * R[8] * dL_dcov3D[2] +
         R[5] * R[5] * dL_dcov3D[3] + R[5] * R[8] * dL_dcov3D[4] + R[8] * R[8] * dL_dcov3D[5];
  gs.x *= 2 * scale.x;
  gs.y *= 2 * scale.y;
  gs.z *= 2 * scale.z;
  dL_dscales = gs;

  T sx2 = scale.x * scale.x;
  T sy2 = scale.y * scale.y;
  T sz2 = scale.z * scale.z;

  T dL_dR[9];
  dL_dR[0] = (2 * R[0] * dL_dcov3D[0] + R[3] * dL_dcov3D[1] + R[6] * dL_dcov3D[2]) * sx2;
  dL_dR[1] = (2 * R[1] * dL_dcov3D[0] + R[4] * dL_dcov3D[1] + R[7] * dL_dcov3D[2]) * sy2;
  dL_dR[2] = (2 * R[2] * dL_dcov3D[0] + R[5] * dL_dcov3D[1] + R[8] * dL_dcov3D[2]) * sz2;
  dL_dR[3] = (2 * R[3] * dL_dcov3D[3] + R[0] * dL_dcov3D[1] + R[6] * dL_dcov3D[4]) * sx2;
  dL_dR[4] = (2 * R[4] * dL_dcov3D[3] + R[1] * dL_dcov3D[1] + R[7] * dL_dcov3D[4]) * sy2;
  dL_dR[5] = (2 * R[5] * dL_dcov3D[3] + R[2] * dL_dcov3D[1] + R[8] * dL_dcov3D[4]) * sz2;
  dL_dR[6] = (2 * R[6] * dL_dcov3D[5] + R[0] * dL_dcov3D[2] + R[3] * dL_dcov3D[4]) * sx2;
  dL_dR[7] = (2 * R[7] * dL_dcov3D[5] + R[1] * dL_dcov3D[2] + R[4] * dL_dcov3D[4]) * sy2;
  dL_dR[8] = (2 * R[8] * dL_dcov3D[5] + R[2] * dL_dcov3D[2] + R[5] * dL_dcov3D[4]) * sz2;

  dL_drots = dL_quaternion_to_R(rot, dL_dR);  // Gradients of loss w.r.t. normalized quaternion
}

// Backward pass for the conversion of scale and rotation to a 3D covariance matrix for each Gaussian.
template <typename T>
__device__ void computeCov3D_backward(const T* scale, const T* rot, const T* dL_dcov3D, T* dL_dscales, T* dL_drots) {
  T R[9] = {0};
  quaternion_to_R(rot, R);  // Recompute (intermediate) results for the 3D covariance computation.

  if (dL_dscales != nullptr) {
    T grad_scale;
    grad_scale = R[0] * R[0] * dL_dcov3D[0] + R[0] * R[3] * dL_dcov3D[1] + R[0] * R[6] * dL_dcov3D[2] +
                 R[3] * R[3] * dL_dcov3D[3] + R[3] * R[6] * dL_dcov3D[4] + R[6] * R[6] * dL_dcov3D[5];
    dL_dscales[0] = grad_scale * 2 * scale[0];
    grad_scale    = R[1] * R[1] * dL_dcov3D[0] + R[1] * R[4] * dL_dcov3D[1] + R[1] * R[7] * dL_dcov3D[2] +
                 R[4] * R[4] * dL_dcov3D[3] + R[4] * R[7] * dL_dcov3D[4] + R[7] * R[7] * dL_dcov3D[5];
    dL_dscales[1] = grad_scale * 2 * scale[1];
    grad_scale    = R[2] * R[2] * dL_dcov3D[0] + R[2] * R[5] * dL_dcov3D[1] + R[2] * R[8] * dL_dcov3D[2] +
                 R[5] * R[5] * dL_dcov3D[3] + R[5] * R[8] * dL_dcov3D[4] + R[8] * R[8] * dL_dcov3D[5];
    dL_dscales[2] = grad_scale * 2 * scale[2];
  }

  if (dL_drots == nullptr) return;
  T sx2 = scale[0] * scale[0];
  T sy2 = scale[1] * scale[1];
  T sz2 = scale[2] * scale[2];

  T dL_dR[9];
  dL_dR[0] = (2 * R[0] * dL_dcov3D[0] + R[3] * dL_dcov3D[1] + R[6] * dL_dcov3D[2]) * sx2;
  dL_dR[1] = (2 * R[1] * dL_dcov3D[0] + R[4] * dL_dcov3D[1] + R[7] * dL_dcov3D[2]) * sy2;
  dL_dR[2] = (2 * R[2] * dL_dcov3D[0] + R[5] * dL_dcov3D[1] + R[8] * dL_dcov3D[2]) * sz2;
  dL_dR[3] = (2 * R[3] * dL_dcov3D[3] + R[0] * dL_dcov3D[1] + R[6] * dL_dcov3D[4]) * sx2;
  dL_dR[4] = (2 * R[4] * dL_dcov3D[3] + R[1] * dL_dcov3D[1] + R[7] * dL_dcov3D[4]) * sy2;
  dL_dR[5] = (2 * R[5] * dL_dcov3D[3] + R[2] * dL_dcov3D[1] + R[8] * dL_dcov3D[4]) * sz2;
  dL_dR[6] = (2 * R[6] * dL_dcov3D[5] + R[0] * dL_dcov3D[2] + R[3] * dL_dcov3D[4]) * sx2;
  dL_dR[7] = (2 * R[7] * dL_dcov3D[5] + R[1] * dL_dcov3D[2] + R[4] * dL_dcov3D[4]) * sy2;
  dL_dR[8] = (2 * R[8] * dL_dcov3D[5] + R[2] * dL_dcov3D[2] + R[5] * dL_dcov3D[4]) * sz2;

  dL_quaternion_to_R(rot, dL_dR, dL_drots);  // Gradients of loss w.r.t. normalized quaternion
}

template <typename T>
__global__ void compute_cov3D_forward_kernel(
    int N, const T* __restrict__ scale, const T* __restrict__ quaternion, T* __restrict__ cov3D) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < N) {
    computeCov3D<T>(*((vec3<T>*) (scale + idx * 3)), *((vec4<T>*) (quaternion + idx * 4)), cov3D + idx * 6);
  }
}

Tensor compute_cov3D_forward(Tensor q, Tensor s) {
  CHECK_CUDA(q);
  CHECK_CUDA(s);
  auto shape              = s.sizes().vec();
  int N                   = s.numel() / 3;
  shape[shape.size() - 1] = 4;
  BCNN_ASSERT(q.sizes().vec() == shape && s.size(-1) == 3, "q.shape[:-1] == s.shape[:-1]");
  shape[shape.size() - 1] = 6;
  Tensor cov3D            = q.new_zeros(shape);

  AT_DISPATCH_FLOATING_TYPES(q.scalar_type(), "compute_cov3D_forward", [&] {
    compute_cov3D_forward_kernel<scalar_t> KERNEL_ARG(div_round_up(N, 256), 256)(
        N, s.contiguous().data_ptr<scalar_t>(), q.contiguous().data_ptr<scalar_t>(), cov3D.data<scalar_t>());
  });
  return cov3D;
}

template <typename T>
__global__ void compute_cov3D_backward_kernel(int N, const T* __restrict__ quaternion, const T* __restrict__ scale,
    const T* __restrict__ dL_dcov3D, T* __restrict__ dL_dq, T* __restrict__ dL_ds) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < N) {
    // computeCov3D_backward<T, vec3<T>, vec4<T>>(
    // idx, *((vec3<T>*) (scale + idx * 3)), *((vec4<T>*) (quaternion + idx * 4)), dL_dcov3D, (vec3<T>*) dL_ds,
    // (vec4<T>*) dL_dq);
    computeCov3D_backward<T>(
        scale + idx * 3, quaternion + idx * 4, dL_dcov3D + idx * 6, dL_ds + idx * 3, dL_dq + idx * 4);
  }
}

vector<Tensor> compute_cov3D_backward(Tensor q, Tensor s, Tensor dL_dCov3D) {
  Tensor dq = torch::zeros_like(q);
  Tensor ds = torch::zeros_like(s);
  int N     = s.numel() / 3;
  AT_DISPATCH_FLOATING_TYPES(q.scalar_type(), "compute_cov3D_backward_kernel", [&] {
    compute_cov3D_backward_kernel<scalar_t> KERNEL_ARG(div_round_up(N, 256), 256)(N,
        q.contiguous().data_ptr<scalar_t>(), s.contiguous().data_ptr<scalar_t>(),
        dL_dCov3D.contiguous().data_ptr<scalar_t>(), dq.data<scalar_t>(), ds.data<scalar_t>());
  });
  return {dq, ds};
}

// instance functions
#define INSTANCE_FUNC(T)                                                                    \
  template __device__ void computeCov3D<T>(const vec3<T>& s, const vec4<T>& rot, T* cov3D); \
  template __device__ void computeCov3D_backward<T>(                                        \
      const T* scale, const T* rot, const T* dL_dcov3D, T* dL_dscales, T* dL_drots);

INSTANCE_FUNC(float);
INSTANCE_FUNC(double);

REGIST_PYTORCH_EXTENSION(gs_gaussian_compute_cov3D, {
  m.def("gs_compute_cov3D_forward", &compute_cov3D_forward, "compute_cov3D_forward (CUDA)");
  m.def("gs_compute_cov3D_backward", &compute_cov3D_backward, "compute_cov3D_backward (CUDA)");
})
}  // namespace GaussianRasterizer
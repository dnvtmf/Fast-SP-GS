#if __INTELLISENSE__
#define __CUDACC__
#endif
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda.h>
#include <stdio.h>

#include "ops_3d.h"
#include "util.cuh"

namespace cg = cooperative_groups;

namespace GaussianRasterizer {
using namespace OPS_3D;

// Forward version of 2D covariance matrix computation
template <typename T = float>
__device__ vec3<T> computeCov2D(
    const vec3<T>& mean, T focal_x, T focal_y, T tan_fovx, T tan_fovy, const T* cov3D, const T* viewmatrix) {
  // The following models the steps outlined by equations 29 and 31 in "EWA Splatting" (Zwicker et al., 2002).
  // Additionally considers aspect / scaling of viewport. Transposes used to account for row-/column-major conventions.
  vec3<T> t = xfm_p_4x3(mean, viewmatrix);
  auto idx  = cg::this_grid().thread_rank();

  // if (idx == 6116449) {
  //   printf("\033[31mp_view: %.6f, %.6f, %.6f\n\033[0m", t.x, t.y, t.z);
  // }

  const T limx = T(1.3) * tan_fovx * t.z;
  const T limy = T(1.3) * tan_fovy * t.z;
  t.x          = clamp(t.x, -limx, limx);
  t.y          = clamp(t.y, -limy, limy);
  // if (idx == 6116449) {
  //   printf("\033[31mclamped_x, y=%.6e, %.6e, limit=%.6e, %.6e\n\033[0m", t.x, t.y, limx, limy);
  //   printf("\033[31mfocal x=%.6e, y=%.6e\n\033[0m", focal_x, focal_y);
  // }

  T J[6] = {focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z), 0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z)};
  // if (idx == 6116449) {
  //   printf("\033[31mJ: %.6e, %.6e, %.6e, %.6e, %.6e, %.6e\n\033[0m", J[0], J[1], J[2], J[3], J[4], J[5]);
  // }
  T W[9] = {viewmatrix[0], viewmatrix[1], viewmatrix[2], viewmatrix[4], viewmatrix[5], viewmatrix[6], viewmatrix[8],
      viewmatrix[9], viewmatrix[10]};

  T M[6] = {0};
  matmul<T, 2, 3, 3>(J, W, M);

  T Vrk[9] = {cov3D[0], cov3D[1], cov3D[2], cov3D[1], cov3D[3], cov3D[4], cov3D[2], cov3D[4], cov3D[5]};

  T* tmp = J;
  zero_mat<T, 2, 3>(tmp);
  matmul<T, 2, 3, 3>(M, Vrk, tmp);
  T* cov = Vrk;
  zero_mat<T, 2, 2>(cov);
  matmul_nt<T, 2, 2, 3>(tmp, M, cov);

  // Apply low-pass filter: every Gaussian should be at least one pixel wide/high. Discard 3rd row and column.
  cov[0] += T(0.3);
  cov[3] += T(0.3);
  return {T(cov[0]), T(cov[1]), T(cov[3])};
}

template <typename T>
__device__ vec3<T> computeCov2DBackward(
    // inputs
    const vec3<T>& mean, const T* cov3D,
    // camear inputs
    T fx, T fy, T tan_fovx, T tan_fovy, const T* view_matrix,
    // grad_outputs and grad_inputs
    const T* dL_dcov2D, const T* dL_dconic, T* dL_dcov3D, T* dL_dvm) {
  vec3<T> t = xfm_p_4x3(mean, view_matrix);

  const T limx       = T(1.3) * tan_fovx * t.z;
  const T limy       = T(1.3) * tan_fovy * t.z;
  const T x_grad_mul = t.x < -limx || t.x > limx ? 0 : 1;
  const T y_grad_mul = t.y < -limy || t.y > limy ? 0 : 1;
  t.x                = clamp(t.x, -limx, limx);
  t.y                = clamp(t.y, -limy, limy);

  T J[6]   = {fx / t.z, 0.0f, -(fx * t.x) / (t.z * t.z), 0.0f, fy / t.z, -(fy * t.y) / (t.z * t.z)};
  T R[9]   = {view_matrix[0], view_matrix[1], view_matrix[2], view_matrix[4], view_matrix[5], view_matrix[6],
        view_matrix[8], view_matrix[9], view_matrix[10]};
  T Vrk[9] = {cov3D[0], cov3D[1], cov3D[2], cov3D[1], cov3D[3], cov3D[4], cov3D[2], cov3D[4], cov3D[5]};

  T M[6] = {0};
  matmul<T, 2, 3, 3>(J, R, M);
  T tmp[6] = {0};
  // zero_mat(tmp);
  T cov2D[4] = {0};  // = (J @ R) Vrk (J @ R)^T
  matmul<T, 2, 3, 3>(M, Vrk, tmp);
  matmul_nt<T, 2, 2, 3>(tmp, M, cov2D);

  // Use helper variables for 2D covariance entries. More compact.
  T a = cov2D[0] += 0.3f;
  T b = cov2D[1];
  T c = cov2D[3] += 0.3f;

  T dL_da = 0, dL_db = 0, dL_dc = 0;

  // Gradients of loss w.r.t. entries of 2D covariance matrix,
  if (dL_dconic != nullptr) {
    // given gradients of loss w.r.t. conic matrix (inverse covariance matrix).
    T denom = a * c - b * b;
    if (denom == 0) return;
    T denom2inv = 1. / ((denom * denom) + 0.0000001f);
    // e.g., dL / da = dL / d_conic_a * d_conic_a / d_a
    T t   = c * dL_dconic[0] - b * dL_dconic[1] + a * dL_dconic[2];
    dL_da = denom2inv * (-c * t + denom * dL_dconic[2]);
    dL_dc = denom2inv * (-a * t + denom * dL_dconic[0]);
    dL_db = denom2inv * (T(2) * b * t - denom * dL_dconic[1]);
  } else {
    dL_da = dL_dcov2D[0];
    dL_db = dL_dcov2D[1];
    dL_dc = dL_dcov2D[2];
  }

  if (dL_dcov3D != nullptr) {
    // Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry,
    // given gradients w.r.t. 2D covariance matrix (diagonal).
    // cov2D = transpose(M) * transpose(Vrk) * M;
    dL_dcov3D[0] = (M[0] * M[0] * dL_da + M[0] * M[3] * dL_db + M[3] * M[3] * dL_dc);
    dL_dcov3D[3] = (M[1] * M[1] * dL_da + M[1] * M[4] * dL_db + M[4] * M[4] * dL_dc);
    dL_dcov3D[5] = (M[2] * M[2] * dL_da + M[2] * M[5] * dL_db + M[5] * M[5] * dL_dc);

    // Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry,
    // given gradients w.r.t. 2D covariance matrix (off-diagonal).
    // Off-diagonal elements appear twice --> double the gradient.
    // cov2D = M @ transpose(Vrk) @ transpose(M) ;
    dL_dcov3D[1] = 2 * M[0] * M[1] * dL_da + (M[0] * M[4] + M[1] * M[3]) * dL_db + 2 * M[3] * M[4] * dL_dc;
    dL_dcov3D[2] = 2 * M[0] * M[2] * dL_da + (M[0] * M[5] + M[2] * M[3]) * dL_db + 2 * M[3] * M[5] * dL_dc;
    dL_dcov3D[4] = 2 * M[1] * M[2] * dL_da + (M[1] * M[5] + M[2] * M[4]) * dL_db + 2 * M[4] * M[5] * dL_dc;
  }
  // Gradients of loss w.r.t. upper 2x3 portion of intermediate matrix M
  // cov2D = M @ transpose(Vrk) @ transpose(M) ;
  T dL_dT[6];
  dL_dT[0] = 2 * (M[0] * cov3D[0] + M[1] * cov3D[1] + M[2] * cov3D[2]) * dL_da +
             (M[3] * cov3D[0] + M[4] * cov3D[1] + M[5] * cov3D[2]) * dL_db;
  dL_dT[1] = 2 * (M[0] * cov3D[1] + M[1] * cov3D[3] + M[2] * cov3D[4]) * dL_da +
             (M[3] * cov3D[1] + M[4] * cov3D[3] + M[5] * cov3D[4]) * dL_db;
  dL_dT[2] = 2 * (M[0] * cov3D[2] + M[1] * cov3D[4] + M[2] * cov3D[5]) * dL_da +
             (M[3] * cov3D[2] + M[4] * cov3D[4] + M[5] * cov3D[5]) * dL_db;
  dL_dT[3] = 2 * (M[3] * cov3D[0] + M[4] * cov3D[1] + M[5] * cov3D[2]) * dL_dc +
             (M[0] * cov3D[0] + M[1] * cov3D[1] + M[2] * cov3D[2]) * dL_db;
  dL_dT[4] = 2 * (M[3] * cov3D[1] + M[4] * cov3D[3] + M[5] * cov3D[4]) * dL_dc +
             (M[0] * cov3D[1] + M[1] * cov3D[3] + M[2] * cov3D[4]) * dL_db;
  dL_dT[5] = 2 * (M[3] * cov3D[2] + M[4] * cov3D[4] + M[5] * cov3D[5]) * dL_dc +
             (M[0] * cov3D[2] + M[1] * cov3D[4] + M[2] * cov3D[5]) * dL_db;

  // Gradients of loss w.r.t. upper 2x3 non-zero entries of Jacobian matrix; M=J @ R -> dJ = dM @ R.T
  T dL_dJ00 = dL_dT[0] * R[0] + dL_dT[1] * R[1] + dL_dT[2] * R[2];
  T dL_dJ02 = dL_dT[0] * R[6] + dL_dT[1] * R[7] + dL_dT[2] * R[8];
  T dL_dJ11 = dL_dT[3] * R[3] + dL_dT[4] * R[4] + dL_dT[5] * R[5];
  T dL_dJ12 = dL_dT[3] * R[6] + dL_dT[4] * R[7] + dL_dT[5] * R[8];

  T tz  = 1.f / t.z;
  T tz2 = tz * tz;
  T tz3 = tz2 * tz;

  // Gradients of loss w.r.t. transformed Gaussian mean t
  T dL_dtx = x_grad_mul * -fx * tz2 * dL_dJ02;
  T dL_dty = y_grad_mul * -fy * tz2 * dL_dJ12;
  T dL_dtz = -fx * tz2 * dL_dJ00 - fy * tz2 * dL_dJ11 + (2 * fx * t.x) * tz3 * dL_dJ02 + (2 * fy * t.y) * tz3 * dL_dJ12;

  // Gradients of loss w.r.t. view matrix; M=J @ R
  if (dL_dvm != nullptr) {
    zero_mat<T, 3, 3>(R);
    matmul_tn<T, 3, 3, 2>(J, dL_dT, R);
    // Gradients of loss w.r.t. view matrix, ie, t = xfm_p_4x3(mean, view_matrix);

    dL_dvm[0]  = R[0] + dL_dtx * mean.x;
    dL_dvm[1]  = R[1] + dL_dtx * mean.y;
    dL_dvm[2]  = R[2] + dL_dtx * mean.z;
    dL_dvm[3]  = dL_dtx;
    dL_dvm[4]  = R[3] + dL_dty * mean.x;
    dL_dvm[5]  = R[4] + dL_dty * mean.y;
    dL_dvm[6]  = R[5] + dL_dty * mean.z;
    dL_dvm[7]  = dL_dty;
    dL_dvm[8]  = R[6] + dL_dtz * mean.x;
    dL_dvm[9]  = R[7] + dL_dtz * mean.y;
    dL_dvm[10] = R[8] + dL_dtz * mean.z;
    dL_dvm[11] = dL_dtz;
  }
  // Account for transformation of mean to t  // t = xfm_p_4x3(mean, view_matrix);
  return xfm_v_4x3_T<T>({dL_dtx, dL_dty, dL_dtz}, view_matrix);
}

template <typename T = float>
__global__ void compute_cov2D_forward_kernel(int N, const T* __restrict__ cov3D, const T* __restrict__ mean,
    const T* __restrict__ viewmatrix, T focal_x, T focal_y, T tan_fovx, T tan_fovy, T* __restrict__ cov2D) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < N) {
    vec3<T> t =
        computeCov2D(*(vec3<T>*) (mean + idx * 3), focal_x, focal_y, tan_fovx, tan_fovy, cov3D + idx * 6, viewmatrix);
    cov2D[idx * 3 + 0] = t.x;
    cov2D[idx * 3 + 1] = t.y;
    cov2D[idx * 3 + 2] = t.z;
  }
}

Tensor compute_cov2D_forward(
    Tensor cov3D, Tensor mean, Tensor viewmatrix, float focal_x, float focal_y, float tan_fovx, float tan_fovy) {
  int N        = mean.numel() / 3;
  Tensor cov2D = torch::zeros_like(mean);

  AT_DISPATCH_FLOATING_TYPES(cov3D.scalar_type(), "compute_cov2D_forward", [&] {
    compute_cov2D_forward_kernel<scalar_t> KERNEL_ARG(div_round_up(N, 256), 256)(N,
        cov3D.contiguous().data_ptr<scalar_t>(), mean.contiguous().data_ptr<scalar_t>(),
        viewmatrix.contiguous().data_ptr<scalar_t>(), focal_x, focal_y, tan_fovx, tan_fovy, cov2D.data<scalar_t>());
  });
  return cov2D;
}

template <typename T>
__global__ void compute_cov2D_backward_kernel(int P, const vec3<T>* means, const T* cov3Ds, const T fx, T fy,
    const T tan_fovx, T tan_fovy, const T* view_matrix, const T* dL_dcov2D, vec3<T>* dL_dmeans, T* dL_dcov, T* dL_dvm) {
  auto idx = cg::this_grid().thread_rank();

  T temp[12] = {0};
  if (idx < P) {
    dL_dmeans[idx] = computeCov2DBackward<T>(means[idx], cov3Ds + idx * 6, fx, fy, tan_fovx, tan_fovy, view_matrix,
        dL_dcov2D + idx * 3, nullptr, dL_dcov + idx * 6, dL_dvm == nullptr ? nullptr : temp);
  }
  if (dL_dvm == nullptr) return;
  for (int i = 0; i < 12; ++i) {
    T W = temp[i];
    reduce_sum_block<T, false>(W);
    if (threadIdx.x == 0) atomicAdd(dL_dvm + i, W);
  }
}

vector<Tensor> compute_cov2D_backward(Tensor cov3D, Tensor mean, Tensor viewmatrix, float focal_x, float focal_y,
    float tan_fovx, float tan_fovy, Tensor grad_cov2D, torch::optional<Tensor> grad_vm) {
  int N             = mean.numel() / 3;
  Tensor grad_cov3D = torch::zeros_like(cov3D);
  Tensor grad_mean  = torch::zeros_like(mean);
  if (grad_vm.has_value()) CHECK_INPUT(grad_vm.value());

  AT_DISPATCH_FLOATING_TYPES(cov3D.scalar_type(), "compute_cov2D_backward", [&] {
    compute_cov2D_backward_kernel<scalar_t> KERNEL_ARG(div_round_up(N, 256), 256)(N,
        (vec3<scalar_t>*) mean.contiguous().data_ptr<scalar_t>(), cov3D.contiguous().data_ptr<scalar_t>(), focal_x,
        focal_y, tan_fovx, tan_fovy, viewmatrix.contiguous().data_ptr<scalar_t>(),
        grad_cov2D.contiguous().data_ptr<scalar_t>(), (vec3<scalar_t>*) grad_mean.data_ptr<scalar_t>(),
        grad_cov3D.data_ptr<scalar_t>(), grad_vm.has_value() ? grad_vm.value().data_ptr<scalar_t>() : nullptr);
  });

  return {grad_cov3D, grad_mean};
}

REGIST_PYTORCH_EXTENSION(gs_gaussian_compute_cov2D, {
  m.def("gs_compute_cov2D_forward", &compute_cov2D_forward, "compute_cov2D_forward (CUDA)");
  m.def("gs_compute_cov2D_backward", &compute_cov2D_backward, "compute_cov2D_backward (CUDA)");
})

#define INSTANCE_FUNC(T)                                                                                           \
  template __device__ vec3<T> computeCov2D<T>(                                                                     \
      const vec3<T>& mean, T focal_x, T focal_y, T tan_fovx, T tan_fovy, const T* cov3D, const T* viewmatrix);     \
  template __device__ vec3<T> computeCov2DBackward<T>(const vec3<T>& mean, const T* cov3D, T fx, T fy, T tan_fovx, \
      T tan_fovy, const T* view_matrix, const T* dL_dcov2D, const T* dL_dconic, T* dL_dcov3D, T* dL_dvm);

INSTANCE_FUNC(float);
INSTANCE_FUNC(double);

}  // namespace GaussianRasterizer
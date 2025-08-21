#include "spherical_harmonic.h"
#include "util.cuh"

namespace OPS_3D {

template <typename T = float>
__global__ void SH_to_RGB_forward_kernel(int degree, int P, int F, bool clamp, const vec3<T> *__restrict__ dirs_or_pos,
    const T *__restrict__ campos, const vec3<T> *__restrict__ shs, vec3<T> *__restrict__ rgb) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < P) {
    rgb[idx] = SH_to_RGB<T>(degree, dirs_or_pos[idx], (vec3<T> *) campos, shs + idx * F, nullptr, clamp);
  }
}

Tensor compute_color_from_SH_forward(
    Tensor dir_or_pos, torch::optional<Tensor> campos, Tensor sh_featrues, int degree, bool clamp) {
  CHECK_INPUT(dir_or_pos);
  if (campos.has_value()) CHECK_INPUT(campos.value());
  CHECK_INPUT(sh_featrues);
  BCNN_ASSERT(dir_or_pos.ndimension() == 2 && dir_or_pos.size(-1) == 3, "Error shape for dir_or_pos [P, 3]");
  if (campos.has_value())
    BCNN_ASSERT(campos.value().ndimension() == 1 && campos.value().numel() == 3, "Error shape for campos");
  int P = dir_or_pos.size(0);
  int F = sh_featrues.size(-2);
  CHECK_SHAPE(sh_featrues, P, F, 3);
  BCNN_ASSERT(0 <= degree && degree <= 7 && (1 << degree) <= F, "degree error");
  Tensor rgb = torch::zeros({P, 3}, dir_or_pos.options());

  AT_DISPATCH_FLOATING_TYPES(rgb.scalar_type(), "SH_to_RGB_forward", ([&] {
    auto campos_ptr = campos.has_value() ? campos.value().contiguous().data_ptr<scalar_t>() : nullptr;
    auto d_ptr      = (vec3<scalar_t> *) dir_or_pos.contiguous().data_ptr<scalar_t>();
    auto sh_ptr     = sh_featrues.contiguous().data_ptr<scalar_t>();
    auto rgb_ptr    = rgb.data_ptr<scalar_t>();

    SH_to_RGB_forward_kernel<scalar_t> KERNEL_ARG(div_round_up(P, 256), 256)(
        degree, P, F, clamp, d_ptr, campos_ptr, (vec3<scalar_t> *) sh_ptr, (vec3<scalar_t> *) rgb_ptr);
    CHECK_CUDA_ERROR("SH_to_RGB_forward_kernel");
  }));

  return rgb;
}

template <typename T = float>
__global__ void SH_to_RGB_backward_kernel(int P, int degree, int F, bool clamp, const vec3<T> *__restrict__ dirs_or_pos,
    const vec3<T> *__restrict__ campos, const T *__restrict__ shs, const vec3<T> *__restrict__ rgb,
    const vec3<T> *__restrict__ grad_rgb, vec3<T> *__restrict__ grad_dir, T *__restrict__ grad_campos,
    T *__restrict__ grad_shs) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  vec3<T> dxyz;
  if (idx < P) {
    dxyz = SH_to_RGB_backward<T>(degree, dirs_or_pos[idx], campos, rgb[idx], (vec3<T> *) (shs + idx * F * 3), nullptr,
        grad_rgb[idx], grad_shs == nullptr ? nullptr : (vec3<T> *) (grad_shs + idx * F * 3), nullptr,
        grad_dir != nullptr || grad_campos != nullptr, clamp);
    if (grad_dir != nullptr) grad_dir[idx] = dxyz;
  }
  if (grad_campos != nullptr) {
    reduce_sum_block<T, false>(dxyz.x);
    reduce_sum_block<T, false>(dxyz.y);
    reduce_sum_block<T, false>(dxyz.z);
    if (threadIdx.x == 0) {
      atomicAdd(grad_campos, -dxyz.x);
      atomicAdd(grad_campos + 1, -dxyz.y);
      atomicAdd(grad_campos + 2, -dxyz.z);
    }
  }
}

void compute_color_from_SH_backward(int degree, bool clamp, Tensor dir_or_pos, torch::optional<Tensor> campos,
    Tensor sh_featrues, Tensor rgb, Tensor grad_rgb, torch::optional<Tensor> &grad_dir,
    torch::optional<Tensor> grad_campos, torch::optional<Tensor> &grad_shs) {
  if (grad_dir.has_value()) {
    BCNN_ASSERT(dir_or_pos.sizes() == grad_dir.value().sizes(), "Error shape for grad_dir");
    CHECK_INPUT(grad_dir.value());
  }
  if (grad_shs.has_value()) {
    BCNN_ASSERT(sh_featrues.sizes() == grad_shs.value().sizes(), "Error shape for grad_shs");
    CHECK_INPUT(grad_shs.value());
  }
  if (grad_campos.has_value()) {
    BCNN_ASSERT(campos.value().sizes() == grad_campos.value().sizes(), "ERROR shape for grad_campos");
    CHECK_INPUT(grad_campos.value());
  }
  CHECK_INPUT(grad_rgb);
  CHECK_INPUT(rgb);
  int P = dir_or_pos.size(0);
  int D = dir_or_pos.size(1);
  int F = sh_featrues.size(-2);

  AT_DISPATCH_FLOATING_TYPES(grad_rgb.scalar_type(), "SH_to_RGB_backward", ([&] {
    auto cam_ptr = (vec3<scalar_t> *) (campos.has_value() ? campos.value().contiguous().data_ptr<scalar_t>() : nullptr);
    auto dir_ptr = (vec3<scalar_t> *) dir_or_pos.contiguous().data_ptr<scalar_t>();
    auto shs_ptr = sh_featrues.contiguous().data_ptr<scalar_t>();
    auto rgb_ptr = (vec3<scalar_t> *) rgb.data_ptr<scalar_t>();
    auto g_rgb_ptr = (vec3<scalar_t> *) grad_rgb.data_ptr<scalar_t>();
    auto g_dir_ptr = (vec3<scalar_t> *) (grad_dir.has_value() ? grad_dir.value().data_ptr<scalar_t>() : nullptr);
    auto g_shs_ptr = grad_shs.has_value() ? grad_shs.value().data_ptr<scalar_t>() : nullptr;
    auto g_cam_ptr = grad_campos.has_value() ? grad_campos.value().contiguous().data_ptr<scalar_t>() : nullptr;

    SH_to_RGB_backward_kernel<scalar_t> KERNEL_ARG(div_round_up(P, 256), 256)(
        P, degree, F, clamp, dir_ptr, cam_ptr, shs_ptr, rgb_ptr, g_rgb_ptr, g_dir_ptr, g_cam_ptr, g_shs_ptr);
  }));
}

REGIST_PYTORCH_EXTENSION(nerf_sh_encode, {
  m.def("SH_to_RGB_forward", &compute_color_from_SH_forward, "compute RGB color from SH forward (CUDA)");
  m.def("SH_to_RGB_backward", &compute_color_from_SH_backward, "compute RGB color from SH backward (CUDA)");
})

// #define INSTANCE_FUNC(T)                                                                                \
//   template __device__ __host__ vec3<T> SH_to_RGB<T>(                                                    \
//       int degree, const vec3<T> &dirs_or_pos, const vec3<T> *campos, const vec3<T> *sh, bool clamp);    \
//   template __device__ __host__ vec3<T> SH_to_RGB_backward<T>(int degree, const vec3<T> &dirs_or_pos,    \
//       const vec3<T> *campos, const vec3<T> &RGB, const vec3<T> *sh, vec3<T> grad_rgb, vec3<T> *grad_sh, \
//       bool need_grad_dir, bool clamp);

// INSTANCE_FUNC(float);
// INSTANCE_FUNC(double);
}  // namespace OPS_3D
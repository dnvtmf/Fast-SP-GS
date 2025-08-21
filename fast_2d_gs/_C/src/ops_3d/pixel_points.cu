#include <cuda.h>
#include <stdio.h>

#include "util.cuh"
#include "ops_3d.h"

namespace OPS_3D {

template <typename T>
__global__ void pixel_to_point_forward_kernel(int N, int W, int WH, const T* __restrict__ depths,
    const T2<T>* __restrict__ pixels, const T* __restrict__ Ts2v, const T* __restrict__ Tv2w, vec3<T>* points) {
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
  const int n   = blockIdx.y;
  __shared__ T Ts2v_n[9], Tv2w_n[16];
  if (threadIdx.x < 9) {
    Ts2v_n[threadIdx.x] = Ts2v[n * 9 + threadIdx.x];
  }
  if (Tv2w != nullptr && threadIdx.x >= 9 && threadIdx.x < 16 + 9) {
    Tv2w_n[threadIdx.x - 9] = Tv2w[n * 16 + threadIdx.x - 9];
  }
  points += n * WH + idx;
  depths += n * WH + idx;
  __syncthreads();
  if (idx < WH) {
    T z = depths[0];
    vec3<T> p;
    if (pixels != nullptr) {
      p = {pixels[n * WH + idx].x * z, pixels[n * WH + idx].y * z, z};
    } else {
      p = {T(idx % W) * z, T(idx / W) * z, z};
    }
    p = xfm_p_3x3(p, Ts2v_n);
    if (Tv2w != nullptr) p = xfm_p_4x3(p, Tv2w_n);
    points[0] = p;
  }
}

template <typename T>
__global__ void pixel_to_point_backward_kernel(int N, int W, int WH, const T* __restrict__ depths,
    const T2<T>* __restrict__ pixels, const T* __restrict__ Ts2v, const T* __restrict__ Tv2w,
    const vec3<T>* __restrict__ grad_points, T* __restrict__ grad_depths, T2<T>* __restrict__ grad_pixels,
    T* __restrict__ grad_Ts2v, T* __restrict__ grad_Tv2w) {
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
  const int n   = blockIdx.y;
  __shared__ T Ts2v_n[9], Tv2w_n[16];
  if (threadIdx.x < 9) {
    Ts2v_n[threadIdx.x] = Ts2v[n * 9 + threadIdx.x];
  }
  if (Tv2w != nullptr && threadIdx.x >= 9 && threadIdx.x < 16 + 9) {
    Tv2w_n[threadIdx.x - 9] = Tv2w[n * 16 + threadIdx.x - 9];
  }
  depths += n * WH + idx;
  grad_points += n * WH + idx;

  __syncthreads();
  vec3<T> p, p1, gp;
  T2<T> pixel;
  if (idx < WH) {
    T z   = depths[0];
    pixel = pixels != nullptr ? pixels[n * WH + idx] : (T2<T>){T(idx % W), T(idx / W)};
    p     = {pixel.x * z, pixel.y * z, z};
    p1    = xfm_p_3x3(p, Ts2v_n);
    gp    = grad_points[0];
  }
  bool need_grad = idx < WH && Tv2w != nullptr && grad_Tv2w != nullptr;
  T g            = 0;
  if (grad_Tv2w != nullptr) {
    g = need_grad ? gp.x * p1.x : 0;
    reduce_sum_block<T, false>(g);
    if (threadIdx.x == 0) atomicAdd(grad_Tv2w + n * 16 + 0, g);
    g = need_grad ? gp.x * p1.y : 0;
    reduce_sum_block<T, false>(g);
    if (threadIdx.x == 0) atomicAdd(grad_Tv2w + n * 16 + 1, g);
    g = need_grad ? gp.x * p1.z : 0;
    reduce_sum_block<T, false>(g);
    if (threadIdx.x == 0) atomicAdd(grad_Tv2w + n * 16 + 2, g);
    g = need_grad ? gp.x : 0;
    reduce_sum_block<T, false>(g);
    if (threadIdx.x == 0) atomicAdd(grad_Tv2w + n * 16 + 3, g);
    g = need_grad ? gp.y * p1.x : 0;
    reduce_sum_block<T, false>(g);
    if (threadIdx.x == 0) atomicAdd(grad_Tv2w + n * 16 + 4, g);
    g = need_grad ? gp.y * p1.y : 0;
    reduce_sum_block<T, false>(g);
    if (threadIdx.x == 0) atomicAdd(grad_Tv2w + n * 16 + 5, g);
    g = need_grad ? gp.y * p1.z : 0;
    reduce_sum_block<T, false>(g);
    if (threadIdx.x == 0) atomicAdd(grad_Tv2w + n * 16 + 6, g);
    g = need_grad ? gp.y : 0;
    reduce_sum_block<T, false>(g);
    if (threadIdx.x == 0) atomicAdd(grad_Tv2w + n * 16 + 7, g);
    g = need_grad ? gp.z * p1.x : 0;
    reduce_sum_block<T, false>(g);
    if (threadIdx.x == 0) atomicAdd(grad_Tv2w + n * 16 + 8, g);
    g = need_grad ? gp.z * p1.y : 0;
    reduce_sum_block<T, false>(g);
    if (threadIdx.x == 0) atomicAdd(grad_Tv2w + n * 16 + 9, g);
    g = need_grad ? gp.z * p1.z : 0;
    reduce_sum_block<T, false>(g);
    if (threadIdx.x == 0) atomicAdd(grad_Tv2w + n * 16 + 10, g);
    g = need_grad ? gp.z : 0;
    reduce_sum_block<T, false>(g);
    if (threadIdx.x == 0) atomicAdd(grad_Tv2w + n * 16 + 11, g);
  }
  if (idx < WH && Tv2w != nullptr) {
    gp = xfm_v_4x3_T(gp, Tv2w_n);
  }
  need_grad = idx < WH && grad_Ts2v != nullptr;
  if (grad_Ts2v != nullptr) {
    g = need_grad ? gp.x * p.x : 0;
    reduce_sum_block<T, false>(g);
    if (threadIdx.x == 0) atomicAdd(grad_Ts2v + n * 9 + 0, g);
    g = need_grad ? gp.x * p.y : 0;
    reduce_sum_block<T, false>(g);
    if (threadIdx.x == 0) atomicAdd(grad_Ts2v + n * 9 + 1, g);
    g = need_grad ? gp.x * p.z : 0;
    reduce_sum_block<T, false>(g);
    if (threadIdx.x == 0) atomicAdd(grad_Ts2v + n * 9 + 2, g);
    g = need_grad ? gp.y * p.x : 0;
    reduce_sum_block<T, false>(g);
    if (threadIdx.x == 0) atomicAdd(grad_Ts2v + n * 9 + 3, g);
    g = need_grad ? gp.y * p.y : 0;
    reduce_sum_block<T, false>(g);
    if (threadIdx.x == 0) atomicAdd(grad_Ts2v + n * 9 + 4, g);
    g = need_grad ? gp.y * p.z : 0;
    reduce_sum_block<T, false>(g);
    if (threadIdx.x == 0) atomicAdd(grad_Ts2v + n * 9 + 5, g);
    g = need_grad ? gp.z * p.x : 0;
    reduce_sum_block<T, false>(g);
    if (threadIdx.x == 0) atomicAdd(grad_Ts2v + n * 9 + 6, g);
    g = need_grad ? gp.z * p.y : 0;
    reduce_sum_block<T, false>(g);
    if (threadIdx.x == 0) atomicAdd(grad_Ts2v + n * 9 + 7, g);
    g = need_grad ? gp.z * p.z : 0;
    reduce_sum_block<T, false>(g);
    if (threadIdx.x == 0) atomicAdd(grad_Ts2v + n * 9 + 8, g);
  }
  if (grad_depths == nullptr && grad_pixels == nullptr) return;
  if (idx < WH) {
    gp = xfm_3x3_T(gp, Ts2v_n);
    if (grad_depths != nullptr) grad_depths[n * WH + idx] = pixel.x * gp.x + pixel.y * gp.y + gp.z;
    if (grad_pixels != nullptr) grad_pixels[n * WH + idx] = {gp.x * depths[0], gp.y * depths[0]};
  }
}

Tensor pixel2points_forward(Tensor depths, torch::optional<Tensor> pixels, Tensor Ts2v, torch::optional<Tensor> Tv2w) {
  CHECK_INPUT(depths);
  BCNN_ASSERT(depths.ndimension() == 3, "depths must have shape [N, H, W]");
  int N = depths.size(0);
  int H = depths.size(1);
  int W = depths.size(2);
  if (pixels.has_value()) {
    CHECK_INPUT(pixels.value());
    CHECK_SHAPE(pixels.value(), N, H, W, 2);
  }
  CHECK_INPUT(Ts2v);
  CHECK_SHAPE(Ts2v, N, 3, 3);
  if (Tv2w.has_value()) {
    CHECK_INPUT(Tv2w.value());
    CHECK_SHAPE(Tv2w.value(), N, 4, 4);
  }
  Tensor points = torch::zeros({N, H, W, 3}, depths.options());

  AT_DISPATCH_FLOATING_TYPES(depths.scalar_type(), "pixel_to_point_forward_kernel", [&] {
    pixel_to_point_forward_kernel<scalar_t> KERNEL_ARG(dim3(div_round_up(H * W, 256), N), 256)(N, W, W * H,
        depths.data_ptr<scalar_t>(),
        (T2<scalar_t>*) (pixels.has_value() ? pixels.value().data_ptr<scalar_t>() : nullptr), Ts2v.data_ptr<scalar_t>(),
        Tv2w.has_value() ? Tv2w.value().data_ptr<scalar_t>() : nullptr, (vec3<scalar_t>*) points.data_ptr<scalar_t>());

    CHECK_CUDA_ERROR("pixel_to_point_forward_kernel");
  });
  return points;
}

void pixel2points_backward(Tensor depths, torch::optional<Tensor> pixels, Tensor Ts2v, torch::optional<Tensor> Tv2w,
    Tensor& grad_points, torch::optional<Tensor>& grad_depth, torch::optional<Tensor>& grad_pixels,
    torch::optional<Tensor>& grad_Ts2v, torch::optional<Tensor>& grad_Tv2w) {
  int N = depths.size(0);
  int H = depths.size(1);
  int W = depths.size(2);
  AT_DISPATCH_FLOATING_TYPES(depths.scalar_type(), "pixel_to_point_backward_kernel", [&] {
    pixel_to_point_backward_kernel<scalar_t> KERNEL_ARG(dim3(div_round_up(H * W, 256), N), 256)(N, W, W * H,
        depths.data_ptr<scalar_t>(),
        (T2<scalar_t>*) (pixels.has_value() ? pixels.value().data_ptr<scalar_t>() : nullptr), Ts2v.data_ptr<scalar_t>(),
        Tv2w.has_value() ? Tv2w.value().data_ptr<scalar_t>() : nullptr,
        (vec3<scalar_t>*) grad_points.contiguous().data_ptr<scalar_t>(),
        grad_depth.has_value() ? grad_depth.value().data_ptr<scalar_t>() : nullptr,
        (T2<scalar_t>*) (grad_pixels.has_value() ? grad_pixels.value().data_ptr<scalar_t>() : nullptr),
        grad_Ts2v.has_value() ? grad_Ts2v.value().data_ptr<scalar_t>() : nullptr,
        grad_Tv2w.has_value() ? grad_Tv2w.value().data_ptr<scalar_t>() : nullptr);

    CHECK_CUDA_ERROR("pixel_to_point_forward_kernel");
  });
}

REGIST_PYTORCH_EXTENSION(ops_3d_pixel_points, {
  m.def("pixel2points_forward", &pixel2points_forward, "pixel2points forward (CUDA)");
  m.def("pixel2points_backward", &pixel2points_backward, "pixel2points backward (CUDA)");
});
}  // namespace OPS_3D
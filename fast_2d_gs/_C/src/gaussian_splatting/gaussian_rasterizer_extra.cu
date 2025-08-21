#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include "gaussian_render.h"
#include "ops_3d.h"
#include "util.cuh"
namespace cg = cooperative_groups;
using namespace OPS_3D;

namespace GaussianRasterizer {

template <typename T, int E_SPLIT = 16>
__global__ void __launch_bounds__(BLOCK_X* BLOCK_Y) render_extra_forward_kernel(int W, int H, int E,
    const T2<int64_t>* __restrict__ ranges, const int32_t* __restrict__ point_list,
    const T2<T>* __restrict__ points_xy_image, const T4<T>* __restrict__ conic_opacity,
    const int32_t* __restrict__ n_contrib, const T* __restrict__ point_extra, T* __restrict__ pixel_extra) {
  // Identify current tile and associated min/max pixel range.
  auto block                       = cg::this_thread_block();
  const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
  const uint2 pix_min              = {block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y};
  const uint2 pix_max              = {min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y, H)};
  const uint2 pix                  = {pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y};
  const uint32_t pix_id            = W * pix.y + pix.x;
  const T2<T> pixf                 = {(T) pix.x, (T) pix.y};
  pixel_extra += pix_id * E;

  const bool inside = pix.x < W && pix.y < H;
  const auto range  = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
  const int rounds  = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

  // Allocate storage for batches of collectively fetched data.
  __shared__ int collected_id[BLOCK_SIZE];
  __shared__ T2<T> collected_xy[BLOCK_SIZE];
  __shared__ T4<T> collected_conic_opacity[BLOCK_SIZE];

  const int last_contributor = inside ? n_contrib[pix_id] : 0;

  T temp_extra[E_SPLIT] = {0};
  // Iterate over batches until all done or range is complete
  for (int e_start = 0; e_start < E; e_start += E_SPLIT) {
    bool done            = !inside;
    int toDo             = range.y - range.x;
    T t                  = 1.0f;
    uint32_t contributor = 0;
#pragma unroll
    for (int e = 0; e < E_SPLIT; ++e) temp_extra[e] = 0;

    for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE) {
      int num_done = __syncthreads_count(done);
      if (num_done == BLOCK_SIZE) break;

      // Collectively fetch per-Gaussian data from global to shared
      int progress = i * BLOCK_SIZE + block.thread_rank();
      if (range.x + progress < range.y) {
        int coll_id                                  = point_list[range.x + progress];
        collected_id[block.thread_rank()]            = coll_id;
        collected_xy[block.thread_rank()]            = points_xy_image[coll_id];
        collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
      }
      block.sync();

      // Iterate over current batch
      for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++) {
        // Keep track of current position in range
        contributor++;
        if (contributor > last_contributor) {
          done = true;
          continue;
        }

        // Resample using conic matrix (cf. "Surface Splatting" by Zwicker et al., 2001)
        T2<T> xy    = collected_xy[j];
        T2<T> d     = {xy.x - pixf.x, xy.y - pixf.y};
        T4<T> con_o = collected_conic_opacity[j];
        T power     = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
        if (power > 0.0f) continue;

        // Eq. (2) from 3D Gaussian splatting paper.
        // Obtain alpha by multiplying with Gaussian opacity and its exponential falloff from mean.
        // Avoid numerical instabilities (see paper appendix).
        T alpha = min(0.99f, con_o.w * exp(power));
        if (alpha < 1.0f / 255.0f) continue;
        T test_T = t * (1 - alpha);
        if (test_T < 0.0001f) {
          done = true;
          continue;
        }
#pragma unroll
        for (int e = 0; e < E_SPLIT; ++e)
          temp_extra[e] += (e_start + e < E) ? (point_extra[collected_id[j] * E + e + e_start] * alpha * t) : 0;
        t = test_T;
      }
    }
    if (inside) {
#pragma unroll
      for (int e = 0; e < E_SPLIT; ++e)
        if (e + e_start < E) pixel_extra[e + e_start] = temp_extra[e];
    }
  }
}

template <typename T, int E_SPLIT = 16>
__global__ void __launch_bounds__(BLOCK_X* BLOCK_Y)
    render_extra_backward_kernel(int W, int H, int E, const T2<int64_t>* __restrict__ ranges,
        const int32_t* __restrict__ point_list, const T2<T>* __restrict__ points_xy_image,
        const T4<T>* __restrict__ conic_opacity, const T* __restrict__ out_opacity,
        const int32_t* __restrict__ n_contrib, const T* __restrict__ point_extra, const T* __restrict__ dL_dpixel_extra,
        T2<T>* __restrict__ dL_dmean2D, T4<T>* __restrict__ dL_dconic_opacity, T* __restrict__ dL_dpoint_extra) {
  // We rasterize again. Compute necessary block info.
  auto block                       = cg::this_thread_block();
  const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
  const uint2 pix_min              = {block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y};
  const uint2 pix_max              = {min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y, H)};
  const uint2 pix                  = {pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y};
  const uint32_t pix_id            = W * pix.y + pix.x;
  const T2<T> pixf                 = {(T) pix.x, (T) pix.y};

  const bool inside = pix.x < W && pix.y < H;
  const auto range  = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
  const int rounds  = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

  __shared__ int collected_id[BLOCK_SIZE];
  __shared__ T2<T> collected_xy[BLOCK_SIZE];
  __shared__ T4<T> collected_conic_opacity[BLOCK_SIZE];
  __shared__ T collected_extra[E_SPLIT * BLOCK_SIZE];

  const T T_final            = inside ? 1.0f - out_opacity[pix_id] : 0;
  const int last_contributor = inside ? n_contrib[pix_id] : 0;

  for (int es = 0; es < E; es += E_SPLIT) {
    bool done            = !inside;
    int toDo             = range.y - range.x;
    uint32_t contributor = toDo;
    T t                  = T_final;
    T accum_rec[E_SPLIT] = {0};
    T dL_dpixel[E_SPLIT];
    if (inside) {
#pragma unroll
      for (int e = 0; e < E_SPLIT; e++) dL_dpixel[e] = (e + es < E) ? dL_dpixel_extra[pix_id * E + e + es] : 0;
    }

    // Traverse all Gaussians
    for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE) {
      // Load auxiliary data into shared memory, start in the BACK and load them in reverse order.
      block.sync();
      const int progress = i * BLOCK_SIZE + block.thread_rank();
      if (range.x + progress < range.y) {
        const int coll_id                            = point_list[range.y - progress - 1];
        collected_id[block.thread_rank()]            = coll_id;
        collected_xy[block.thread_rank()]            = points_xy_image[coll_id];
        collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
#pragma unroll
        for (int e = 0; e < E_SPLIT; e++)
          collected_extra[e * BLOCK_SIZE + block.thread_rank()] = (e + es < E) ? point_extra[coll_id * E + e + es] : 0;
      }
      block.sync();

      // Iterate over Gaussians
      for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++) {
        // Keep track of current Gaussian ID. Skip, if this one is behind the last contributor for this pixel.
        contributor--;
        if (contributor >= last_contributor) continue;

        // Compute blending values, as before.
        const T2<T> xy    = collected_xy[j];
        const T2<T> d     = {xy.x - pixf.x, xy.y - pixf.y};
        const T4<T> con_o = collected_conic_opacity[j];
        const T power     = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
        if (power > 0.0f) continue;

        const T G     = exp(power);
        const T alpha = min(0.99f, con_o.w * G);
        if (alpha < 1.0f / 255.0f) continue;

        t = t / (1.f - alpha);

        const T dchannel_dextra = alpha * t;

        // Propagate gradients to per-Gaussian colors and keep gradients w.r.t. alpha
        // (blending factor for a Gaussian/pixel pair).
        T dL_dalpha         = 0.0f;
        const int global_id = collected_id[j];
        for (int e = 0; e < E_SPLIT; e++) {
          if (e + es >= E) continue;
          const T c           = collected_extra[e * BLOCK_SIZE + j];
          const T dL_dchannel = dL_dpixel[e];
          dL_dalpha += (c * t * (1.0 - alpha) - accum_rec[e]) * dL_dchannel;
          accum_rec[e] += c * dchannel_dextra;
          if (dL_dpoint_extra != nullptr)
            atomicAdd(&(dL_dpoint_extra[global_id * E + e + es]), dchannel_dextra * dL_dchannel);
        }
        dL_dalpha     = dL_dalpha / (T(1.0) - alpha);
        const T dL_dG = -con_o.w * G * dL_dalpha;
        if (dL_dmean2D != nullptr) {
          // Update gradients w.r.t. 2D mean position of the Gaussian
          atomicAdd(&dL_dmean2D[global_id].x, dL_dG * (con_o.x * d.x + con_o.y * d.y));
          atomicAdd(&dL_dmean2D[global_id].y, dL_dG * (con_o.z * d.y + con_o.y * d.x));
        }

        if (dL_dconic_opacity != nullptr) {
          // Update gradients w.r.t. 2D covariance (2x2 matrix, symmetric)
          atomicAdd(&dL_dconic_opacity[global_id].x, T(0.5) * d.x * d.x * dL_dG);
          atomicAdd(&dL_dconic_opacity[global_id].y, T(1.0) * d.x * d.y * dL_dG);
          atomicAdd(&dL_dconic_opacity[global_id].z, T(0.5) * d.y * d.y * dL_dG);
          // Update gradients w.r.t. opacity of the Gaussian
          atomicAdd(&dL_dconic_opacity[global_id].w, G * dL_dalpha);
        }
      }
    }
  }
}

template <typename T, int E_SPLIT = 16>
__global__ void __launch_bounds__(BLOCK_X* BLOCK_Y)
    render_extra_backward_kernel_v2(int W, int H, int E, const T2<int64_t>* __restrict__ ranges,
        const int32_t* __restrict__ point_list, const T2<T>* __restrict__ points_xy_image,
        const T4<T>* __restrict__ conic_opacity, const T* __restrict__ out_opacity,
        const int32_t* __restrict__ n_contrib, const T* __restrict__ point_extra, const T* __restrict__ dL_dpixel_extra,
        T2<T>* __restrict__ dL_dmean2D, T4<T>* __restrict__ dL_dconic_opacity, T* __restrict__ dL_dpoint_extra) {
  // We rasterize again. Compute necessary block info.
  auto block                       = cg::this_thread_block();
  const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
  const uint2 pix_min              = {block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y};
  const uint2 pix_max              = {min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y, H)};
  const uint2 pix                  = {pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y};
  const uint32_t pix_id            = W * pix.y + pix.x;
  const T2<T> pixf                 = {(T) pix.x, (T) pix.y};

  const bool inside = pix.x < W && pix.y < H;
  const auto range  = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
  const int rounds  = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

  __shared__ int collected_id[BLOCK_SIZE];
  __shared__ T2<T> collected_xy[BLOCK_SIZE];
  __shared__ T4<T> collected_conic_opacity[BLOCK_SIZE];
  __shared__ T collected_extra[E_SPLIT * BLOCK_SIZE];

  const T T_final            = inside ? 1.0f - out_opacity[pix_id] : 0;
  const int last_contributor = inside ? n_contrib[pix_id] : 0;

  for (int es = 0; es < E; es += E_SPLIT) {
    bool done            = !inside;
    int toDo             = range.y - range.x;
    uint32_t contributor = toDo;
    T t                  = T_final;
    T accum_rec[E_SPLIT] = {0};
    T dL_dpixel[E_SPLIT];
    if (inside) {
#pragma unroll
      for (int e = 0; e < E_SPLIT; e++) dL_dpixel[e] = (e + es < E) ? dL_dpixel_extra[pix_id * E + e + es] : 0;
    }

    // Traverse all Gaussians
    for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE) {
      // Load auxiliary data into shared memory, start in the BACK and load them in reverse order.
      block.sync();
      const int progress = i * BLOCK_SIZE + block.thread_rank();
      if (range.x + progress < range.y) {
        const int coll_id                            = point_list[range.y - progress - 1];
        collected_id[block.thread_rank()]            = coll_id;
        collected_xy[block.thread_rank()]            = points_xy_image[coll_id];
        collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
#pragma unroll
        for (int e = 0; e < E_SPLIT; e++)
          collected_extra[e * BLOCK_SIZE + block.thread_rank()] = (e + es < E) ? point_extra[coll_id * E + e + es] : 0;
      }
      block.sync();

      // Iterate over Gaussians
      for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++) {
        // Keep track of current Gaussian ID. Skip, if this one is behind the last contributor for this pixel.
        contributor--;
        if (contributor >= last_contributor) continue;

        // Compute blending values, as before.
        const T2<T> xy    = collected_xy[j];
        const T2<T> d     = {xy.x - pixf.x, xy.y - pixf.y};
        const T4<T> con_o = collected_conic_opacity[j];
        const T power     = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
        if (power > 0.0f) continue;

        const T G     = exp(power);
        const T alpha = min(0.99f, con_o.w * G);
        if (alpha < 1.0f / 255.0f) continue;

        t = t / (1.f - alpha);

        const T dchannel_dextra = alpha * t;

        // Propagate gradients to per-Gaussian colors and keep gradients w.r.t. alpha
        // (blending factor for a Gaussian/pixel pair).
        T dL_dalpha         = 0.0f;
        const int global_id = collected_id[j];
        for (int e = 0; e < E_SPLIT; e++) {
          if (e + es >= E) continue;
          const T c           = collected_extra[e * BLOCK_SIZE + j];
          const T dL_dchannel = dL_dpixel[e];
          dL_dalpha += (c * t * (1.0 - alpha) - accum_rec[e]) * dL_dchannel;
          accum_rec[e] += c * dchannel_dextra;

          atomicAdd(&(dL_dpoint_extra[global_id * E + e + es]), dchannel_dextra * dL_dchannel);
        }
        dL_dalpha     = dL_dalpha / (T(1.0) - alpha);
        const T dL_dG = -con_o.w * G * dL_dalpha;
        // Update gradients w.r.t. 2D mean position of the Gaussian
        atomicAdd(&dL_dmean2D[global_id].x, dL_dG * (con_o.x * d.x + con_o.y * d.y));
        atomicAdd(&dL_dmean2D[global_id].y, dL_dG * (con_o.z * d.y + con_o.y * d.x));

        // Update gradients w.r.t. 2D covariance (2x2 matrix, symmetric)
        atomicAdd(&dL_dconic_opacity[global_id].x, T(0.5) * d.x * d.x * dL_dG);
        atomicAdd(&dL_dconic_opacity[global_id].y, T(1.0) * d.x * d.y * dL_dG);
        atomicAdd(&dL_dconic_opacity[global_id].z, T(0.5) * d.y * d.y * dL_dG);
        // Update gradients w.r.t. opacity of the Gaussian
        atomicAdd(&dL_dconic_opacity[global_id].w, G * dL_dalpha);
      }
    }
  }
}

Tensor render_extra_forward(int width, int height, const Tensor& extras, const Tensor& means2D,
    const Tensor& conic_opacity, const Tensor& ranges, const Tensor& point_list, const Tensor& n_contrib) {
  CHECK_INPUT(extras);
  BCNN_ASSERT(extras.ndimension() == 2, "Error shape for extras");
  BCNN_ASSERT(extras.dtype() == means2D.dtype(), "extras's dtype must be same with means2D");
  int P              = extras.size(0);  // num points
  int E              = extras.size(1);  // num extras
  Tensor pixel_extra = torch::zeros({height, width, E}, extras.options());
  if (P == 0) return pixel_extra;

  const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
  const dim3 block(BLOCK_X, BLOCK_Y, 1);

  AT_DISPATCH_FLOATING_TYPES(means2D.scalar_type(), "render_extra_forward_kernel", [&] {
    render_extra_forward_kernel<scalar_t> KERNEL_ARG(tile_grid, block)(width, height, E,
        (T2<int64_t>*) ranges.data_ptr<int64_t>(), point_list.data_ptr<int32_t>(),
        (T2<scalar_t>*) means2D.data_ptr<scalar_t>(), (T4<scalar_t>*) conic_opacity.data_ptr<scalar_t>(),
        n_contrib.data_ptr<int32_t>(), extras.contiguous().data_ptr<scalar_t>(), pixel_extra.data_ptr<scalar_t>());
    CHECK_CUDA_ERROR("render_extra_forward_kernel");
  });
  return pixel_extra;
}

void render_extra_backward(int width, int height, const Tensor& extras, const Tensor& out_opacity,
    const Tensor& grad_pixel_extras, const Tensor& means2D, const Tensor& conic_opacity, const Tensor& ranges,
    const Tensor& point_list, const Tensor& n_contrib, torch::optional<Tensor>& grad_extra,
    torch::optional<Tensor>& grad_means2D, torch::optional<Tensor>& grad_conic) {
  int P = extras.size(0);  // num points
  int E = extras.size(1);  // num extras

  if (P == 0) return;

  const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
  const dim3 block(BLOCK_X, BLOCK_Y, 1);
  AT_DISPATCH_FLOATING_TYPES(extras.scalar_type(), "render_extra_backward_kernel", [&] {
    render_extra_backward_kernel KERNEL_ARG(tile_grid, block)(width, height, E,
        (T2<int64_t>*) ranges.data_ptr<int64_t>(), point_list.data_ptr<int32_t>(),
        (T2<scalar_t>*) means2D.data_ptr<scalar_t>(), (T4<scalar_t>*) conic_opacity.data_ptr<scalar_t>(),
        out_opacity.data_ptr<scalar_t>(), n_contrib.data_ptr<int32_t>(), extras.contiguous().data_ptr<scalar_t>(),
        grad_pixel_extras.contiguous().data_ptr<scalar_t>(),
        // outputs grad
        (T2<scalar_t>*) (grad_means2D.has_value() ? grad_means2D.value().data_ptr<scalar_t>() : nullptr),
        (T4<scalar_t>*) (grad_conic.has_value() ? grad_conic.value().data_ptr<scalar_t>() : nullptr),
        (grad_extra.has_value() ? grad_extra.value().data_ptr<scalar_t>() : nullptr));
    CHECK_CUDA_ERROR("render_extra_backward_kernel");
  });
}

void render_extra_backward_v2(int width, int height, const Tensor& extras, const Tensor& out_opacity,
    const Tensor& grad_pixel_extras, const Tensor& means2D, const Tensor& conic_opacity, const Tensor& ranges,
    const Tensor& point_list, const Tensor& n_contrib, torch::optional<Tensor>& grad_extra,
    torch::optional<Tensor>& grad_means2D, torch::optional<Tensor>& grad_conic) {
  int P = extras.size(0);  // num points
  int E = extras.size(1);  // num extras

  if (P == 0) return;

  const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
  const dim3 block(BLOCK_X, BLOCK_Y, 1);
  AT_DISPATCH_FLOATING_TYPES(extras.scalar_type(), "render_extra_backward_kernel", [&] {
    render_extra_backward_kernel_v2 KERNEL_ARG(tile_grid, block)(width, height, E,
        (T2<int64_t>*) ranges.data_ptr<int64_t>(), point_list.data_ptr<int32_t>(),
        (T2<scalar_t>*) means2D.data_ptr<scalar_t>(), (T4<scalar_t>*) conic_opacity.data_ptr<scalar_t>(),
        out_opacity.data_ptr<scalar_t>(), n_contrib.data_ptr<int32_t>(), extras.contiguous().data_ptr<scalar_t>(),
        grad_pixel_extras.contiguous().data_ptr<scalar_t>(),
        // outputs grad
        (T2<scalar_t>*) (grad_means2D.has_value() ? grad_means2D.value().data_ptr<scalar_t>() : nullptr),
        (T4<scalar_t>*) (grad_conic.has_value() ? grad_conic.value().data_ptr<scalar_t>() : nullptr),
        (grad_extra.has_value() ? grad_extra.value().data_ptr<scalar_t>() : nullptr));
    CHECK_CUDA_ERROR("render_extra_backward_kernel_v2");
  });
}

REGIST_PYTORCH_EXTENSION(gs_gaussian_rasterize_extra, {
  m.def("gaussian_rasterize_extra_forward", &render_extra_forward, "gaussian rasterize extra_forward (CUDA)");
  m.def("gaussian_rasterize_extra_backward", &render_extra_backward, "gaussian rasterize extra_backward (CUDA)");
  m.def("gaussian_rasterize_extra_backward_v2", &render_extra_backward_v2, "gaussian rasterize extra_backward (CUDA)");
})
}  // namespace GaussianRasterizer
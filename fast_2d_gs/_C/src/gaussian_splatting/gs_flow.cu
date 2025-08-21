#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include "gaussian_render.h"
#include "ops_3d.h"
#include "util.cuh"
namespace cg = cooperative_groups;
using namespace OPS_3D;

namespace GaussianRasterizer {

template <typename T>
__global__ void __launch_bounds__(BLOCK_X* BLOCK_Y) render_flow_forward_kernel(int W, int H,
    const T2<int64_t>* __restrict__ ranges, const int32_t* __restrict__ point_list,
    const T2<T>* __restrict__ points_xy_image, const T4<T>* __restrict__ conic_opacity,
    const int32_t* __restrict__ n_contrib, const T* __restrict__ m_flow, T* __restrict__ flow) {
  // Identify current tile and associated min/max pixel range.
  auto block                       = cg::this_thread_block();
  const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
  const uint2 pix_min              = {block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y};
  const uint2 pix_max              = {min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y, H)};
  const uint2 pix                  = {pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y};
  const uint32_t pix_id            = W * pix.y + pix.x;
  const T2<T> pixf                 = {(T) pix.x, (T) pix.y};
  flow += pix_id * 2;

  const bool inside = pix.x < W && pix.y < H;
  const auto range  = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
  const int rounds  = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

  // Allocate storage for batches of collectively fetched data.
  // __shared__ int collected_id[BLOCK_SIZE];
  __shared__ T2<T> collected_xy[BLOCK_SIZE];
  __shared__ T4<T> collected_conic_opacity[BLOCK_SIZE];
  __shared__ T cache_m[BLOCK_SIZE * 6];

  const int last_contributor = inside ? n_contrib[pix_id] : 0;

  T dx = 0, dy = 0;
  // Iterate over batches until all done or range is complete
  bool done            = !inside;
  int toDo             = range.y - range.x;
  T t                  = 1.0f;
  uint32_t contributor = 0;

  for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE) {
    int num_done = __syncthreads_count(done);
    if (num_done == BLOCK_SIZE) break;

    // Collectively fetch per-Gaussian data from global to shared
    int progress = i * BLOCK_SIZE + block.thread_rank();
    if (range.x + progress < range.y) {
      int coll_id = point_list[range.x + progress];
      // collected_id[block.thread_rank()]            = coll_id;
      collected_xy[block.thread_rank()]            = points_xy_image[coll_id];
      collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
#pragma unroll
      for (int j = 0; j < 6; ++j) cache_m[block.thread_rank() * 6 + j] = m_flow[coll_id * 6 + j];
    }
    block.sync();

    for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++) {
      contributor++;
      if (contributor > last_contributor) {
        done = true;
        continue;
      }

      T2<T> xy    = collected_xy[j];
      T2<T> d     = {xy.x - pixf.x, xy.y - pixf.y};
      T4<T> con_o = collected_conic_opacity[j];
      T power     = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
      if (power > 0.0f) continue;

      T alpha = min(0.99f, con_o.w * exp(power));
      if (alpha < 1.0f / 255.0f) continue;
      T test_T = t * (1 - alpha);
      if (test_T < 0.0001f) {
        done = true;
        continue;
      }
      dx += alpha * t * (d.x * cache_m[j * 6 + 0] + d.y * cache_m[j * 6 + 1] + cache_m[j * 6 + 2] + pixf.x);
      dy += alpha * t * (d.x * cache_m[j * 6 + 3] + d.y * cache_m[j * 6 + 4] + cache_m[j * 6 + 5] + pixf.y);
      t = test_T;
    }
  }
  if (inside) {
    flow[0] = dx;
    flow[1] = dy;
  }
}

template <typename T>
__global__ void __launch_bounds__(BLOCK_X* BLOCK_Y)
    render_flow_backward_kernel(int W, int H, const T2<int64_t>* __restrict__ ranges,
        const int32_t* __restrict__ point_list, const T2<T>* __restrict__ points_xy_image,
        const T4<T>* __restrict__ conic_opacity, const T* __restrict__ out_opacity,
        const int32_t* __restrict__ n_contrib, const T* __restrict__ m_flow, const T* __restrict__ g_flow,
        T2<T>* __restrict__ g_mean2D, T4<T>* __restrict__ g_conic_opacity, T* __restrict__ g_m_flow) {
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
  __shared__ T collected_m[6 * BLOCK_SIZE];

  const T T_final            = inside ? 1.0f - out_opacity[pix_id] : 0;
  const int last_contributor = inside ? n_contrib[pix_id] : 0;

  bool done            = !inside;
  int toDo             = range.y - range.x;
  uint32_t contributor = toDo;
  T t                  = T_final;
  T accum_rec[2]       = {0};
  const T g_flow_x     = inside ? g_flow[pix_id * 2 + 0] : 0;
  const T g_flow_y     = inside ? g_flow[pix_id * 2 + 1] : 0;

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
      for (int e = 0; e < 6; e++) collected_m[e * BLOCK_SIZE + block.thread_rank()] = m_flow[coll_id * 6 + e];
    }
    block.sync();

    // Iterate over Gaussians
    for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++) {
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

      const T sigma       = alpha * t;
      const int global_id = collected_id[j];
      // dx += sigma * (d.x * collected_m[0] + d.y * collected_m[1] + collected_m[2] + pixf.x);
      // dy += sigma * (d.x * collected_m[3] + d.y * collected_m[4] + collected_m[5] + pixf.y);
      T dx = d.x * collected_m[0 * BLOCK_SIZE + j] + d.y * collected_m[1 * BLOCK_SIZE + j] +
             collected_m[2 * BLOCK_SIZE + j] + pixf.x;
      T dy = d.x * collected_m[3 * BLOCK_SIZE + j] + d.y * collected_m[4 * BLOCK_SIZE + j] +
             collected_m[5 * BLOCK_SIZE + j] + pixf.y;

      T dL_dalpha = (dx * t * (T(1.0) - alpha) - accum_rec[0]) * g_flow_x;
      dL_dalpha += (dy * t * (T(1.0) - alpha) - accum_rec[1]) * g_flow_y;
      accum_rec[0] += dx * sigma;
      accum_rec[1] += dy * sigma;

      T gx = sigma * g_flow_x;
      T gy = sigma * g_flow_y;
      atomicAdd(g_m_flow + global_id * 6 + 0, gx * d.x);
      atomicAdd(g_m_flow + global_id * 6 + 1, gx * d.y);
      atomicAdd(g_m_flow + global_id * 6 + 2, gx);
      atomicAdd(g_m_flow + global_id * 6 + 3, gy * d.x);
      atomicAdd(g_m_flow + global_id * 6 + 4, gy * d.y);
      atomicAdd(g_m_flow + global_id * 6 + 5, gy);

      dL_dalpha     = dL_dalpha / (T(1.0) - alpha);
      const T dL_dG = -con_o.w * G * dL_dalpha;
      // Update gradients w.r.t. 2D mean position of the Gaussian
      atomicAdd(&g_mean2D[global_id].x, dL_dG * (con_o.x * d.x + con_o.y * d.y) + gx * collected_m[0 * BLOCK_SIZE + j] +
                                            gy * collected_m[3 * BLOCK_SIZE + j]);
      atomicAdd(&g_mean2D[global_id].y, dL_dG * (con_o.z * d.y + con_o.y * d.x) + gx * collected_m[1 * BLOCK_SIZE + j] +
                                            gy * collected_m[4 * BLOCK_SIZE + j]);

      // Update gradients w.r.t. 2D covariance (2x2 matrix, symmetric)
      atomicAdd(&g_conic_opacity[global_id].x, T(0.5) * d.x * d.x * dL_dG);
      atomicAdd(&g_conic_opacity[global_id].y, T(1.0) * d.x * d.y * dL_dG);
      atomicAdd(&g_conic_opacity[global_id].z, T(0.5) * d.y * d.y * dL_dG);
      // Update gradients w.r.t. opacity of the Gaussian
      atomicAdd(&g_conic_opacity[global_id].w, G * dL_dalpha);
    }
  }
}

Tensor render_flow_forward(int width, int height, const Tensor& m_flow, const Tensor& means2D,
    const Tensor& conic_opacity, const Tensor& ranges, const Tensor& point_list, const Tensor& n_contrib) {
  CHECK_INPUT(m_flow);
  BCNN_ASSERT(m_flow.dtype() == means2D.dtype(), "m_flow's dtype must be same with means2D");
  int P = means2D.size(0);  // num point
  CHECK_SHAPE(m_flow, P, 2, 3);
  Tensor flow = torch::zeros({height, width, 2}, m_flow.options());
  if (P == 0) return flow;

  const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
  const dim3 block(BLOCK_X, BLOCK_Y, 1);

  AT_DISPATCH_FLOATING_TYPES(means2D.scalar_type(), "render_extra_forward_kernel", [&] {
    render_flow_forward_kernel<scalar_t> KERNEL_ARG(tile_grid, block)(width, height,
        (T2<int64_t>*) ranges.data_ptr<int64_t>(), point_list.data_ptr<int32_t>(),
        (T2<scalar_t>*) means2D.data_ptr<scalar_t>(), (T4<scalar_t>*) conic_opacity.data_ptr<scalar_t>(),
        n_contrib.data_ptr<int32_t>(), m_flow.contiguous().data_ptr<scalar_t>(), flow.data_ptr<scalar_t>());
    CHECK_CUDA_ERROR("render_flow_forward_kernel");
  });
  return flow;
}

std::tuple<Tensor, Tensor, Tensor> render_flow_backward(int width, int height, const Tensor& m_flow,
    const Tensor& out_opacity, const Tensor& grad_flow, const Tensor& means2D, const Tensor& conic_opacity,
    const Tensor& ranges, const Tensor& point_list, const Tensor& n_contrib, torch::optional<Tensor>& grad_means2D,
    torch::optional<Tensor>& grad_conic) {
  int P = m_flow.size(0);  // num points
  CHECK_INPUT(grad_flow);

  Tensor dL_dmeans2D = grad_means2D.has_value() ? grad_means2D.value() : torch::zeros({P, 2}, m_flow.options());
  Tensor dL_dconic_o = grad_conic.has_value() ? grad_conic.value() : torch::zeros({P, 4}, m_flow.options());
  Tensor g_m_flow    = torch::zeros({P, 2, 3}, m_flow.options());
  if (P == 0) return std::make_tuple(g_m_flow, dL_dmeans2D, dL_dconic_o);

  const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
  const dim3 block(BLOCK_X, BLOCK_Y, 1);
  AT_DISPATCH_FLOATING_TYPES(m_flow.scalar_type(), "render_flow_backward_kernel", [&] {
    render_flow_backward_kernel KERNEL_ARG(tile_grid, block)(width, height, (T2<int64_t>*) ranges.data_ptr<int64_t>(),
        point_list.data_ptr<int32_t>(), (T2<scalar_t>*) means2D.data_ptr<scalar_t>(),
        (T4<scalar_t>*) conic_opacity.data_ptr<scalar_t>(), out_opacity.data_ptr<scalar_t>(),
        n_contrib.data_ptr<int32_t>(), m_flow.contiguous().data_ptr<scalar_t>(),
        grad_flow.contiguous().data_ptr<scalar_t>(), (T2<scalar_t>*) dL_dmeans2D.data_ptr<scalar_t>(),
        (T4<scalar_t>*) dL_dconic_o.data_ptr<scalar_t>(), g_m_flow.data_ptr<scalar_t>());
    CHECK_CUDA_ERROR("render_flow_backward_kernel");
  });
  return std::make_tuple(g_m_flow, dL_dmeans2D, dL_dconic_o);
}
REGIST_PYTORCH_EXTENSION(gs_gaussian_flow, {
  m.def("gs_flow_forward", &render_flow_forward, "gaussian rasterize flow_forward (CUDA)");
  m.def("gs_flow_backward", &render_flow_backward, "gaussian rasterize flow_backward (CUDA)");
})
}  // namespace GaussianRasterizer
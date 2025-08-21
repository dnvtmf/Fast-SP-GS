#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include "gaussian_render.h"
#include "ops_3d.h"
#include "util.cuh"

namespace cg = cooperative_groups;
namespace GaussianRasterizer {
using namespace OPS_3D;

template <typename T>
__global__ void __launch_bounds__(BLOCK_X* BLOCK_Y)
    render_topk_weights(int W, int H, const int topk, const T2<int64_t>* __restrict__ ranges,
        const int32_t* __restrict__ point_list, const T2<T>* __restrict__ points_xy_image,
        const T4<T>* __restrict__ conic_opacity, int32_t* __restrict__ top_indices, T* __restrict__ top_weights) {
  // Identify current tile and associated min/max pixel range.
  auto block                       = cg::this_thread_block();
  const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
  const uint2 pix_min              = {block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y};
  const uint2 pix_max              = {min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y, H)};
  const uint2 pix                  = {pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y};
  const uint32_t pix_id            = W * pix.y + pix.x;
  const T2<T> pixf                 = {(float) pix.x, (float) pix.y};

  top_weights += pix_id * topk;
  top_indices += pix_id * topk;

  const bool inside = pix.x < W && pix.y < H;
  auto range        = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

  const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

  bool done = !inside;
  int toDo  = range.y - range.x;

  // Allocate storage for batches of collectively fetched data.
  __shared__ int collected_id[BLOCK_SIZE];
  __shared__ T2<T> collected_xy[BLOCK_SIZE];
  __shared__ T4<T> collected_conic_opacity[BLOCK_SIZE];

  // Initialize helper variables
  T t = 1.0f;

  // Iterate over batches until all done or range is complete
  for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE) {
    block.sync();
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

      T w     = alpha * t;
      int idx = collected_id[j];
      for (int k = 0; k < topk; ++k) {
        if (w >= top_weights[k]) {
          auto tw        = top_weights[k];
          top_weights[k] = w;
          w              = tw;
          auto t_idx     = top_indices[k];
          top_indices[k] = idx;
          idx            = t_idx;
        }
      }
      t = test_T;
    }
  }
}

template <typename T>
__global__ void __launch_bounds__(BLOCK_X* BLOCK_Y) gs_2d_topk_kernel(
    // scalar
    int W, int H, int topk, T near_n, T far_n,
    // inputs
    const T2<int64_t>* __restrict__ ranges, const int32_t* __restrict__ point_list,
    const T2<T>* __restrict__ points_xy_image, const T* __restrict__ transMats,
    const T4<T>* __restrict__ normal_opacity,
    // outputs
    int32_t* __restrict__ top_indices, T* __restrict__ top_weights) {
  auto block                 = cg::this_thread_block();
  uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
  uint2 pix_min              = {block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y};
  uint2 pix_max              = {min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y, H)};
  uint2 pix                  = {pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y};
  uint32_t pix_id            = W * pix.y + pix.x;
  T2<T> pixf                 = {(T) pix.x, (T) pix.y};

  top_weights += pix_id * topk;
  top_indices += pix_id * topk;
  // Check if this thread is associated with a valid pixel or outside.
  bool inside = pix.x < W && pix.y < H;
  // Done threads can help with fetching, but don't rasterize
  bool done = !inside;

  // Load start/end range of IDs to process in bit sorted list.
  T2<int64_t> range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
  const int rounds  = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
  int toDo          = range.y - range.x;

  // Allocate storage for batches of collectively fetched data.
  __shared__ int collected_id[BLOCK_SIZE];
  __shared__ T2<T> collected_xy[BLOCK_SIZE];
  __shared__ T4<T> collected_normal_opacity[BLOCK_SIZE];
  __shared__ vec3<T> collected_Tu[BLOCK_SIZE];
  __shared__ vec3<T> collected_Tv[BLOCK_SIZE];
  __shared__ vec3<T> collected_Tw[BLOCK_SIZE];

  // Initialize helper variables
  float t = T(1.0);

  T weight_max          = 0;
  int32_t weight_max_id = -1;
  bool flag_update      = false;

  // Iterate over batches until all done or range is complete
  for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE) {
    // End if entire block votes that it is done rasterizing
    int num_done = __syncthreads_count(done);
    if (num_done == BLOCK_SIZE) break;

    // Collectively fetch per-Gaussian data from global to shared
    int progress = i * BLOCK_SIZE + block.thread_rank();
    if (range.x + progress < range.y) {
      int coll_id                                   = point_list[range.x + progress];
      collected_id[block.thread_rank()]             = coll_id;
      collected_xy[block.thread_rank()]             = points_xy_image[coll_id];
      collected_normal_opacity[block.thread_rank()] = normal_opacity[coll_id];
      collected_Tu[block.thread_rank()]             = {
          transMats[9 * coll_id + 0], transMats[9 * coll_id + 1], transMats[9 * coll_id + 2]};
      collected_Tv[block.thread_rank()] = {
          transMats[9 * coll_id + 3], transMats[9 * coll_id + 4], transMats[9 * coll_id + 5]};
      collected_Tw[block.thread_rank()] = {
          transMats[9 * coll_id + 6], transMats[9 * coll_id + 7], transMats[9 * coll_id + 8]};
    }
    block.sync();

    // Iterate over current batch
    for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++) {
      // Fisrt compute two homogeneous planes, See Eq. (8)
      const T2<T> xy   = collected_xy[j];
      const vec3<T> Tu = collected_Tu[j];
      const vec3<T> Tv = collected_Tv[j];
      const vec3<T> Tw = collected_Tw[j];
      vec3<T> k        = pixf.x * Tw - Tu;
      vec3<T> l        = pixf.y * Tw - Tv;
      vec3<T> p        = k.cross(l);
      if (p.z == 0.0) continue;
      T2<T> s = {p.x / p.z, p.y / p.z};
      T rho3d = (s.x * s.x + s.y * s.y);
      T2<T> d = {xy.x - pixf.x, xy.y - pixf.y};
      T rho2d = T(2.0) * (d.x * d.x + d.y * d.y);

      // compute intersection and depth
      T rho   = min(rho3d, rho2d);
      T depth = (rho3d <= rho2d) ? (s.x * Tw.x + s.y * Tw.y) + Tw.z : Tw.z;
      if (depth < near_n) continue;
      T4<T> nor_o = collected_normal_opacity[j];
      T opa       = nor_o.w;

      T power = -T(0.5) * rho;
      if (power > T(0.0)) continue;
      T alpha = min(T(0.99), opa * exp(power));
      if (alpha < T(1.0 / 255.0)) continue;
      T test_T = t * (1 - alpha);
      if (test_T < T(0.0001)) {
        done = true;
        continue;
      }

      T w     = alpha * t;
      int idx = collected_id[j];
      for (int k = 0; k < topk; ++k) {
        if (w >= top_weights[k]) {
          auto tw        = top_weights[k];
          top_weights[k] = w;
          w              = tw;
          auto t_idx     = top_indices[k];
          top_indices[k] = idx;
          idx            = t_idx;
        }
      }
      t = test_T;
    }
  }
}

std::tuple<Tensor, Tensor> gaussian_topk_weights(int topk, int W, int H, int P, const int R, const Tensor& means2D,
    const Tensor& conic_opacity, const Tensor& ranges, const Tensor& point_list) {
  Tensor top_indices = torch::full({H, W, topk}, -1, means2D.options().dtype(torch::kInt32));
  Tensor top_weights = torch::full({H, W, topk}, 0, means2D.options());
  if (P == 0) return std::make_tuple(top_indices, top_weights);

  const dim3 tile_grid((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, 1);
  const dim3 block(BLOCK_X, BLOCK_Y, 1);
  AT_DISPATCH_FLOATING_TYPES(means2D.scalar_type(), "render_topk_weights", [&] {
    render_topk_weights<scalar_t> KERNEL_ARG(tile_grid, block)(W, H, topk, (T2<int64_t>*) ranges.data_ptr<int64_t>(),
        point_list.data_ptr<int32_t>(), (T2<scalar_t>*) means2D.data_ptr<scalar_t>(),
        (T4<scalar_t>*) conic_opacity.data_ptr<scalar_t>(), top_indices.data<int>(), top_weights.data<scalar_t>());
    CHECK_CUDA_ERROR("render_topk_weights");
  });

  return std::make_tuple(top_indices, top_weights);
}

std::tuple<Tensor, Tensor> gs_2d_topk_weights(int topk, int W, int H, double near, double far, const Tensor& means2D,
    const Tensor& normal_opacity, Tensor& trans_mat, const Tensor& ranges, const Tensor& point_list) {
  Tensor top_indices = torch::full({H, W, topk}, -1, means2D.options().dtype(torch::kInt32));
  Tensor top_weights = torch::full({H, W, topk}, 0, means2D.options());
  int P              = means2D.size(0);
  if (P == 0) return std::make_tuple(top_indices, top_weights);

  const dim3 tile_grid((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, 1);
  const dim3 block(BLOCK_X, BLOCK_Y, 1);
  using scalar_t = float;
  // AT_DISPATCH_FLOATING_TYPES(means2D.scalar_type(), "gs_2d_topk_kernel", [&] {
  gs_2d_topk_kernel<scalar_t> KERNEL_ARG(tile_grid, block)(W, H, topk, near, far,
      (T2<int64_t>*) ranges.data_ptr<int64_t>(), point_list.data_ptr<int32_t>(),
      (T2<scalar_t>*) means2D.data_ptr<scalar_t>(), trans_mat.data_ptr<scalar_t>(),
      (T4<scalar_t>*) normal_opacity.data_ptr<scalar_t>(), top_indices.data<int>(), top_weights.data<scalar_t>());

  // CHECK_CUDA_ERROR("gs_2d_topk_kernel");
  // });

  return std::make_tuple(top_indices, top_weights);
}

REGIST_PYTORCH_EXTENSION(gs_gaussian_topk, {
  m.def("gaussian_topk_weights", &gaussian_topk_weights);
  m.def("gs_2d_topk_weights", &gs_2d_topk_weights);
})
}  // namespace GaussianRasterizer
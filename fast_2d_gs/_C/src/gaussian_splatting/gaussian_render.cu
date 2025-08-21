/*
paper: 3D Gaussian Splatting for Real-Time Radiance Field Rendering, SIGGRAPH 2023
code:  https://github.com/graphdeco-inria/diff-gaussian-rasterization
*/
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include "gaussian_render.h"
#include "ops_3d.h"
#include "util.cuh"

namespace cg = cooperative_groups;
using namespace OPS_3D;
namespace GaussianRasterizer {

constexpr bool IS_CHANNEL_FIRST = true;

// Main rasterization method. Collaboratively works on one tile per block, each thread treats one pixel.
// Alternates between fetching and rasterizing data.
template <typename T, uint32_t CHANNELS, uint32_t NUM_EXTRA>
__global__ void __launch_bounds__(BLOCK_X* BLOCK_Y) render_forward_kernel(const T2<int64_t>* __restrict__ ranges,
    const int32_t* __restrict__ point_list, int W, int H, const T2<T>* __restrict__ points_xy_image,
    const T* __restrict__ features, const T4<T>* __restrict__ conic_opacity, const T* extra,
    int32_t* __restrict__ n_contrib, /*const float* __restrict__ bg_color,*/
    T* __restrict__ out_color, T* __restrict__ out_opacity, T* __restrict__ out_extra,
    int32_t* __restrict__ accum_max_count, T* __restrict__ accum_weights_p, int32_t* __restrict__ accum_weights_count,
    // for per_gaussian_backward
    const int64_t* __restrict__ per_tile_bucket_offset, int32_t* __restrict__ bucket_to_tile, T* __restrict__ sampled_T,
    T* __restrict__ sampled_ar, int32_t* __restrict__ max_contrib) {
  // Identify current tile and associated min/max pixel range.
  auto block                 = cg::this_thread_block();
  uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
  uint2 pix_min              = {block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y};
  uint2 pix_max              = {min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y, H)};
  uint2 pix                  = {pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y};
  uint32_t pix_id            = W * pix.y + pix.x;
  T2<T> pixf                 = {(T) pix.x, (T) pix.y};

  // Check if this thread is associated with a valid pixel or outside.
  bool inside = pix.x < W && pix.y < H;
  // Done threads can help with fetching, but don't rasterize
  bool done = !inside;

  // Load start/end range of IDs to process in bit sorted list.
  uint32_t tile_id = block.group_index().y * horizontal_blocks + block.group_index().x;
  auto range       = ranges[tile_id];
  const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
  int toDo         = range.y - range.x;

  // what is the number of buckets before me? what is my offset?
  uint32_t bbm;
  // let's first quickly also write the bucket-to-tile mapping
  int num_buckets = (toDo + 31) / 32;
  if (bucket_to_tile != nullptr) {
    bbm = tile_id == 0 ? 0 : per_tile_bucket_offset[tile_id - 1];
    for (int i = 0; i < (num_buckets + BLOCK_SIZE - 1) / BLOCK_SIZE; ++i) {
      int bucket_idx = i * BLOCK_SIZE + block.thread_rank();
      if (bucket_idx < num_buckets) {
        bucket_to_tile[bbm + bucket_idx] = tile_id;
      }
    }
  }

  // Allocate storage for batches of collectively fetched data.
  __shared__ int collected_id[BLOCK_SIZE];
  __shared__ T2<T> collected_xy[BLOCK_SIZE];
  __shared__ T4<T> collected_conic_opacity[BLOCK_SIZE];

  // Initialize helper variables
  T t                       = T(1.0);
  int32_t contributor       = 0;
  int32_t last_contributor  = 0;
  T C[CHANNELS + NUM_EXTRA] = {0};
  T* E                      = C + CHANNELS;

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
      int coll_id                                  = point_list[range.x + progress];
      collected_id[block.thread_rank()]            = coll_id;
      collected_xy[block.thread_rank()]            = points_xy_image[coll_id];
      collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
    }
    block.sync();

    // Iterate over current batch
    for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++) {
      // add incoming T value for every 32nd gaussian
      if (j % 32 == 0 && sampled_T != nullptr) {
        sampled_T[(bbm * BLOCK_SIZE) + block.thread_rank()] = t;
        for (int ch = 0; ch < CHANNELS + NUM_EXTRA; ++ch) {
          sampled_ar[(bbm * (CHANNELS + NUM_EXTRA) + ch) * BLOCK_SIZE + block.thread_rank()] = C[ch];
        }
        ++bbm;
      }

      // Keep track of current position in range
      contributor++;

      // Resample using conic matrix (cf. "Surface Splatting" by Zwicker et al., 2001)
      T2<T> xy    = collected_xy[j];
      T2<T> d     = {xy.x - pixf.x, xy.y - pixf.y};
      T4<T> con_o = collected_conic_opacity[j];
      T power     = T(-0.5) * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
      if (power > T(0.0)) continue;

      // Eq. (2) from 3D Gaussian splatting paper.
      // Obtain alpha by multiplying with Gaussian opacity and its exponential falloff from mean.
      // Avoid numerical instabilities (see paper appendix).
      T alpha = min(T(0.99), con_o.w * exp(power));
      if (alpha < T(1.0 / 255.0)) continue;
      T test_T = t * (1 - alpha);
      if (test_T < T(0.0001)) {
        done = true;
        continue;
      }
      // if (pix.y == 126 && pix.x == 74) {
      //   printf("\033[33mgs=%d, xy=%.6f, %.6f, power=%.6f, alpha=%.6f, sigma=%.6f\n\033[0m", collected_id[j], xy.x,
      //   xy.y,
      //       power, alpha, test_T);
      // }

      // Eq. (3) from 3D Gaussian splatting paper.
      T w = alpha * t;
      for (int ch = 0; ch < CHANNELS; ch++) C[ch] += features[collected_id[j] * CHANNELS + ch] * w;
      if constexpr (NUM_EXTRA > 0)
        for (int ch = 0; ch < NUM_EXTRA; ch++) E[ch] += extra[collected_id[j] * NUM_EXTRA + ch] * w;
      if (weight_max < w) {
        weight_max    = w;
        weight_max_id = collected_id[j];
        flag_update   = true;
      }
      if (accum_weights_p != nullptr) atomicAdd(&(accum_weights_p[collected_id[j]]), w);
      if (accum_weights_count != nullptr) atomicAdd(&(accum_weights_count[collected_id[j]]), 1);
      t = test_T;

      // Keep track of last range entry to update this pixel.
      last_contributor = contributor;
    }
  }
  if (flag_update && accum_max_count != nullptr) atomicAdd(&(accum_max_count[weight_max_id]), 1);

  // All threads that treat valid pixel write out their final rendering data to the frame and auxiliary buffers.
  if (inside) {
    out_opacity[pix_id] = 1.f - t;
    n_contrib[pix_id]   = last_contributor;
    if constexpr (IS_CHANNEL_FIRST) {
      for (int ch = 0; ch < CHANNELS; ch++) out_color[ch * H * W + pix_id] = C[ch];  // + T * bg_color[ch];
      if constexpr (NUM_EXTRA > 0)
        for (int ch = 0; ch < NUM_EXTRA; ch++) out_extra[ch * H * W + pix_id] = E[ch];
    } else {
      for (int ch = 0; ch < CHANNELS; ch++) out_color[pix_id * CHANNELS + ch] = C[ch];  // + T * bg_color[ch];
      if constexpr (NUM_EXTRA > 0)
        for (int ch = 0; ch < NUM_EXTRA; ch++) out_extra[pix_id * NUM_EXTRA + ch] = E[ch];
    }
  }

  // max reduce the last contributor
  if (max_contrib != nullptr) {
    reduce_max_block<int32_t, false>(last_contributor);
    if (block.thread_rank() == 0) max_contrib[tile_id] = last_contributor;
  }
}

template <typename T>
void render_forward(const dim3 grid, dim3 block, const T2<int64_t>* ranges, const int32_t* point_list, int W, int H,
    int E, const T2<T>* means2D, const T* colors, const T4<T>* conic_opacity, const T* extra, int32_t* n_contrib,
    T* out_color, T* out_opacity, T* out_extra, int32_t* accum_max_count, T* accum_weights_p,
    int32_t* accum_weights_count, const int64_t* per_tile_bucket_offset, int32_t* bucket_to_tile, T* sampled_T,
    T* sampled_ar, int32_t* max_contrib) {
  switch (E) {
    case 0:
      render_forward_kernel<T, NUM_CHANNELS, 0> KERNEL_ARG(grid, block)(ranges, point_list, W, H, means2D, colors,
          conic_opacity, extra, n_contrib, out_color, out_opacity, out_extra, accum_max_count, accum_weights_p,
          accum_weights_count, per_tile_bucket_offset, bucket_to_tile, sampled_T, sampled_ar, max_contrib);
      break;
    case 1:
      render_forward_kernel<T, NUM_CHANNELS, 1> KERNEL_ARG(grid, block)(ranges, point_list, W, H, means2D, colors,
          conic_opacity, extra, n_contrib, out_color, out_opacity, out_extra, accum_max_count, accum_weights_p,
          accum_weights_count, per_tile_bucket_offset, bucket_to_tile, sampled_T, sampled_ar, max_contrib);
      break;
    case 2:
      render_forward_kernel<T, NUM_CHANNELS, 2> KERNEL_ARG(grid, block)(ranges, point_list, W, H, means2D, colors,
          conic_opacity, extra, n_contrib, out_color, out_opacity, out_extra, accum_max_count, accum_weights_p,
          accum_weights_count, per_tile_bucket_offset, bucket_to_tile, sampled_T, sampled_ar, max_contrib);
      break;
    case 3:
      render_forward_kernel<T, NUM_CHANNELS, 3> KERNEL_ARG(grid, block)(ranges, point_list, W, H, means2D, colors,
          conic_opacity, extra, n_contrib, out_color, out_opacity, out_extra, accum_max_count, accum_weights_p,
          accum_weights_count, per_tile_bucket_offset, bucket_to_tile, sampled_T, sampled_ar, max_contrib);
      break;
    case 4:
      render_forward_kernel<T, NUM_CHANNELS, 4> KERNEL_ARG(grid, block)(ranges, point_list, W, H, means2D, colors,
          conic_opacity, extra, n_contrib, out_color, out_opacity, out_extra, accum_max_count, accum_weights_p,
          accum_weights_count, per_tile_bucket_offset, bucket_to_tile, sampled_T, sampled_ar, max_contrib);
      break;
    default: BCNN_ASSERT(false, "Only Support 0,1,2,3,4 extra features"); break;
  }
  CHECK_CUDA_ERROR("render_forward");
}

vector<Tensor> gaussian_rasterize_forward(int width, int height, Tensor& means2D, Tensor& conic_opacity,
    const Tensor& colors, const torch::optional<Tensor>& extras, const Tensor& point_list, const Tensor& ranges,
    torch::optional<Tensor>& accum_max_count, torch::optional<Tensor>& accum_weights_p,
    torch::optional<Tensor>& accum_weights_count, torch::optional<Tensor>& per_tile_bucket_offset) {
  CHECK_INPUT(colors);
  CHECK_NDIM(colors, 2);
  int P = colors.size(0);  // num points
  int E = 0;               // num extras
  if (extras.has_value()) {
    CHECK_NDIM(extras.value(), 2);
    E = extras.value().size(1);  // num extras
    CHECK_INPUT(extras.value());
    CHECK_SHAPE(extras.value(), P, E);
  }
  if (accum_max_count.has_value()) {
    CHECK_INPUT_AND_TYPE(accum_max_count.value(), torch::kInt32);
    CHECK_SHAPE(accum_max_count.value(), P);
  }
  if (accum_weights_p.has_value()) {
    CHECK_INPUT(accum_weights_p.value());
    CHECK_SHAPE(accum_weights_p.value(), P);
  }
  if (accum_weights_count.has_value()) {
    CHECK_INPUT_AND_TYPE(accum_weights_count.value(), torch::kInt32);
    CHECK_SHAPE(accum_weights_count.value(), P);
  }
  Tensor pixel_colors;
  Tensor pixel_opacity = torch::zeros({height, width}, colors.options());
  Tensor n_contrib     = torch::zeros({height, width}, colors.options().dtype(torch::kInt32));
  Tensor pixel_extras;
  if constexpr (IS_CHANNEL_FIRST) {
    pixel_colors = torch::zeros({3, height, width}, colors.options());
    if (extras.has_value()) pixel_extras = torch::zeros({E, height, width}, colors.options());
  } else {
    pixel_colors = torch::zeros({height, width, 3}, colors.options());
    if (extras.has_value()) pixel_extras = torch::zeros({height, width, E}, colors.options());
  }
  vector<Tensor> outputs = {pixel_colors, pixel_opacity, pixel_extras, n_contrib};

  bool for_fast_bwd = per_tile_bucket_offset.has_value();
  Tensor max_contrib, bucket_to_tile, sampled_T, sampled_ar;
  if (for_fast_bwd) {
    CHECK_INPUT_AND_TYPE(per_tile_bucket_offset.value(), torch::kInt64);
    int B       = per_tile_bucket_offset.value()[-1].item<int64_t>();
    max_contrib = torch::zeros(
        {(height + BLOCK_Y - 1) / BLOCK_Y, (width + BLOCK_X - 1) / BLOCK_X}, colors.options().dtype(torch::kInt32));
    bucket_to_tile = torch::zeros({B}, colors.options().dtype(torch::kInt32));
    sampled_T      = torch::zeros({B, BLOCK_SIZE}, colors.options());
    sampled_ar     = torch::zeros({NUM_CHANNELS + E, B, BLOCK_SIZE}, colors.options());
    outputs.push_back(max_contrib);
    outputs.push_back(bucket_to_tile);
    outputs.push_back(sampled_T);
    outputs.push_back(sampled_ar);
  }

  if (P == 0) return outputs;

  const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
  const dim3 block(BLOCK_X, BLOCK_Y, 1);

  // AT_DISPATCH_FLOATING_TYPES(means2D.scalar_type(), "gaussian_rasterize_forward", [&] {
  using scalar_t = float;
  render_forward<scalar_t>(tile_grid, block, (T2<int64_t>*) ranges.contiguous().data_ptr<int64_t>(),
      point_list.contiguous().data_ptr<int32_t>(), width, height, E,
      (T2<scalar_t>*) means2D.contiguous().data_ptr<scalar_t>(), colors.contiguous().data_ptr<scalar_t>(),
      (T4<scalar_t>*) conic_opacity.contiguous().data_ptr<scalar_t>(),
      extras.has_value() ? extras.value().contiguous().data_ptr<scalar_t>() : nullptr, n_contrib.data_ptr<int32_t>(),
      pixel_colors.data_ptr<scalar_t>(), pixel_opacity.data_ptr<scalar_t>(),
      extras.has_value() ? pixel_extras.data_ptr<scalar_t>() : nullptr,                             //
      accum_max_count.has_value() ? accum_max_count.value().data_ptr<int32_t>() : nullptr,          //
      accum_weights_p.has_value() ? accum_weights_p.value().data_ptr<scalar_t>() : nullptr,         //
      accum_weights_count.has_value() ? accum_weights_count.value().data_ptr<int32_t>() : nullptr,  //
      for_fast_bwd ? per_tile_bucket_offset.value().data_ptr<int64_t>() : nullptr,                  //
      for_fast_bwd ? bucket_to_tile.data_ptr<int32_t>() : nullptr,                                  //
      for_fast_bwd ? sampled_T.data_ptr<scalar_t>() : nullptr,                                      //
      for_fast_bwd ? sampled_ar.data_ptr<scalar_t>() : nullptr,                                     //
      for_fast_bwd ? max_contrib.data_ptr<int32_t>() : nullptr                                      //
  );
  CHECK_CUDA_ERROR("render_forward");
  // });
  return outputs;
}

// Backward version of the rendering procedure.
template <typename T, uint32_t C, uint32_t E = 0>
__global__ void __launch_bounds__(BLOCK_X* BLOCK_Y)
    render_backward_kernel(int P, int W, int H, int64_t R, const T2<int64_t>* __restrict__ ranges,
        const int32_t* __restrict__ point_list, /*const float* __restrict__ bg_color,*/
        const T2<T>* __restrict__ points_xy_image, const T4<T>* __restrict__ conic_opacity,
        const T* __restrict__ colors, const T* __restrict__ extras, const T* __restrict__ out_opacity,
        const int32_t* __restrict__ n_contrib, const T* __restrict__ dL_dpixels, const T* __restrict__ dL_dout_extra,
        const T* __restrict__ dL_dout_opacity, T2<T>* __restrict__ dL_dmean2D, T4<T>* __restrict__ dL_dconic_opacity,
        T* __restrict__ dL_dcolors, T* __restrict__ dL_dextras) {
  // We rasterize again. Compute necessary block info.
  auto block            = cg::this_thread_block();
  const uint2 pix       = {blockIdx.x * BLOCK_X + threadIdx.x, blockIdx.y * BLOCK_Y + threadIdx.y};
  const uint32_t pix_id = W * pix.y + pix.x;  // avoid over-bound
  const T2<T> pixf      = {(T) pix.x, (T) pix.y};
  const bool inside     = pix.x < W && pix.y < H;

  auto range       = ranges[blockIdx.y * gridDim.x + blockIdx.x];
  const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

  bool done = !inside;
  int toDo  = range.y - range.x;

  __shared__ int collected_id[BLOCK_SIZE];
  __shared__ T2<T> collected_xy[BLOCK_SIZE];
  __shared__ T4<T> collected_conic_opacity[BLOCK_SIZE];
  __shared__ T collected_colors[(C + E) * BLOCK_SIZE];
  T* collected_extras = collected_colors + C * BLOCK_SIZE;

  // In the forward, we stored the final value for T, the product of all (1 - alpha) factors.
  const T T_final = inside ? T(1.0) - out_opacity[pix_id] : 0;
  T t             = T_final;
  const T dL_dT   = inside ? -dL_dout_opacity[pix_id] : 0;

  // We start from the back. The ID of the last contributing Gaussian is known from each pixel from the forward.
  int32_t contributor        = toDo;
  const int last_contributor = inside ? n_contrib[pix_id] : 0;

  T accum_rec[C + E] = {0};
  T dL_dpixel[C + E];
  T* accum_ext = accum_rec + C;
  T* dL_doute  = dL_dpixel + C;

  if (inside) {
    if constexpr (IS_CHANNEL_FIRST) {
#pragma unroll
      for (int c = 0; c < C; c++) dL_dpixel[c] = dL_dpixels[c * H * W + pix_id];
#pragma unroll
      for (int e = 0; e < E; e++) dL_doute[e] = dL_dout_extra[e * H * W + pix_id];
    } else {
#pragma unroll
      for (int c = 0; c < C; c++) dL_dpixel[c] = dL_dpixels[pix_id * C + c];
#pragma unroll
      for (int e = 0; e < E; e++) dL_doute[e] = dL_dout_extra[pix_id * E + e];
    }
  }

  // T last_data[C + E] = {0}, last_alpha = 0;

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
      for (int c = 0; c < C; c++) collected_colors[c * BLOCK_SIZE + block.thread_rank()] = colors[coll_id * C + c];
#pragma unroll
      for (int e = 0; e < E; e++) collected_extras[e * BLOCK_SIZE + block.thread_rank()] = extras[coll_id * E + e];
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
      const T power     = T(-0.5) * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
      if (power > T(0.0)) continue;

      const T G     = exp(power);
      const T alpha = min(T(0.99), con_o.w * G);
      if (alpha < T(1.0 / 255.0)) continue;

      t = t / (T(1.0) - alpha);

      const T dchannel_dcolor = alpha * t;

      // Propagate gradients to per-Gaussian colors and keep gradients w.r.t. alpha
      // (blending factor for a Gaussian/pixel pair).
      T dL_dalpha         = T(0.0);
      const int global_id = collected_id[j];
#pragma unroll
      for (int ch = 0; ch < C; ch++) {
        const T c           = collected_colors[ch * BLOCK_SIZE + j];
        const T dL_dchannel = dL_dpixel[ch];
        dL_dalpha += (c * t * (T(1.0) - alpha) - accum_rec[ch]) * dL_dchannel;
        accum_rec[ch] += c * dchannel_dcolor;
        // accum_rec[ch] = last_alpha * last_data[ch] + (T(1.0) - last_alpha) * accum_rec[ch];
        // last_data[ch] = c;
        // dL_dalpha += (c - accum_rec[ch]) * dL_dchannel;

        // Update the gradients w.r.t. color of the Gaussian.
        // Atomic, since this pixel is just one of potentially many that were affected by this Gaussian.
        atomicAdd(&(dL_dcolors[global_id * C + ch]), dchannel_dcolor * dL_dchannel);
      }
#pragma unroll
      for (int ch = 0; ch < E; ch++) {
        const T c           = collected_extras[ch * BLOCK_SIZE + j];
        const T dL_dchannel = dL_doute[ch];
        dL_dalpha += (c * t * (T(1.0) - alpha) - accum_ext[ch]) * dL_dchannel;
        accum_ext[ch] += c * dchannel_dcolor;

        // accum_ext[ch]     = last_alpha * last_data[ch + C] + (T(1.0) - last_alpha) * accum_ext[ch];
        // last_data[ch + C] = c;
        // dL_dalpha += (c - accum_ext[ch]) * dL_dchannel;

        // Update the gradients w.r.t. color of the Gaussian.
        // Atomic, since this pixel is just one of potentially many that were affected by this Gaussian.
        atomicAdd(&(dL_dextras[global_id * E + ch]), dchannel_dcolor * dL_dchannel);
      }
      // Account for fact that alpha also influences how much of the background color is added if nothing left to blend
      dL_dalpha = (dL_dalpha - T_final * dL_dT) / (T(1.0) - alpha);
      // dL_dalpha = dL_dalpha * t - T_final * dL_dT / (T(1.0) - alpha);
      // last_alpha = alpha;

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
      // TODO: 优化 一个tile对应的point list是一致的， 使用reduce_sum减少atomicAdd数量
    }
  }
}

template <typename T>
void render_backward(int P, int W, int H, int E, int64_t R, const dim3 grid, const dim3 block,
    // inputs
    const T2<T>* means2D, const T4<T>* conic_opacity, const T* colors, const T* extras,
    // aux inputs
    const T2<int64_t>* ranges, const int32_t* point_list, const int32_t* n_contrib,
    // outputs & grad outputs
    const T* out_opacity, const T* dL_dpixels, const T* dL_dout_extras, const T* dL_dout_opacity,
    // grad inputs
    T2<T>* dL_dmean2D, T4<T>* dL_dconic_opacity, T* dL_dcolors, T* dL_dextras) {
  switch (E) {
    case 0:
      render_backward_kernel<T, NUM_CHANNELS, 0> KERNEL_ARG(grid, block)(P, W, H, R, ranges, point_list, means2D,
          conic_opacity, colors, extras, out_opacity, n_contrib, dL_dpixels, dL_dout_extras, dL_dout_opacity,
          dL_dmean2D, dL_dconic_opacity, dL_dcolors, dL_dextras);
      break;
    case 1:
      render_backward_kernel<T, NUM_CHANNELS, 1> KERNEL_ARG(grid, block)(P, W, H, R, ranges, point_list, means2D,
          conic_opacity, colors, extras, out_opacity, n_contrib, dL_dpixels, dL_dout_extras, dL_dout_opacity,
          dL_dmean2D, dL_dconic_opacity, dL_dcolors, dL_dextras);
      break;
    case 2:
      render_backward_kernel<T, NUM_CHANNELS, 2> KERNEL_ARG(grid, block)(P, W, H, R, ranges, point_list, means2D,
          conic_opacity, colors, extras, out_opacity, n_contrib, dL_dpixels, dL_dout_extras, dL_dout_opacity,
          dL_dmean2D, dL_dconic_opacity, dL_dcolors, dL_dextras);
      break;
    case 3:
      render_backward_kernel<T, NUM_CHANNELS, 3> KERNEL_ARG(grid, block)(P, W, H, R, ranges, point_list, means2D,
          conic_opacity, colors, extras, out_opacity, n_contrib, dL_dpixels, dL_dout_extras, dL_dout_opacity,
          dL_dmean2D, dL_dconic_opacity, dL_dcolors, dL_dextras);
      break;
    case 4:
      render_backward_kernel<T, NUM_CHANNELS, 4> KERNEL_ARG(grid, block)(P, W, H, R, ranges, point_list, means2D,
          conic_opacity, colors, extras, out_opacity, n_contrib, dL_dpixels, dL_dout_extras, dL_dout_opacity,
          dL_dmean2D, dL_dconic_opacity, dL_dcolors, dL_dextras);
      break;
    default: BCNN_ASSERT(false, "Only support NUM_EXTRA=0,1,2,3,4"); break;
  }
}

std::tuple<Tensor, torch::optional<Tensor>, Tensor, Tensor> gaussian_rasterize_backward(const Tensor& means2D,
    const Tensor& conic_opacity, const Tensor& colors, const torch::optional<Tensor> extras, const Tensor& out_opacity,
    const Tensor& dL_dout_color, const Tensor& dL_dout_opacity, const torch::optional<Tensor>& dL_dout_extra,
    const Tensor& ranges, const Tensor& point_list, const Tensor& n_contrib) {
  CHECK_CUDA(dL_dout_color);
  CHECK_CUDA(dL_dout_opacity);
  if (dL_dout_extra.has_value()) CHECK_CUDA(dL_dout_extra.value());

  const int P = means2D.size(0);
  const int H = dL_dout_color.size(IS_CHANNEL_FIRST ? 1 : 0);
  const int W = dL_dout_color.size(IS_CHANNEL_FIRST ? 2 : 1);
  const int E = dL_dout_extra.has_value() ? extras.value().size(-1) : 0;
  int64_t R   = point_list.size(0);

  auto options       = out_opacity.options();
  Tensor dL_dmeans2D = torch::zeros({P, 2}, options);
  Tensor dL_dconic_o = torch::zeros({P, 4}, options);
  // torch::Tensor dL_dopacity = torch::zeros({P, 1}, options);
  Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, options);
  torch::optional<Tensor> dL_dextras;
  if (E > 0) dL_dextras = torch::zeros({P, E}, options);

  if (P == 0) return std::make_tuple(dL_dcolors, dL_dextras, dL_dmeans2D, dL_dconic_o);

  const dim3 tile_grid((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, 1);
  const dim3 block(BLOCK_X, BLOCK_Y, 1);

  AT_DISPATCH_FLOATING_TYPES(means2D.scalar_type(), "gaussian_rasterize_backward", [&] {
    // using scalar_t = float;
    render_backward<scalar_t>(P, W, H, E, R, tile_grid, block,
        // inputs
        (T2<scalar_t>*) means2D.contiguous().data_ptr<scalar_t>(),
        (T4<scalar_t>*) conic_opacity.contiguous().data_ptr<scalar_t>(), colors.contiguous().data<scalar_t>(),
        E > 0 ? extras.value().contiguous().data<scalar_t>() : nullptr,
        // aux inputs
        (T2<int64_t>*) ranges.contiguous().data_ptr<int64_t>(), point_list.contiguous().data_ptr<int32_t>(),
        n_contrib.contiguous().data_ptr<int32_t>(),
        // outputs & grad_outputs
        out_opacity.contiguous().data<scalar_t>(), dL_dout_color.contiguous().data<scalar_t>(),
        E > 0 ? dL_dout_extra.value().contiguous().data<scalar_t>() : nullptr,
        dL_dout_opacity.contiguous().data<scalar_t>(),
        // grad_inputs
        (T2<scalar_t>*) dL_dmeans2D.contiguous().data<scalar_t>(),
        (T4<scalar_t>*) dL_dconic_o.contiguous().data<scalar_t>(),
        // dL_dopacity.contiguous().data<scalar_t>(),
        dL_dcolors.contiguous().data<scalar_t>(), E > 0 ? dL_dextras.value().data<scalar_t>() : nullptr);
    CHECK_CUDA_ERROR("render_backward");
  });
  return std::make_tuple(dL_dcolors, dL_dextras, dL_dmeans2D, dL_dconic_o);
}

// Based on Taming 3D-GS
template <typename T, uint32_t C, uint32_t E = 0>
__global__ void __launch_bounds__(BLOCK_X* BLOCK_Y) render_backward_per_gaussian_kernel(
    // const
    int P, int W, int H, int B,
    // aux
    const T2<int64_t>* __restrict__ ranges, const int32_t* __restrict__ point_list, /*const T* __restrict__ bg_color,*/
    const T2<T>* __restrict__ points_xy_image, const T4<T>* __restrict__ conic_opacity,
    const int64_t* __restrict__ per_tile_bucket_offset, const int32_t* __restrict__ bucket_to_tile,
    const T* __restrict__ sampled_T, const T* __restrict__ sampled_ar,
    // inputs
    const T* __restrict__ colors, const T* __restrict__ extras,
    // outputs
    const T* __restrict__ images, const T* __restrict__ out_extras, const T* __restrict__ out_opacity,
    const int32_t* __restrict__ n_contrib, const int32_t* __restrict__ max_contrib,
    // grad of outputs
    const T* __restrict__ dL_dpixels, const T* __restrict__ dL_dout_extra, const T* __restrict__ dL_dout_opacity,
    // grad of inputs
    T2<T>* __restrict__ dL_dmean2D, T4<T>* __restrict__ dL_dconic_opacity, T* __restrict__ dL_dcolors,
    T* __restrict__ dL_dextras) {
  auto block            = cg::this_thread_block();
  auto my_warp          = cg::tiled_partition<32>(block);
  uint32_t g_bucket_idx = block.group_index().x * my_warp.meta_group_size() + my_warp.meta_group_rank();
  bool valid_bucket     = g_bucket_idx < (uint32_t) B;
  if (!valid_bucket) return;

  bool valid_splat = false;
  uint32_t tile_id, bbm;
  T2<int64_t> range;
  int num_splats_in_tile, bucket_idx_in_tile;
  int splat_idx_in_tile, splat_idx_global;

  tile_id            = bucket_to_tile[g_bucket_idx];
  range              = ranges[tile_id];
  num_splats_in_tile = range.y - range.x;
  bbm                = tile_id == 0 ? 0 : per_tile_bucket_offset[tile_id - 1];
  bucket_idx_in_tile = g_bucket_idx - bbm;
  splat_idx_in_tile  = bucket_idx_in_tile * 32 + my_warp.thread_rank();
  splat_idx_global   = range.x + splat_idx_in_tile;
  valid_splat        = splat_idx_in_tile < num_splats_in_tile;
  if (bucket_idx_in_tile * 32 >= max_contrib[tile_id]) return;

  int gaussian_idx = 0;
  T2<T> xy         = {0.0, 0.0};
  T4<T> con_o      = {0, 0, 0, 0};
  T c[C + E]       = {0};

  if (valid_splat) {
    gaussian_idx = point_list[splat_idx_global];
    xy           = points_xy_image[gaussian_idx];
    con_o        = conic_opacity[gaussian_idx];
    for (int ch = 0; ch < C; ++ch) c[ch] = colors[gaussian_idx * C + ch];
    for (int e = 0; e < E; ++e) c[e + C] = extras[gaussian_idx * E + e];
  }
  // Gradient accumulation variables
  T r_dL_dmean2D_x      = 0;
  T r_dL_dmean2D_y      = 0;
  T r_dL_dconic_x       = 0;
  T r_dL_dconic_y       = 0;
  T r_dL_dconic_z       = 0;
  T r_dL_dconic_w       = 0;
  T r_dL_dcolors[C + E] = {0};

  const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
  const uint2 tile                 = {tile_id % horizontal_blocks, tile_id / horizontal_blocks};
  const uint2 pix_min              = {tile.x * BLOCK_X, tile.y * BLOCK_Y};

  T t, t_final, dL_dt_final;
  T last_contributor;
  T ar[C + E];
  T dL_dpixel[C + E];

  // iterate over all pixels in the tile
  for (int i = 0; i < BLOCK_SIZE + 31; ++i) {
    // SHUFFLING
    t                = my_warp.shfl_up(t, 1);
    last_contributor = my_warp.shfl_up(last_contributor, 1);
    t_final          = my_warp.shfl_up(t_final, 1);
    dL_dt_final      = my_warp.shfl_up(dL_dt_final, 1);
    for (int ch = 0; ch < C + E; ++ch) {
      ar[ch]        = my_warp.shfl_up(ar[ch], 1);
      dL_dpixel[ch] = my_warp.shfl_up(dL_dpixel[ch], 1);
    }
    // which pixel index should this thread deal with?
    int idx               = i - my_warp.thread_rank();
    const uint2 pix       = {pix_min.x + idx % BLOCK_X, pix_min.y + idx / BLOCK_X};
    const uint32_t pix_id = W * pix.y + pix.x;
    const T2<T> pixf      = {(T) pix.x, (T) pix.y};
    bool valid_pixel      = pix.x < W && pix.y < H;

    // every 32nd thread should read the stored state from memory
    if (valid_splat && valid_pixel && my_warp.thread_rank() == 0 && idx < BLOCK_SIZE) {
      t = sampled_T[g_bucket_idx * BLOCK_SIZE + idx];
      for (int ch = 0; ch < C; ++ch) {
        ar[ch] = -images[ch * H * W + pix_id] + sampled_ar[g_bucket_idx * BLOCK_SIZE * (C + E) + ch * BLOCK_SIZE + idx];
        dL_dpixel[ch] = dL_dpixels[ch * H * W + pix_id];
      }
      for (int e = 0; e < E; ++e) {
        ar[e + C] = -out_extras[e * H * W + pix_id] +
                    sampled_ar[g_bucket_idx * BLOCK_SIZE * (C + E) + (e + C) * BLOCK_SIZE + idx];
        dL_dpixel[e + E] = dL_dout_extra[e * H * W + pix_id];
      }
      t_final          = 1 - out_opacity[pix_id];
      last_contributor = n_contrib[pix_id];
      dL_dt_final      = -dL_dout_opacity[pix_id];
    }
    // do work
    if (valid_splat && valid_pixel && 0 <= idx && idx < BLOCK_SIZE) {
      if (W <= pix.x || H <= pix.y) continue;
      if (splat_idx_in_tile >= last_contributor) continue;

      const T2<T> d = {xy.x - pixf.x, xy.y - pixf.y};
      const T power = -T(0.5) * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
      if (power > 0) continue;
      const T G     = exp(power);
      const T alpha = min(T(0.99), con_o.w * G);
      if (alpha < T(1.0 / 255.)) continue;
      const T dchannel_dcolor = alpha * t;

      // add the gradient contribution of this pixel to the gaussian
      T bg_dot_dpixel = 0;
      T dL_dalpha     = 0;
      for (int ch = 0; ch < C; ++ch) {
        ar[ch] += t * alpha * c[ch];
        const T& dL_dchannel = dL_dpixel[ch];
        r_dL_dcolors[ch] += dchannel_dcolor * dL_dchannel;
        dL_dalpha += ((c[ch] * t) + (T(1.0) / (T(1.0) - alpha)) * ar[ch]) * dL_dchannel;
        // dL_dt_final += bg_color[ch] * dL_dpixel[ch];
      }
      for (int e = C; e < C + E; ++e) {
        ar[e] += t * alpha * c[e];
        const T& dL_de = dL_dpixel[e];
        r_dL_dcolors[e] += dchannel_dcolor * dL_de;
        dL_dalpha += ((c[e] * t) + (T(1.0) / (T(1.0) - alpha)) * ar[e]) * dL_de;
      }
      // if (pix.y == 126 && pix.x == 74) {
      //   printf("\033[32m[%d %d], idx=%d, ar=%f,%f,%f, alpha=%f, t=%f, dL_dalpha=%f, G=%f, bid=%d, i=%d\033[0m\n",
      //   pix.x,
      //       pix.y, gaussian_idx, -ar[0], -ar[1], -ar[2], alpha, t, dL_dalpha, G, bucket_idx_in_tile, i);
      // }
      dL_dalpha += (-t_final / (1.0 - alpha)) * dL_dt_final;
      t *= (1 - alpha);

      // Helpful reusable temporary variables
      const T dL_dG = -con_o.w * G * dL_dalpha;

      // accumulate the gradients
      r_dL_dmean2D_x += dL_dG * (d.x * con_o.x + d.y * con_o.y);
      r_dL_dmean2D_y += dL_dG * (d.y * con_o.z + d.x * con_o.y);

      r_dL_dconic_x += T(0.5) * d.x * d.x * dL_dG;
      r_dL_dconic_y += T(1.0) * d.x * d.y * dL_dG;
      r_dL_dconic_z += T(0.5) * d.y * d.y * dL_dG;
      r_dL_dconic_w += G * dL_dalpha;
    }
  }
  // finally add the gradients using atomics
  if (valid_splat) {
    atomicAdd(&dL_dmean2D[gaussian_idx].x, r_dL_dmean2D_x);
    atomicAdd(&dL_dmean2D[gaussian_idx].y, r_dL_dmean2D_y);
    atomicAdd(&dL_dconic_opacity[gaussian_idx].x, r_dL_dconic_x);
    atomicAdd(&dL_dconic_opacity[gaussian_idx].y, r_dL_dconic_y);
    atomicAdd(&dL_dconic_opacity[gaussian_idx].z, r_dL_dconic_z);
    atomicAdd(&dL_dconic_opacity[gaussian_idx].w, r_dL_dconic_w);
    for (int ch = 0; ch < C; ++ch) {
      atomicAdd(&dL_dcolors[gaussian_idx * C + ch], r_dL_dcolors[ch]);
    }
    for (int e = 0; e < E; ++e) {
      atomicAdd(&dL_dextras[gaussian_idx * E + e], r_dL_dcolors[e + C]);
    }
  }
}

template <typename T>
void render_backward_per_gaussian(int P, int W, int H, int E, int64_t R, int B,
    // inputs
    const T2<T>* means2D, const T4<T>* conic_opacity, const T* colors, const T* extras,
    // aux inputs
    const T2<int64_t>* ranges, const int32_t* point_list, const int32_t* n_contrib, const int32_t* max_contrib,
    const int64_t* per_tile_bucket_offset, const int32_t* bucket_to_tile, const T* sampled_T, const T* sampled_ar,
    // outputs & grad outputs
    const T* images, const T* out_extras, const T* out_opacity, const T* dL_dpixels, const T* dL_dout_extras,
    const T* dL_dout_opacity,
    // grad inputs
    T2<T>* dL_dmean2D, T4<T>* dL_dconic_opacity, T* dL_dcolors, T* dL_dextras) {
  int block = 32;
  int grid  = div_round_up(B * 32, block);

  switch (E) {
    case 0:
      render_backward_per_gaussian_kernel<T, NUM_CHANNELS, 0> KERNEL_ARG(grid, block)(P, W, H, B, ranges, point_list,
          means2D, conic_opacity, per_tile_bucket_offset, bucket_to_tile, sampled_T, sampled_ar, colors, extras, images,
          out_extras, out_opacity, n_contrib, max_contrib, dL_dpixels, dL_dout_extras, dL_dout_opacity, dL_dmean2D,
          dL_dconic_opacity, dL_dcolors, dL_dextras);
      break;
    case 1:
      render_backward_per_gaussian_kernel<T, NUM_CHANNELS, 1> KERNEL_ARG(grid, block)(P, W, H, B, ranges, point_list,
          means2D, conic_opacity, per_tile_bucket_offset, bucket_to_tile, sampled_T, sampled_ar, colors, extras, images,
          out_extras, out_opacity, n_contrib, max_contrib, dL_dpixels, dL_dout_extras, dL_dout_opacity, dL_dmean2D,
          dL_dconic_opacity, dL_dcolors, dL_dextras);
      break;
    case 2:
      render_backward_per_gaussian_kernel<T, NUM_CHANNELS, 2> KERNEL_ARG(grid, block)(P, W, H, B, ranges, point_list,
          means2D, conic_opacity, per_tile_bucket_offset, bucket_to_tile, sampled_T, sampled_ar, colors, extras, images,
          out_extras, out_opacity, n_contrib, max_contrib, dL_dpixels, dL_dout_extras, dL_dout_opacity, dL_dmean2D,
          dL_dconic_opacity, dL_dcolors, dL_dextras);
      break;
    case 3:
      render_backward_per_gaussian_kernel<T, NUM_CHANNELS, 3> KERNEL_ARG(grid, block)(P, W, H, B, ranges, point_list,
          means2D, conic_opacity, per_tile_bucket_offset, bucket_to_tile, sampled_T, sampled_ar, colors, extras, images,
          out_extras, out_opacity, n_contrib, max_contrib, dL_dpixels, dL_dout_extras, dL_dout_opacity, dL_dmean2D,
          dL_dconic_opacity, dL_dcolors, dL_dextras);
      break;
    case 4:
      render_backward_per_gaussian_kernel<T, NUM_CHANNELS, 4> KERNEL_ARG(grid, block)(P, W, H, B, ranges, point_list,
          means2D, conic_opacity, per_tile_bucket_offset, bucket_to_tile, sampled_T, sampled_ar, colors, extras, images,
          out_extras, out_opacity, n_contrib, max_contrib, dL_dpixels, dL_dout_extras, dL_dout_opacity, dL_dmean2D,
          dL_dconic_opacity, dL_dcolors, dL_dextras);
      break;
    default: BCNN_ASSERT(false, "Only support NUM_EXTRA=0,1,2,3,4"); break;
  }
}

std::tuple<Tensor, torch::optional<Tensor>, Tensor, Tensor> gaussian_rasterize_fast_backward(const Tensor& means2D,
    const Tensor& conic_opacity, const Tensor& colors, const torch::optional<Tensor> extras, const Tensor& images,
    const torch::optional<Tensor>& out_extras, const Tensor& out_opacity, const Tensor& dL_dout_color,
    const Tensor& dL_dout_opacity, const torch::optional<Tensor>& dL_dout_extra, const Tensor& ranges,
    const Tensor& point_list, const Tensor& n_contrib, const Tensor& max_contrib, const Tensor& per_tile_bucket_offset,
    const Tensor& bucket_to_tile, const Tensor& sampled_T, const Tensor& sampled_ar) {
  CHECK_CUDA(dL_dout_color);
  CHECK_CUDA(dL_dout_opacity);
  if (dL_dout_extra.has_value()) CHECK_CUDA(dL_dout_extra.value());

  const int P = means2D.size(0);
  const int H = dL_dout_color.size(IS_CHANNEL_FIRST ? 1 : 0);
  const int W = dL_dout_color.size(IS_CHANNEL_FIRST ? 2 : 1);
  const int E = dL_dout_extra.has_value() ? extras.value().size(-1) : 0;
  int64_t R   = point_list.size(0);

  auto options       = out_opacity.options();
  Tensor dL_dmeans2D = torch::zeros({P, 2}, options);
  Tensor dL_dconic_o = torch::zeros({P, 4}, options);
  // torch::Tensor dL_dopacity = torch::zeros({P, 1}, options);
  Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, options);
  torch::optional<Tensor> dL_dextras;
  if (E > 0) dL_dextras = torch::zeros({P, E}, options);

  if (P == 0) return std::make_tuple(dL_dcolors, dL_dextras, dL_dmeans2D, dL_dconic_o);

  AT_DISPATCH_FLOATING_TYPES(means2D.scalar_type(), "gaussian_rasterize_backward", [&] {
    // using scalar_t = float;
    int B = per_tile_bucket_offset[-1].item<int64_t>();
    render_backward_per_gaussian<scalar_t>(P, W, H, E, R, B,
        // inputs
        (T2<scalar_t>*) means2D.contiguous().data_ptr<scalar_t>(),
        (T4<scalar_t>*) conic_opacity.contiguous().data_ptr<scalar_t>(), colors.contiguous().data<scalar_t>(),
        E > 0 ? extras.value().contiguous().data<scalar_t>() : nullptr,
        // aux inputs
        (T2<int64_t>*) ranges.contiguous().data_ptr<int64_t>(), point_list.contiguous().data_ptr<int32_t>(),
        n_contrib.contiguous().data_ptr<int32_t>(), max_contrib.contiguous().data_ptr<int32_t>(),
        per_tile_bucket_offset.contiguous().data_ptr<int64_t>(), bucket_to_tile.contiguous().data_ptr<int32_t>(),
        sampled_T.contiguous().data_ptr<scalar_t>(), sampled_ar.contiguous().data_ptr<scalar_t>(),
        // outputs & grad_outputs
        images.contiguous().data<scalar_t>(), E > 0 ? out_extras.value().contiguous().data<scalar_t>() : nullptr,
        out_opacity.contiguous().data<scalar_t>(), dL_dout_color.contiguous().data<scalar_t>(),
        E > 0 ? dL_dout_extra.value().contiguous().data<scalar_t>() : nullptr,
        dL_dout_opacity.contiguous().data<scalar_t>(),
        // grad_inputs
        (T2<scalar_t>*) dL_dmeans2D.contiguous().data<scalar_t>(),
        (T4<scalar_t>*) dL_dconic_o.contiguous().data<scalar_t>(),
        // dL_dopacity.contiguous().data<scalar_t>(),
        dL_dcolors.contiguous().data<scalar_t>(), E > 0 ? dL_dextras.value().data<scalar_t>() : nullptr);
    CHECK_CUDA_ERROR("render_backward");
  });
  return std::make_tuple(dL_dcolors, dL_dextras, dL_dmeans2D, dL_dconic_o);
}

REGIST_PYTORCH_EXTENSION(gs_gaussian_render, {
  m.def("gaussian_rasterize_forward", &gaussian_rasterize_forward, "gaussian_rasterize_forward (CUDA)");
  m.def("gaussian_rasterize_backward", &gaussian_rasterize_backward, "gaussian_rasterize_backward (CUDA)");
  m.def(
      "gaussian_rasterize_fast_backward", &gaussian_rasterize_fast_backward, "gaussian_rasterize_fast_backward (CUDA)");
})

#define INSTANCE_FUNC(T)                                                                                             \
  template void render_forward<T>(const dim3 grid, dim3 block, const T2<int64_t>* ranges, const int32_t* point_list, \
      int W, int H, int E, const T2<T>* means2D, const T* colors, const T4<T>* conic_opacity, const T* extra,        \
      int32_t* n_contrib, T* out_color, T* out_opacity, T* out_extra, int32_t* accum_max_count, T* accum_weights_p,  \
      int32_t* accum_weights_count, const int64_t* per_tile_bucket_offset, int32_t* bucket_to_tile, T* sampled_T,    \
      T* sampled_ar, int32_t* max_contrib);                                                                          \
  template void render_backward<T>(int P, int W, int H, int E, int64_t R, const dim3 grid, const dim3 block,         \
      const T2<T>* means2D, const T4<T>* conic_opacity, const T* colors, const T* extras, const T2<int64_t>* ranges, \
      const int32_t* point_list, const int32_t* n_contrib, const T* out_opacity, const T* dL_dpixels,                \
      const T* dL_dout_extras, const T* dL_dout_opacity, T2<T>* dL_dmean2D, T4<T>* dL_dconic_opacity, T* dL_dcolors, \
      T* dL_dextras);

INSTANCE_FUNC(float);
INSTANCE_FUNC(double);

}  // namespace GaussianRasterizer
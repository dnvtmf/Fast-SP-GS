#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include "gaussian_render.h"
#include "ops_3d.h"
#include "util.cuh"

namespace cg = cooperative_groups;
using namespace OPS_3D;
namespace GaussianRasterizer {
#define DEPTH_OFFSET 0
// #define ALPHA_OFFSET 1
#define NORMAL_OFFSET 1
#define MIDDEPTH_OFFSET 4
#define DISTORTION_OFFSET 5
#define M1_OFFSET 6
#define M2_OFFSET 7
#define DETACH_WEIGHT 0

constexpr double FilterInvSquare = 2.0;

template <typename T, uint32_t CHANNELS, bool RENDER_AXUTILITY = false>
__global__ void __launch_bounds__(BLOCK_X* BLOCK_Y) gs_2d_fast_render_forward_kernel(
    // scalar
    int W, int H, T near_n, T far_n,
    // inputs
    const T2<int64_t>* __restrict__ ranges, const int32_t* __restrict__ point_list,
    const T2<T>* __restrict__ points_xy_image, const T* __restrict__ features, const T* __restrict__ transMats,
    const T* __restrict__ inverse_M, const T4<T>* __restrict__ normal_opacity, const T* __restrict__ trMats_t2,
    // outputs
    int32_t* __restrict__ n_contrib, T* __restrict__ out_color, T* __restrict__ out_opacity, T* __restrict__ out_others,
    T2<T>* __restrict__ out_flow,
    // for fast backward
    const int64_t* __restrict__ per_tile_bucket_offset, int32_t* __restrict__ bucket_to_tile, T* __restrict__ sampled_T,
    T* __restrict__ sampled_acc, T* __restrict__ sampled_aux, int32_t* __restrict__ max_contrib,
    //
    int32_t* __restrict__ accum_max_count = nullptr, T* __restrict__ accum_weights_p = nullptr,
    int32_t* __restrict__ accum_weights_count = nullptr) {
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
  uint32_t tile_id  = block.group_index().y * horizontal_blocks + block.group_index().x;
  T2<int64_t> range = ranges[tile_id];
  const int rounds  = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
  int toDo          = range.y - range.x;
  uint32_t bbm;
  int num_buckets = (toDo + 31) / 32;
  if (bucket_to_tile != nullptr) {
    bbm = tile_id == 0 ? 0 : per_tile_bucket_offset[tile_id - 1];
    for (int i = block.thread_rank(); i < num_buckets; i += BLOCK_SIZE) bucket_to_tile[bbm + i] = tile_id;
  }

  // Allocate storage for batches of collectively fetched data.
  __shared__ int collected_id[BLOCK_SIZE];
  __shared__ T2<T> collected_xy[BLOCK_SIZE];
  __shared__ T4<T> collected_normal_opacity[BLOCK_SIZE];
  __shared__ vec3<T> collected_mA[BLOCK_SIZE];
  __shared__ vec3<T> collected_mB[BLOCK_SIZE];
  __shared__ vec3<T> collected_mC[BLOCK_SIZE];
  __shared__ vec3<T> collected_Tw[BLOCK_SIZE];
  __shared__ T collected_M[BLOCK_SIZE * 9];

  // Initialize helper variables
  T t                      = T(1.0);
  int32_t contributor      = 0;
  int32_t last_contributor = 0;
  T C[CHANNELS]            = {0};

  T weight_max          = 0;
  int32_t weight_max_id = -1;
  bool flag_update      = false;

  // render axutility ouput
  const int NUM  = out_flow != nullptr ? 10 : 8;
  T N[3]         = {0};
  T D            = {0};
  T M1           = {0};
  T M2           = {0};
  T distortion   = {0};
  T median_depth = {0};
  // T median_weight = {0};
  int32_t median_contributor = {-1};
  T2<T> flow                 = {0};

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
      collected_mA[block.thread_rank()]             = {
          inverse_M[9 * coll_id + 0], inverse_M[9 * coll_id + 1], inverse_M[9 * coll_id + 2]};
      collected_mB[block.thread_rank()] = {
          inverse_M[9 * coll_id + 3], inverse_M[9 * coll_id + 4], inverse_M[9 * coll_id + 5]};
      collected_mC[block.thread_rank()] = {
          inverse_M[9 * coll_id + 6], inverse_M[9 * coll_id + 7], inverse_M[9 * coll_id + 8]};
      collected_Tw[block.thread_rank()] = {
          transMats[9 * coll_id + 6], transMats[9 * coll_id + 7], transMats[9 * coll_id + 8]};
      if constexpr (RENDER_AXUTILITY) {
        if (trMats_t2 != nullptr) {
#pragma unroll
          for (int k = 0; k < 9; ++k) collected_M[block.thread_rank() * 9 + k] = trMats_t2[coll_id * 9 + k];
        }
      }
    }
    block.sync();

    // Iterate over current batch
    for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++) {
      // log sampled data
      if (j % 32 == 0 && sampled_T != nullptr) {
        sampled_T[bbm * BLOCK_SIZE + block.thread_rank()] = t;
        for (int c = 0; c < CHANNELS; ++c) {
          sampled_acc[(bbm * CHANNELS + c) * BLOCK_SIZE + block.thread_rank()] = C[c];
        }
        if constexpr (RENDER_AXUTILITY) {
          sampled_aux[(bbm * NUM + DEPTH_OFFSET) * BLOCK_SIZE + block.thread_rank()] = D;
          for (int c = 0; c < 3; ++c)
            sampled_aux[(bbm * NUM + NORMAL_OFFSET + c) * BLOCK_SIZE + block.thread_rank()] = N[c];
          sampled_aux[(bbm * NUM + DISTORTION_OFFSET) * BLOCK_SIZE + block.thread_rank()] = distortion;
          sampled_aux[(bbm * NUM + M1_OFFSET) * BLOCK_SIZE + block.thread_rank()]         = M1;
          sampled_aux[(bbm * NUM + M2_OFFSET) * BLOCK_SIZE + block.thread_rank()]         = M2;
          if (out_flow != nullptr) {
            sampled_aux[(bbm * NUM + 8) * BLOCK_SIZE + block.thread_rank()] = flow.x;
            sampled_aux[(bbm * NUM + 9) * BLOCK_SIZE + block.thread_rank()] = flow.y;
          }
        }
        ++bbm;
      }
      // Keep track of current position in range
      contributor++;

      // Fisrt compute two homogeneous planes, See Eq. (8)
      const T2<T> xy   = collected_xy[j];
      const vec3<T> mA = collected_mA[j];
      const vec3<T> mB = collected_mB[j];
      const vec3<T> mC = collected_mC[j];
      const vec3<T> Tw = collected_Tw[j];
      vec3<T> p        = {
          mA.x * pixf.x + mA.y * pixf.y + mA.z,
          mB.x * pixf.x + mB.y * pixf.y + mB.z,
          mC.x * pixf.x + mC.y * pixf.y + mC.z,
      };
      // if (pix_id == 15 && collected_id[j] == 100821) printf("\033[33mxy=%f %f\033[0m\n", xy.x, xy.y);
      if (p.z == 0.0) continue;
      T2<T> s = {p.x / p.z, p.y / p.z};
      T rho3d = (s.x * s.x + s.y * s.y);
      T2<T> d = {xy.x - pixf.x, xy.y - pixf.y};
      T rho2d = T(FilterInvSquare) * (d.x * d.x + d.y * d.y);

      // compute intersection and depth
      // T rho   = rho3d;
      T rho   = min(rho3d, rho2d);
      T depth = (s.x * Tw.x + s.y * Tw.y) + Tw.z;
      // T depth = rho3d <= rho2d ? (s.x * Tw.x + s.y * Tw.y) + Tw.z : Tw.z;
      if (depth < near_n) continue;
      T4<T> nor_o = collected_normal_opacity[j];
      T normal[3] = {nor_o.x, nor_o.y, nor_o.z};
      T opa       = nor_o.w;

      T power = -T(0.5) * rho;
      if (power > T(0.0)) continue;

      // Eq. (2) from 3D Gaussian splatting paper.
      // Obtain alpha by multiplying with Gaussian opacity and its exponential falloff from mean.
      // Avoid numerical instabilities (see paper appendix).
      T alpha = min(T(0.99), opa * exp(power));
      if (alpha < T(1.0 / 255.0)) continue;
      T test_T = t * (1 - alpha);
      if (test_T < T(0.0001)) {
        done = true;
        continue;
      }

      T w = alpha * t;
      if constexpr (RENDER_AXUTILITY) {
        // Render depth distortion map Efficient implementation of distortion loss, see 2DGS' paper appendix.
        T A = 1 - t;
        T m = far_n / (far_n - near_n) * (1 - near_n / depth);
        distortion += (m * m * A + M2 - 2 * m * M1) * w;

        D += depth * w;
        M1 += m * w;
        M2 += m * m * w;
        if (t > 0.5) {
          median_depth = depth;
          // median_weight = w;
          median_contributor = contributor;
        }
        // Render normal map
        for (int ch = 0; ch < 3; ch++) N[ch] += normal[ch] * w;

        if (weight_max < w) {
          weight_max    = w;
          weight_max_id = collected_id[j];
          flag_update   = true;
        }
        if (accum_weights_p) atomicAdd(&(accum_weights_p[collected_id[j]]), w);
        if (accum_weights_count) atomicAdd(&(accum_weights_count[collected_id[j]]), 1);
        if (trMats_t2 != nullptr) {
          T u_t2 = collected_M[j * 9 + 0] * s.x + collected_M[j * 9 + 1] * s.y + collected_M[j * 9 + 2];
          T v_t2 = collected_M[j * 9 + 3] * s.x + collected_M[j * 9 + 4] * s.y + collected_M[j * 9 + 5];
          T w_t2 = collected_M[j * 9 + 6] * s.x + collected_M[j * 9 + 7] * s.y + collected_M[j * 9 + 8];
          w_t2   = abs(w_t2) < 1e-5 ? 1e-5 : w_t2;  // avoid divide 0
          flow.x += (u_t2 / w_t2 - pixf.x) * w;
          flow.y += (v_t2 / w_t2 - pixf.y) * w;
        }
      }
      if (flag_update && accum_max_count) atomicAdd(&(accum_max_count[weight_max_id]), 1);

      // Eq. (3) from 3D Gaussian splatting paper.
      for (int ch = 0; ch < CHANNELS; ch++) C[ch] += features[collected_id[j] * CHANNELS + ch] * w;
      t = test_T;

      // Keep track of last range entry to update this pixel.
      last_contributor = contributor;
      // if (pix_id == 10673) {
      //   printf("\033[32mgs=%d, xy=%f, %f, rho=%f, depth=%f, alpha=%f, sigma=%f\033[0m\n", collected_id[j], s.x, s.y,
      //       rho, depth, alpha, w);
      // }
    }
  }

  // All threads that treat valid pixel write out their final rendering data to the frame and auxiliary buffers.
  if (inside) {
    t                   = T(1) - t;
    out_opacity[pix_id] = t;
    n_contrib[pix_id]   = last_contributor;
    for (int ch = 0; ch < CHANNELS; ch++) out_color[ch * H * W + pix_id] = C[ch];  //+ t * bg_color[ch];
    if constexpr (RENDER_AXUTILITY) {
      n_contrib[pix_id + H * W] = median_contributor;

      out_others[pix_id + H * W * DEPTH_OFFSET] = D;
      // out_others[pix_id + ALPHA_OFFSET * H * W] = 1 - t;
      for (int ch = 0; ch < 3; ch++) out_others[pix_id + (NORMAL_OFFSET + ch) * H * W] = N[ch];
      out_others[pix_id + H * W * MIDDEPTH_OFFSET]   = median_depth;
      out_others[pix_id + H * W * DISTORTION_OFFSET] = distortion;
      // out_others[pix_id + MEDIAN_WEIGHT_OFFSET * H * W] = median_weight;
      out_others[pix_id + H * W * M1_OFFSET] = M1;
      out_others[pix_id + H * W * M2_OFFSET] = M2;
      if (out_flow != nullptr) {
        flow.x           = t > 0 ? flow.x / t : T(0);
        flow.y           = t > 0 ? flow.y / t : T(0);
        out_flow[pix_id] = flow;
      }
    }
  }
  // max reduce the last contributor
  if (max_contrib != nullptr) {
    reduce_max_block<int32_t, false>(last_contributor);
    if (block.thread_rank() == 0) max_contrib[tile_id] = last_contributor;
  }
}  // namespace GaussianRasterizer

template <typename T = float, uint32_t C = NUM_CHANNELS, bool RENDER_AXUTILITY = true>
__global__ void __launch_bounds__(BLOCK_X* BLOCK_Y) gs_2d_fast_render_backward_kernel(
    // scalaes
    int P, int W, int H, int64_t R, T near_n, T far_n,
    // aux
    const T2<int64_t>* __restrict__ ranges, const int32_t* __restrict__ point_list,
    const int32_t* __restrict__ n_contrib,
    // inputs
    const T2<T>* __restrict__ points_xy_image, const T4<T>* __restrict__ normal_opacity, const T* __restrict__ colors,
    const T* __restrict__ transMats, const T* __restrict__ inverse_M, const T* __restrict__ transMat2,
    // outputs
    const T* __restrict__ out_opacity, const T* __restrict__ out_others,
    // grad outputs
    const T* __restrict__ dL_dpixels, const T* __restrict__ dL_dothers, const T* __restrict__ dL_dout_opacity,
    const T2<T>* __restrict__ dL_dout_flows,
    // grad inputs
    T2<T>* __restrict__ dL_dmean2D, T* __restrict__ dL_dnormal_dopacity, T* __restrict__ dL_dcolors,
    T* __restrict__ dL_dtransMat, T* __restrict__ dL_dinverse_M) {
  // We rasterize again. Compute necessary block info.
  auto block                       = cg::this_thread_block();
  const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
  const uint2 pix_min              = {block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y};
  const uint2 pix_max              = {min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y, H)};
  const uint2 pix                  = {pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y};
  const uint32_t pix_id            = W * pix.y + pix.x;
  const T2<T> pixf                 = {(T) pix.x, (T) pix.y};

  const bool inside       = pix.x < W && pix.y < H;
  const T2<int64_t> range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

  const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

  bool done = !inside;
  int toDo  = range.y - range.x;

  __shared__ int collected_id[BLOCK_SIZE];
  __shared__ T2<T> collected_xy[BLOCK_SIZE];
  __shared__ T4<T> collected_normal_opacity[BLOCK_SIZE];
  __shared__ T collected_colors[C * BLOCK_SIZE];
  __shared__ vec3<T> collected_mA[BLOCK_SIZE];
  __shared__ vec3<T> collected_mB[BLOCK_SIZE];
  __shared__ vec3<T> collected_mC[BLOCK_SIZE];
  __shared__ vec3<T> collected_Tw[BLOCK_SIZE];
  __shared__ T collected_M[BLOCK_SIZE * 9];
  // __shared__ float collected_depths[BLOCK_SIZE];

  // In the forward, we stored the final value for T, the product of all (1 - alpha) factors.
  const T t_final = inside ? 1 - out_opacity[pix_id] : 0;
  T t             = t_final;

  // We start from the back. The ID of the last contributing Gaussian is known from each pixel from the forward.
  uint32_t contributor       = toDo;
  const int last_contributor = inside ? n_contrib[pix_id] : 0;

  T accum_rec[C] = {0};
  T dL_dpixel[C] = {0};

  T dL_dreg;
  T dL_ddepth;
  T dL_daccum;
  T dL_dnormal2D[3];
  const int median_contributor = (RENDER_AXUTILITY & inside) ? n_contrib[pix_id + H * W] : 0;
  T dL_dmedian_depth;
  T dL_dmax_dweight;

  if constexpr (RENDER_AXUTILITY) {
    if (inside) {
      dL_daccum = dL_dout_opacity[pix_id];
      dL_ddepth = dL_dothers[DEPTH_OFFSET * H * W + pix_id];
      dL_dreg   = dL_dothers[DISTORTION_OFFSET * H * W + pix_id];
      for (int i = 0; i < 3; i++) dL_dnormal2D[i] = dL_dothers[(NORMAL_OFFSET + i) * H * W + pix_id];
      dL_dmedian_depth = dL_dothers[MIDDEPTH_OFFSET * H * W + pix_id];
      // dL_dmax_dweight = dL_depths[MEDIAN_WEIGHT_OFFSET * H * W + pix_id];
    }
  }

  // for compute gradient with respect to depth and normal
  T last_depth          = 0;
  T last_normal[3]      = {0};
  T accum_depth_rec     = 0;
  T accum_alpha_rec     = 0;
  T accum_normal_rec[3] = {0};
  // for compute gradient with respect to the distortion map
  const T final_D  = (RENDER_AXUTILITY & inside) ? out_others[pix_id + M1_OFFSET * H * W] : 0;
  const T final_D2 = (RENDER_AXUTILITY & inside) ? out_others[pix_id + M2_OFFSET * H * W] : 0;
  const T final_A  = 1 - t_final;
  T last_dL_dT     = 0;

  if (inside) {
    for (int i = 0; i < C; i++) dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];
  }

  T last_alpha    = 0;
  T last_color[C] = {0};

  T2<T> accum_flow = {0}, last_flow = {0}, dL_dflow;
  if (inside && dL_dout_flows != nullptr) dL_dflow = dL_dout_flows[pix_id];

  // Traverse all Gaussians
  for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE) {
    // Load auxiliary data into shared memory, start in the BACK and load them in revers order.
    block.sync();
    const int progress = i * BLOCK_SIZE + block.thread_rank();
    if (range.x + progress < range.y) {
      const int coll_id                             = point_list[range.y - progress - 1];
      collected_id[block.thread_rank()]             = coll_id;
      collected_xy[block.thread_rank()]             = points_xy_image[coll_id];
      collected_normal_opacity[block.thread_rank()] = normal_opacity[coll_id];
      collected_mA[block.thread_rank()]             = {
          inverse_M[9 * coll_id + 0], inverse_M[9 * coll_id + 1], inverse_M[9 * coll_id + 2]};
      collected_mB[block.thread_rank()] = {
          inverse_M[9 * coll_id + 3], inverse_M[9 * coll_id + 4], inverse_M[9 * coll_id + 5]};
      collected_mC[block.thread_rank()] = {
          inverse_M[9 * coll_id + 6], inverse_M[9 * coll_id + 7], inverse_M[9 * coll_id + 8]};
      collected_Tw[block.thread_rank()] = {
          transMats[9 * coll_id + 6], transMats[9 * coll_id + 7], transMats[9 * coll_id + 8]};
      for (int c = 0; c < C; c++) collected_colors[c * BLOCK_SIZE + block.thread_rank()] = colors[coll_id * C + c];
      if (dL_dout_flows != nullptr)
        for (int c = 0; c < 9; ++c) collected_M[block.thread_rank() * 9 + c] = transMat2[coll_id * 9 + c];
      // collected_depths[block.thread_rank()] = depths[coll_id];
    }
    block.sync();

    // Iterate over Gaussians
    for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++) {
      // Keep track of current Gaussian ID. Skip, if this one is behind the last contributor for this pixel.
      contributor--;
      if (contributor >= last_contributor) continue;

      // compute ray-splat intersection as before Fisrt compute two homogeneous planes, See Eq. (8)
      const T2<T> xy   = collected_xy[j];
      const vec3<T> mA = collected_mA[j];
      const vec3<T> mB = collected_mB[j];
      const vec3<T> mC = collected_mC[j];
      const vec3<T> Tw = collected_Tw[j];
      vec3<T> p        = {
          mA.x * pixf.x + mA.y * pixf.y + mA.z,
          mB.x * pixf.x + mB.y * pixf.y + mB.z,
          mC.x * pixf.x + mC.y * pixf.y + mC.z,
      };
      if (p.z == 0.0) continue;
      T2<T> s = {p.x / p.z, p.y / p.z};
      T rho3d = (s.x * s.x + s.y * s.y);
      T2<T> d = {xy.x - pixf.x, xy.y - pixf.y};
      T rho2d = T(FilterInvSquare) * (d.x * d.x + d.y * d.y);

      // compute intersection and depth
      T rho = min(rho3d, rho2d);
      T c_d = s.x * Tw.x + s.y * Tw.y + Tw.z;
      if (c_d < near_n) continue;
      T4<T> nor_o = collected_normal_opacity[j];
      T normal[3] = {nor_o.x, nor_o.y, nor_o.z};
      T opa       = nor_o.w;

      // accumulations
      T power = -T(0.5) * rho;
      if (power > T(0.0)) continue;

      const T G     = exp(power);
      const T alpha = min(T(0.99), opa * G);
      if (alpha < T(1.0 / 255.0)) continue;

      t                       = t / (T(1.) - alpha);
      const T dchannel_dcolor = alpha * t;
      const T w               = alpha * t;
      // Propagate gradients to per-Gaussian colors and keep
      // gradients w.r.t. alpha (blending factor for a Gaussian/pixel pair).
      T dL_dalpha         = T(0.0);
      const int global_id = collected_id[j];
      for (int ch = 0; ch < C; ch++) {
        const T c = collected_colors[ch * BLOCK_SIZE + j];
        // Update last color (to be used in the next iteration)
        accum_rec[ch]  = last_alpha * last_color[ch] + (T(1.) - last_alpha) * accum_rec[ch];
        last_color[ch] = c;

        const T dL_dchannel = dL_dpixel[ch];
        dL_dalpha += (c - accum_rec[ch]) * dL_dchannel;
        // Update the gradients w.r.t. color of the Gaussian.
        // Atomic, since this pixel is just one of potentially many that were affected by this Gaussian.
        atomicAdd(&(dL_dcolors[global_id * C + ch]), dchannel_dcolor * dL_dchannel);
      }

      T dL_dz      = T(0.0);
      T dL_dweight = 0;
      T dL_dsx = 0, dL_dsy = 0;
      if constexpr (RENDER_AXUTILITY) {
        const T m_d    = far_n / (far_n - near_n) * (1 - near_n / c_d);
        const T dmd_dd = (far_n * near_n) / ((far_n - near_n) * c_d * c_d);
        if (contributor == median_contributor - 1) {
          dL_dz += dL_dmedian_depth;
          // dL_dweight += dL_dmax_dweight;
        }
#if DETACH_WEIGHT
        // if not detached weight, sometimes it will bia toward creating extragated 2D Gaussians near front
        dL_dweight += 0;
#else
        dL_dweight += (final_D2 + m_d * m_d * final_A - 2 * m_d * final_D) * dL_dreg;
#endif
        dL_dalpha += dL_dweight - last_dL_dT;
        // propagate the current weight W_{i} to next weight W_{i-1}
        last_dL_dT     = dL_dweight * alpha + (1 - alpha) * last_dL_dT;
        const T dL_dmd = T(2.0) * (t * alpha) * (m_d * final_A - final_D) * dL_dreg;
        dL_dz += dL_dmd * dmd_dd;

        // Propagate gradients w.r.t ray-splat depths
        accum_depth_rec = last_alpha * last_depth + (T(1.) - last_alpha) * accum_depth_rec;
        last_depth      = c_d;
        dL_dalpha += (c_d - accum_depth_rec) * dL_ddepth;
        // Propagate gradients w.r.t. color ray-splat alphas
        accum_alpha_rec = last_alpha * 1.0 + (T(1.) - last_alpha) * accum_alpha_rec;
        dL_dalpha += (1 - accum_alpha_rec) * dL_daccum;

        // Propagate gradients to per-Gaussian normals
        for (int ch = 0; ch < 3; ch++) {
          accum_normal_rec[ch] = last_alpha * last_normal[ch] + (T(1.) - last_alpha) * accum_normal_rec[ch];
          last_normal[ch]      = normal[ch];
          dL_dalpha += (normal[ch] - accum_normal_rec[ch]) * dL_dnormal2D[ch];
          atomicAdd((&dL_dnormal_dopacity[global_id * 4 + ch]), w * dL_dnormal2D[ch]);
        }
        dL_dz += w * dL_ddepth;
      }
      if (dL_dout_flows != nullptr) {
        T u_t2       = collected_M[j * 9 + 0] * s.x + collected_M[j * 9 + 1] * s.y + collected_M[j * 9 + 2];
        T v_t2       = collected_M[j * 9 + 3] * s.x + collected_M[j * 9 + 4] * s.y + collected_M[j * 9 + 5];
        T w_t2       = collected_M[j * 9 + 6] * s.x + collected_M[j * 9 + 7] * s.y + collected_M[j * 9 + 8];
        w_t2         = abs(w_t2) < 1e-5 ? 1e-5 : w_t2;  // avoid divide 0
        accum_flow.x = last_alpha * last_flow.x + (T(1.) - last_alpha) * accum_flow.x;
        accum_flow.y = last_alpha * last_flow.y + (T(1.) - last_alpha) * accum_flow.y;
        last_flow    = {(u_t2 / w_t2 - pixf.x), (v_t2 / w_t2 - pixf.y)};
        dL_dalpha += (last_flow.x - accum_flow.x) * dL_dflow.x + (last_flow.y - accum_flow.y) * dL_dflow.y;
        T gu = w / w_t2 * dL_dflow.x, gv = w / w_t2 * dL_dflow.y,
          gw = -w / (w_t2 * w_t2) * (u_t2 * dL_dflow.x + v_t2 * dL_dflow.y);
        dL_dsx += collected_M[j * 9 + 0] * gu + collected_M[j * 9 + 3] * gv + collected_M[j * 9 + 6] * gw;
        dL_dsy += collected_M[j * 9 + 1] * gu + collected_M[j * 9 + 4] * gv + collected_M[j * 9 + 7] * gw;
      }
      // if (pix_id == 10673) {
      //   printf("\033[33m[%d %d], gs=%d, xy=%f, %f, rho=%f, depth=%f, alpha=%f, sigma=%f\033[0m\n", pix.x, pix.y,
      //       collected_id[j], xy.x, xy.y, rho, c_d, alpha, w);
      // }
      dL_dalpha *= t;
      last_alpha = alpha;  // Update last alpha (to be used in the next iteration)

      // Account for fact that alpha also influences how much of the background color is added if nothing left to blend
      //   T bg_dot_dpixel = 0;
      //   for (int i = 0; i < C; i++) bg_dot_dpixel += bg_color[i] * dL_dpixel[i];
      //   dL_dalpha += (-t_final / (T(1.)- alpha)) * bg_dot_dpixel;

      // Helpful reusable temporary variables
      const T dL_dG = nor_o.w * dL_dalpha;

      if (rho3d <= rho2d) {
        // Update gradients w.r.t. covariance of Gaussian 3x3 (T)
        dL_dsx += dL_dG * -G * s.x + dL_dz * Tw.x;
        dL_dsy += dL_dG * -G * s.y + dL_dz * Tw.y;
        const T dsx_pz      = dL_dsx / p.z;
        const T dsy_pz      = dL_dsy / p.z;
        const vec3<T> dL_dp = {dsx_pz, dsy_pz, -(dsx_pz * s.x + dsy_pz * s.y)};

        // Update gradients w.r.t. 3D covariance (3x3 matrix)
        atomicAdd(&dL_dinverse_M[global_id * 9 + 0], dL_dp.x * pixf.x);
        atomicAdd(&dL_dinverse_M[global_id * 9 + 1], dL_dp.x * pixf.y);
        atomicAdd(&dL_dinverse_M[global_id * 9 + 2], dL_dp.x);
        atomicAdd(&dL_dinverse_M[global_id * 9 + 3], dL_dp.y * pixf.x);
        atomicAdd(&dL_dinverse_M[global_id * 9 + 4], dL_dp.y * pixf.y);
        atomicAdd(&dL_dinverse_M[global_id * 9 + 5], dL_dp.y);
        atomicAdd(&dL_dinverse_M[global_id * 9 + 6], dL_dp.z * pixf.x);
        atomicAdd(&dL_dinverse_M[global_id * 9 + 7], dL_dp.z * pixf.y);
        atomicAdd(&dL_dinverse_M[global_id * 9 + 8], dL_dp.z);

        atomicAdd(&dL_dtransMat[global_id * 9 + 6], dL_dz * s.x);
        atomicAdd(&dL_dtransMat[global_id * 9 + 7], dL_dz * s.y);
        atomicAdd(&dL_dtransMat[global_id * 9 + 8], dL_dz);
      } else {
        // Update gradients w.r.t. center of Gaussian 2D mean position
        const T dG_ddelx = -G * FilterInvSquare * d.x;
        const T dG_ddely = -G * FilterInvSquare * d.y;
        atomicAdd(&dL_dmean2D[global_id].x, dL_dG * dG_ddelx);  // not scaled
        atomicAdd(&dL_dmean2D[global_id].y, dL_dG * dG_ddely);  // not scaled

        atomicAdd(&dL_dtransMat[global_id * 9 + 6], dL_dz * s.x);
        atomicAdd(&dL_dtransMat[global_id * 9 + 7], dL_dz * s.y);
        atomicAdd(&dL_dtransMat[global_id * 9 + 8], dL_dz);
      }

      // Update gradients w.r.t. opacity of the Gaussian
      atomicAdd(&(dL_dnormal_dopacity[global_id * 4 + 3]), G * dL_dalpha);
    }
  }
}

template <typename T = float, uint32_t C = NUM_CHANNELS, bool RENDER_AXUTILITY = true>
__global__ void __launch_bounds__(BLOCK_X* BLOCK_Y) gs_2d_fast_render_backward_v2_kernel(
    // scalaes
    int P, int W, int H, int64_t R, uint32_t B, T near_n, T far_n,
    // aux
    const T2<int64_t>* __restrict__ ranges, const int32_t* __restrict__ point_list,
    const int32_t* __restrict__ n_contrib, const int32_t* __restrict__ max_contrib,
    const int64_t* __restrict__ per_tile_bucket_offset, const int32_t* __restrict__ bucket_to_tile,
    const T* __restrict__ sampled_T, const T* __restrict__ sampled_ar, const T* __restrict__ sampled_aux,
    // inputs
    const T2<T>* __restrict__ points_xy_image, const T4<T>* __restrict__ normal_opacity, const T* __restrict__ colors,
    const T* __restrict__ transMats, const T* __restrict__ inverse_M, const T* __restrict__ Mats_t2,
    // outputs
    const T* __restrict__ images, const T* __restrict__ out_opacity, const T* __restrict__ out_others,
    const T* __restrict__ out_flows,
    // grad outputs
    const T* __restrict__ dL_dpixels, const T* __restrict__ dL_dothers, const T* __restrict__ dL_dout_opacity,
    const T* __restrict__ dL_dout_flow,
    // grad inputs
    T2<T>* __restrict__ dL_dmean2D, T* __restrict__ dL_dnormal_dopacity, T* __restrict__ dL_dcolors,
    T* __restrict__ dL_dtransMat, T* __restrict__ dL_dinverse_M) {
  auto block        = cg::this_thread_block();
  auto warp         = cg::tiled_partition<32>(block);
  uint32_t g_bucket = block.group_index().x * warp.meta_group_size() + warp.meta_group_rank();
  if (g_bucket >= B) return;
  bool valid_splat = false;
  uint32_t tile_id, bbm;
  T2<int64_t> range;
  int num_splats_in_tile, bucket_idx_in_tile;
  int splat_idx_l, splat_idx_g;
  tile_id            = bucket_to_tile[g_bucket];
  range              = ranges[tile_id];
  num_splats_in_tile = range.y - range.x;
  bbm                = tile_id == 0 ? 0 : per_tile_bucket_offset[tile_id - 1];
  bucket_idx_in_tile = g_bucket - bbm;
  splat_idx_l        = bucket_idx_in_tile * 32 + warp.thread_rank();
  splat_idx_g        = range.x + splat_idx_l;
  valid_splat        = splat_idx_l < num_splats_in_tile;
  if (bucket_idx_in_tile * 32 >= max_contrib[tile_id]) return;

  int g_idx = 0;
  T2<T> xy  = {0};
  vec3<T> M1, M2, M3, Tw;
  T rgb[C] = {0};

  T normal[3], g_normal[3] = {}, acc_normal[3] = {};
  T opa;
  if (valid_splat) {
    g_idx = point_list[splat_idx_g];
    xy    = points_xy_image[g_idx];
    M1    = {inverse_M[9 * g_idx + 0], inverse_M[9 * g_idx + 1], inverse_M[9 * g_idx + 2]};
    M2    = {inverse_M[9 * g_idx + 3], inverse_M[9 * g_idx + 4], inverse_M[9 * g_idx + 5]};
    M3    = {inverse_M[9 * g_idx + 6], inverse_M[9 * g_idx + 7], inverse_M[9 * g_idx + 8]};
    Tw    = {transMats[9 * g_idx + 6], transMats[9 * g_idx + 7], transMats[9 * g_idx + 8]};
    for (int c = 0; c < C; ++c) rgb[c] = colors[g_idx * C + c];
    normal[0] = normal_opacity[g_idx].x;
    normal[1] = normal_opacity[g_idx].y;
    normal[2] = normal_opacity[g_idx].z;
    opa       = normal_opacity[g_idx].w;
  }
  vec3<T> g_M1, g_M2, g_M3, g_Tw;
  T g_rgb[C] = {}, g_opacity = 0;
  T2<T> g_means2D = {0, 0};

  const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
  const uint2 tile                 = {tile_id % horizontal_blocks, tile_id / horizontal_blocks};
  const uint2 pix_min              = {tile.x * BLOCK_X, tile.y * BLOCK_Y};

  T t = 0, t_final = 0, g_t_final = 0;
  int32_t last_contributor, median_contributor;
  T acc_rgb[C], g_pixel[C];

  // RENDER_AXUTILITY
  T dL_daux[8], acc_aux[8];
  T dL_dflow[2], acc_flow[2];
  const int NUM_AUX = dL_dout_flow == nullptr ? 8 : 10;

  // Flow
  T mat_t2[9];
  if (valid_splat && dL_dout_flow != nullptr) {
    for (int c = 0; c < 9; ++c) mat_t2[c] = Mats_t2[g_idx * 9 + c];
  }
  // interate over all pixels in this tile
  for (int i = 0; i < BLOCK_SIZE + 31; ++i) {
    // send data to next pixel
    t                = warp.shfl_up(t, 1);
    last_contributor = warp.shfl_up(last_contributor, 1);
    t_final          = warp.shfl_up(t_final, 1);
    g_t_final        = warp.shfl_up(g_t_final, 1);
    for (int c = 0; c < C; ++c) {
      acc_rgb[c] = warp.shfl_up(acc_rgb[c], 1);
      g_pixel[c] = warp.shfl_up(g_pixel[c], 1);
    }
    if constexpr (RENDER_AXUTILITY) {
      median_contributor = warp.shfl_up(median_contributor, 1);
      for (int c = 0; c < 8; ++c) {
        dL_daux[c] = warp.shfl_up(dL_daux[c], 1);
        acc_aux[c] = warp.shfl_up(acc_aux[c], 1);
      }
      if (dL_dout_flow != nullptr) {
        for (int c = 0; c < 2; ++c) {
          dL_dflow[c] = warp.shfl_up(dL_dflow[c], 1);
          acc_flow[c] = warp.shfl_up(acc_flow[c], 1);
        }
      }
    }

    // get index of pixel
    int idx               = i - warp.thread_rank();
    const uint2 pix       = {pix_min.x + idx % BLOCK_X, pix_min.y + idx / BLOCK_X};
    const uint32_t pix_id = W * pix.y + pix.x;
    const T2<T> pixf      = {(T) pix.x, (T) pix.y};
    bool valid_pixel      = pix.x < W && pix.y < H;

    // load data for every 32nd splat
    if (valid_splat && valid_pixel && warp.thread_rank() == 0 && idx < BLOCK_SIZE) {
      t = sampled_T[g_bucket * BLOCK_SIZE + idx];
      for (int c = 0; c < C; ++c) {
        acc_rgb[c] = -images[c * H * W + pix_id] + sampled_ar[(g_bucket * C + c) * BLOCK_SIZE + idx];
        g_pixel[c] = dL_dpixels[c * H * W + pix_id];
      }
      t_final          = 1 - out_opacity[pix_id];
      last_contributor = n_contrib[pix_id];
      g_t_final        = -dL_dout_opacity[pix_id];

      if constexpr (RENDER_AXUTILITY) {
        median_contributor = n_contrib[pix_id + H * W];
        for (int c = 0; c < 8; ++c) {
          dL_daux[c] = dL_dothers[pix_id + H * W * c];
          acc_aux[c] = -out_others[pix_id + H * W * c] + sampled_aux[(g_bucket * NUM_AUX + c) * BLOCK_SIZE + idx];
        }
        if (dL_dout_flow != nullptr) {
          for (int c = 0; c < 2; ++c) {
            dL_dflow[c] = dL_dout_flow[pix_id * 2 + c];
            acc_flow[c] = -out_flows[pix_id * 2 + c] + sampled_aux[(g_bucket * NUM_AUX + 8 + c) * BLOCK_SIZE + idx];
          }
        }
      }
    }

    // do work
    if (valid_splat && valid_pixel && 0 <= idx && idx < BLOCK_SIZE) {
      if (W <= pix.x || H <= pix.y) continue;
      if (splat_idx_l >= last_contributor) continue;
      const T2<T> d = {xy.x - pixf.x, xy.y - pixf.y};
      vec3<T> p     = {
          M1.x * pixf.x + M1.y * pixf.y + M1.z,
          M2.x * pixf.x + M2.y * pixf.y + M2.z,
          M3.x * pixf.x + M3.y * pixf.y + M3.z,
      };
      if (p.z == 0.0) continue;
      T2<T> s = {p.x / p.z, p.y / p.z};
      T rho3d = s.x * s.x + s.y * s.y;
      T rho2d = T(FilterInvSquare) * (d.x * d.x + d.y * d.y);
      T rho   = min(rho3d, rho2d);
      T depth = (s.x * Tw.x + s.y * Tw.y) + Tw.z;
      if (depth < near_n) continue;

      T power = -T(0.5) * rho;
      if (power > T(0.0)) continue;
      T G     = exp(power);
      T alpha = min(T(0.99), opa * G);
      if (alpha < T(1.0 / 255.0)) continue;
      T w = alpha * t;

      T g_alpha = 0;
      for (int c = 0; c < C; ++c) {
        acc_rgb[c] += w * rgb[c];
        g_rgb[c] += w * g_pixel[c];
        g_alpha += ((rgb[c] * t) + acc_rgb[c] / (T(1.) - alpha)) * g_pixel[c];
      }
      g_alpha += (-t_final / (1.0 - alpha)) * g_t_final;
      T dL_dweight = 0;
      T dL_dz      = 0;
      T dL_dsx = 0, dL_dsy = 0;
      if constexpr (RENDER_AXUTILITY) {
        const T m_d    = far_n / (far_n - near_n) * (1 - near_n / depth);
        const T dmd_dd = (far_n * near_n) / ((far_n - near_n) * depth * depth);
        // D += depth * w
        acc_aux[DEPTH_OFFSET] += w * depth;
        dL_dz += w * dL_daux[DEPTH_OFFSET];
        g_alpha += ((depth * t) + acc_aux[DEPTH_OFFSET] / (T(1.0) - alpha)) * dL_daux[DEPTH_OFFSET];
        if (splat_idx_l == median_contributor - 1) {
          dL_dz += dL_dothers[pix_id + H * W * MIDDEPTH_OFFSET];
        }
        // distortion += (m * m * A + M2 - 2 * m * M1) * w;
        // M1 += m * w;
        acc_aux[M1_OFFSET] += w * m_d;
        T g_m = w * dL_daux[M1_OFFSET];
        g_alpha += (m_d * t + acc_aux[M1_OFFSET] / (T(1.0) - alpha)) * dL_daux[M1_OFFSET];
        // M2 += m * m * w;
        acc_aux[M2_OFFSET] += w * m_d * m_d;
        g_m = T(2) * m_d * w * dL_daux[M2_OFFSET];
        g_alpha += (m_d * m_d * t + acc_aux[M2_OFFSET] / (T(1.0) - alpha)) * dL_daux[M2_OFFSET];
        dL_dz += dmd_dd * g_m;
        // N[ch] += normal[ch] * w;
        for (int c = 0; c < 3; ++c) {
          acc_aux[NORMAL_OFFSET + c] += w * normal[c];
          g_normal[c] += w * dL_daux[NORMAL_OFFSET + c];
          g_alpha += (normal[c] * t + acc_aux[NORMAL_OFFSET + c] / (T(1.0) - alpha)) * dL_daux[NORMAL_OFFSET + c];
        }
      }
      if (dL_dout_flow != nullptr) {
        T u_t2     = mat_t2[0] * s.x + mat_t2[1] * s.y + mat_t2[2];
        T v_t2     = mat_t2[3] * s.x + mat_t2[4] * s.y + mat_t2[5];
        T w_t2     = mat_t2[6] * s.x + mat_t2[7] * s.y + mat_t2[8];
        w_t2       = abs(w_t2) < 1e-5 ? 1e-5 : w_t2;  // avoid divide 0
        T2<T> flow = {(u_t2 / w_t2 - pixf.x), (v_t2 / w_t2 - pixf.y)};
        acc_flow[0] += w * flow.x;
        acc_flow[1] += w * flow.y;
        g_alpha += (flow.x * t + acc_flow[0] / (T(1.0) - alpha)) * dL_dflow[0];
        g_alpha += (flow.y * t + acc_flow[1] / (T(1.0) - alpha)) * dL_dflow[1];
        T gu = w / w_t2 * dL_dflow[0], gv = w / w_t2 * dL_dflow[1],
          gw = -w / (w_t2 * w_t2) * (u_t2 * dL_dflow[0] + v_t2 * dL_dflow[1]);
        dL_dsx += mat_t2[0] * gu + mat_t2[3] * gv + mat_t2[6] * gw;
        dL_dsy += mat_t2[1] * gu + mat_t2[4] * gv + mat_t2[7] * gw;
      }
      // if (pix_id == 127) {
      //   printf("\033[33mgs=%d, xy=%f, %f, rho=%f, alpha=%f, sigma=%f\033[0m\n", g_idx, s.x, s.y, rho, alpha, w);
      // }
      t *= (1 - alpha);
      const T dL_dG = opa * g_alpha;
      if (rho3d <= rho2d) {
        dL_dsx += dL_dz * Tw.x - G * dL_dG * s.x;
        dL_dsy += dL_dz * Tw.y - G * dL_dG * s.y;
        const T dsx_pz      = dL_dsx / p.z;
        const T dsy_pz      = dL_dsy / p.z;
        const vec3<T> dL_dp = {dsx_pz, dsy_pz, -(dsx_pz * s.x + dsy_pz * s.y)};

        g_M1.x += dL_dp.x * pixf.x;
        g_M1.y += dL_dp.x * pixf.y;
        g_M1.z += dL_dp.x;
        g_M2.x += dL_dp.y * pixf.x;
        g_M2.y += dL_dp.y * pixf.y;
        g_M2.z += dL_dp.y;
        g_M3.x += dL_dp.z * pixf.x;
        g_M3.y += dL_dp.z * pixf.y;
        g_M3.z += dL_dp.z;
      } else {
        const T dG_ddelx = -G * FilterInvSquare * d.x;
        const T dG_ddely = -G * FilterInvSquare * d.y;
        g_means2D.x += dL_dG * dG_ddelx;
        g_means2D.y += dL_dG * dG_ddely;
      }
      g_Tw.x += dL_dz * s.x;
      g_Tw.y += dL_dz * s.y;
      g_Tw.z += dL_dz;
      g_opacity += G * g_alpha;
    }
  }
  // write final results
  if (valid_splat) {
    atomicAdd(&dL_dmean2D[g_idx].x, g_means2D.x);
    atomicAdd(&dL_dmean2D[g_idx].y, g_means2D.y);
    if constexpr (RENDER_AXUTILITY) {
      atomicAdd(&dL_dnormal_dopacity[g_idx * 4 + 0], g_normal[0]);
      atomicAdd(&dL_dnormal_dopacity[g_idx * 4 + 1], g_normal[1]);
      atomicAdd(&dL_dnormal_dopacity[g_idx * 4 + 2], g_normal[2]);
    }
    atomicAdd(&dL_dnormal_dopacity[g_idx * 4 + 3], g_opacity);
    for (int c = 0; c < C; ++c) atomicAdd(&dL_dcolors[g_idx * C + c], g_rgb[c]);
    atomicAdd(&dL_dtransMat[g_idx * 9 + 6], g_Tw.x);
    atomicAdd(&dL_dtransMat[g_idx * 9 + 7], g_Tw.y);
    atomicAdd(&dL_dtransMat[g_idx * 9 + 8], g_Tw.z);
    atomicAdd(&dL_dinverse_M[g_idx * 9 + 0], g_M1.x);
    atomicAdd(&dL_dinverse_M[g_idx * 9 + 1], g_M1.y);
    atomicAdd(&dL_dinverse_M[g_idx * 9 + 2], g_M1.z);
    atomicAdd(&dL_dinverse_M[g_idx * 9 + 3], g_M2.x);
    atomicAdd(&dL_dinverse_M[g_idx * 9 + 4], g_M2.y);
    atomicAdd(&dL_dinverse_M[g_idx * 9 + 5], g_M2.z);
    atomicAdd(&dL_dinverse_M[g_idx * 9 + 6], g_M3.x);
    atomicAdd(&dL_dinverse_M[g_idx * 9 + 7], g_M3.y);
    atomicAdd(&dL_dinverse_M[g_idx * 9 + 8], g_M3.z);
  }
}

vector<Tensor> GS_2D_fast_render_forward(int width, int height, double near_n, double far_n, bool only_image,
    Tensor& means2D, const Tensor& colors, Tensor& normal_opacity, Tensor& trans_mat, Tensor& inverse_m,
    const Tensor& point_list, const Tensor& ranges, torch::optional<Tensor>& trans_mat_t2,
    torch::optional<Tensor>& accum_max_count, torch::optional<Tensor>& accum_weights_p,
    torch::optional<Tensor>& accum_weights_count, torch::optional<Tensor>& per_tile_bucket_offset) {
  CHECK_INPUT(colors);
  CHECK_INPUT(trans_mat);
  CHECK_NDIM(colors, 2);
  int P = colors.size(0);  // num points
  CHECK_SHAPE(trans_mat, P, 3, 3);
  CHECK_SHAPE(inverse_m, P, 3, 3);
  CHECK_INPUT_AND_TYPE(ranges, torch::kLong);
  CHECK_INPUT_AND_TYPE(point_list, torch::kInt32);

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

  Tensor pixel_colors  = torch::zeros({3, height, width}, colors.options());
  Tensor pixel_opacity = torch::zeros({height, width}, colors.options());
  Tensor n_contrib     = torch::zeros({2, height, width}, colors.options().dtype(torch::kInt32));
  Tensor pixel_extras, pixel_flow;
  if (!only_image) pixel_extras = torch::zeros({8, height, width}, colors.options());
  if (!only_image && trans_mat_t2.has_value()) pixel_flow = torch::zeros({height, width, 2}, colors.options());
  vector<Tensor> outputs = {pixel_colors, pixel_opacity, pixel_extras, pixel_flow, n_contrib};
  Tensor max_contrib, sampled_T, sampled_acc, sampled_aux, bucket_to_tile;
  bool is_fast_bwd = per_tile_bucket_offset.has_value();
  if (is_fast_bwd) {
    int B       = per_tile_bucket_offset.value()[-1].item<int64_t>();
    max_contrib = torch::zeros(
        {(height + BLOCK_Y - 1) / BLOCK_Y, (width + BLOCK_X - 1) / BLOCK_X}, colors.options().dtype(torch::kInt32));
    bucket_to_tile = torch::empty({B}, colors.options().dtype(torch::kInt32));
    sampled_T      = torch::empty({B, BLOCK_SIZE}, colors.options());
    sampled_acc    = torch::empty({B, NUM_CHANNELS, BLOCK_SIZE}, colors.options());
    sampled_aux    = torch::empty({B, trans_mat_t2.has_value() ? 10 : 8, BLOCK_SIZE}, colors.options());
    outputs.push_back(max_contrib);
    outputs.push_back(bucket_to_tile);
    outputs.push_back(sampled_T);
    outputs.push_back(sampled_acc);
    outputs.push_back(sampled_aux);
  }
  if (P == 0) return outputs;

  const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
  const dim3 block(BLOCK_X, BLOCK_Y, 1);

  //   AT_DISPATCH_FLOATING_TYPES(means2D.scalar_type(), "gaussian_rasterize_forward", [&] {
  /*

    const T2<int64_t>* __restrict__ ranges, const int32_t* __restrict__ point_list,
    const T2<T>* __restrict__ points_xy_image, const T* __restrict__ features, const T* __restrict__ transMats,
    const T4<T>* __restrict__ normal_opacity,


    int32_t* __restrict__ n_contrib, T* __restrict__ out_color, T* __restrict__ out_opacity,
    T* __restrict__ out_others
    */
  using scalar_t = float;
  switch (only_image) {
    case false:
      gs_2d_fast_render_forward_kernel<scalar_t, NUM_CHANNELS, true> KERNEL_ARG(tile_grid, block)(
          // constant
          width, height, (scalar_t) near_n, (scalar_t) far_n,
          // inputs
          (T2<int64_t>*) ranges.contiguous().data_ptr<int64_t>(), point_list.contiguous().data_ptr<int32_t>(),
          (T2<scalar_t>*) means2D.contiguous().data_ptr<scalar_t>(), colors.contiguous().data_ptr<scalar_t>(),
          trans_mat.data_ptr<scalar_t>(), inverse_m.data_ptr<scalar_t>(),
          (T4<scalar_t>*) normal_opacity.contiguous().data_ptr<scalar_t>(),
          trans_mat_t2.has_value() ? trans_mat_t2.value().data_ptr<scalar_t>() : nullptr,
          // outputs
          n_contrib.data_ptr<int32_t>(), pixel_colors.data_ptr<scalar_t>(), pixel_opacity.data_ptr<scalar_t>(),
          pixel_extras.data_ptr<scalar_t>(),                                                       //
          (T2<scalar_t>*) (trans_mat_t2.has_value() ? pixel_flow.data_ptr<scalar_t>() : nullptr),  //
          is_fast_bwd ? per_tile_bucket_offset.value().data_ptr<int64_t>() : nullptr,              //
          is_fast_bwd ? bucket_to_tile.data_ptr<int32_t>() : nullptr,                              //
          is_fast_bwd ? sampled_T.data_ptr<scalar_t>() : nullptr,                                  //
          is_fast_bwd ? sampled_acc.data_ptr<scalar_t>() : nullptr,                                //
          is_fast_bwd ? sampled_aux.data_ptr<scalar_t>() : nullptr,                                //
          is_fast_bwd ? max_contrib.data_ptr<int32_t>() : nullptr,                                 //
          accum_max_count.has_value() ? accum_max_count.value().data_ptr<int32_t>() : nullptr,     //
          accum_weights_p.has_value() ? accum_weights_p.value().data_ptr<scalar_t>() : nullptr,    //
          accum_weights_count.has_value() ? accum_weights_count.value().data_ptr<int32_t>() : nullptr);
      break;
    default:
      gs_2d_fast_render_forward_kernel<scalar_t, NUM_CHANNELS, false> KERNEL_ARG(tile_grid, block)(
          // constant
          width, height, (scalar_t) near_n, (scalar_t) far_n,
          // inputs
          (T2<int64_t>*) ranges.contiguous().data_ptr<int64_t>(), point_list.contiguous().data_ptr<int32_t>(),
          (T2<scalar_t>*) means2D.contiguous().data_ptr<scalar_t>(), colors.contiguous().data_ptr<scalar_t>(),
          trans_mat.data_ptr<scalar_t>(), inverse_m.data_ptr<scalar_t>(),
          (T4<scalar_t>*) normal_opacity.contiguous().data_ptr<scalar_t>(), nullptr,
          // outputs
          n_contrib.data_ptr<int32_t>(), pixel_colors.data_ptr<scalar_t>(), pixel_opacity.data_ptr<scalar_t>(), nullptr,
          nullptr,
          // for fast backward
          nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
          //
          nullptr, nullptr, nullptr);
      break;
  }
  // cudaDeviceSynchronize();
  CHECK_CUDA_ERROR("gs_2d_render_forward_kernel");
  //   });
  return outputs;
}

vector<Tensor> GS_2D_fast_render_backward(const Tensor& means2D, const Tensor& normal_opacity, const Tensor& colors,
    const Tensor& trans_mat, const Tensor& inverse_m, const torch::optional<Tensor>& trans_mat2,
    // outputs
    const Tensor& images, const Tensor& out_opacity, const torch::optional<Tensor>& out_others,
    const torch::optional<Tensor>& out_flows,
    // grad outputs
    const Tensor& dL_dout_color, const Tensor& dL_dout_opacity, const torch::optional<Tensor>& dL_dothers,
    const torch::optional<Tensor>& dL_dout_flow,
    // aux
    const Tensor& ranges, const Tensor& point_list, const Tensor& n_contrib, double near_n, double far_n,
    // for fast backward
    const torch::optional<Tensor>& per_tile_bucket_offset, const torch::optional<Tensor>& bucket_to_tile,
    const torch::optional<Tensor>& max_contrib, torch::optional<Tensor>& sampled_T,
    torch::optional<Tensor>& sampled_acc, torch::optional<Tensor>& sampled_aux) {
  CHECK_CUDA(dL_dout_color);
  CHECK_CUDA(dL_dout_opacity);
  if (dL_dothers.has_value()) CHECK_CUDA(dL_dothers.value());

  const int P = means2D.size(0);
  const int H = dL_dout_color.size(1);
  const int W = dL_dout_color.size(2);
  int64_t R   = point_list.size(0);

  auto options       = out_opacity.options();
  Tensor dL_dmeans2D = torch::zeros({P, 2}, options);
  Tensor dL_dnorm_op = torch::zeros({P, 4}, options);
  Tensor dL_dcolors  = torch::zeros({P, NUM_CHANNELS}, options);
  Tensor dL_dtrans_m = torch::zeros({P, 3, 3}, options);
  Tensor dL_dinver_m = torch::zeros({P, 3, 3}, options);

  if (P == 0) return {dL_dcolors, dL_dmeans2D, dL_dtrans_m, dL_dinver_m, dL_dnorm_op};

  const dim3 tile_grid((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, 1);
  const dim3 block(BLOCK_X, BLOCK_Y, 1);
  bool is_fast_bwd = per_tile_bucket_offset.has_value();
  uint32_t B       = is_fast_bwd ? per_tile_bucket_offset.value()[-1].item<int64_t>() : 0;
  // AT_DISPATCH_FLOATING_TYPES(means2D.scalar_type(), "gs_2d_render_backward_kernel", [&] {

  using scalar_t = float;
  switch (int(is_fast_bwd) * 2 + int(out_others.has_value() && dL_dothers.has_value())) {
    case 0:
      gs_2d_fast_render_backward_kernel<scalar_t, NUM_CHANNELS, false> KERNEL_ARG(tile_grid, block)(
          // const
          P, W, H, R, (scalar_t) near_n, (scalar_t) far_n,
          // aux inputs
          (T2<int64_t>*) ranges.contiguous().data_ptr<int64_t>(), point_list.contiguous().data_ptr<int32_t>(),
          n_contrib.contiguous().data_ptr<int32_t>(),
          // inputs
          (T2<scalar_t>*) means2D.contiguous().data_ptr<scalar_t>(),
          (T4<scalar_t>*) normal_opacity.contiguous().data_ptr<scalar_t>(), colors.contiguous().data_ptr<scalar_t>(),
          trans_mat.data_ptr<scalar_t>(), inverse_m.data_ptr<scalar_t>(),
          dL_dout_flow.has_value() ? trans_mat2.value().data_ptr<scalar_t>() : nullptr,
          // outputs
          out_opacity.contiguous().data<scalar_t>(), nullptr,
          // grad_outputs
          dL_dout_color.contiguous().data<scalar_t>(),
          dL_dothers.has_value() ? dL_dothers.value().contiguous().data<scalar_t>() : nullptr,
          dL_dout_opacity.contiguous().data<scalar_t>(),
          dL_dout_flow.has_value() ? (T2<scalar_t>*) dL_dout_flow.value().data_ptr<scalar_t>() : nullptr,
          // grad_inputs
          (T2<scalar_t>*) dL_dmeans2D.contiguous().data<scalar_t>(), dL_dnorm_op.contiguous().data<scalar_t>(),
          dL_dcolors.contiguous().data<scalar_t>(), dL_dtrans_m.data<scalar_t>(), dL_dinver_m.data_ptr<scalar_t>());
    case 1:
      gs_2d_fast_render_backward_kernel<scalar_t, NUM_CHANNELS, true> KERNEL_ARG(tile_grid, block)(
          // connst
          P, W, H, R, (scalar_t) near_n, (scalar_t) far_n,
          // aux inputs
          (T2<int64_t>*) ranges.contiguous().data_ptr<int64_t>(), point_list.contiguous().data_ptr<int32_t>(),
          n_contrib.contiguous().data_ptr<int32_t>(),
          // inputs
          (T2<scalar_t>*) means2D.contiguous().data_ptr<scalar_t>(),
          (T4<scalar_t>*) normal_opacity.contiguous().data_ptr<scalar_t>(), colors.contiguous().data_ptr<scalar_t>(),
          trans_mat.data_ptr<scalar_t>(), inverse_m.data_ptr<scalar_t>(),
          dL_dout_flow.has_value() ? trans_mat2.value().data_ptr<scalar_t>() : nullptr,
          // outputs
          out_opacity.contiguous().data<scalar_t>(), out_others.value().data<scalar_t>(),
          // grad_outputs
          dL_dout_color.contiguous().data<scalar_t>(),
          dL_dothers.has_value() ? dL_dothers.value().contiguous().data<scalar_t>() : nullptr,
          dL_dout_opacity.contiguous().data<scalar_t>(),
          dL_dout_flow.has_value() ? (T2<scalar_t>*) dL_dout_flow.value().data_ptr<scalar_t>() : nullptr,
          // grad_inputs
          (T2<scalar_t>*) dL_dmeans2D.contiguous().data<scalar_t>(), dL_dnorm_op.contiguous().data<scalar_t>(),
          dL_dcolors.contiguous().data<scalar_t>(), dL_dtrans_m.data<scalar_t>(), dL_dinver_m.data_ptr<scalar_t>());
      break;

    default:
      gs_2d_fast_render_backward_v2_kernel<scalar_t, NUM_CHANNELS, true> KERNEL_ARG(B, 32)(
          // const value
          P, W, H, R, B, (scalar_t) near_n, (scalar_t) far_n,
          // aux inputs
          (T2<int64_t>*) ranges.contiguous().data_ptr<int64_t>(), point_list.contiguous().data_ptr<int32_t>(),
          n_contrib.contiguous().data_ptr<int32_t>(), max_contrib.value().contiguous().data_ptr<int32_t>(),
          per_tile_bucket_offset.value().contiguous().data_ptr<int64_t>(),
          bucket_to_tile.value().contiguous().data_ptr<int32_t>(), sampled_T.value().contiguous().data_ptr<scalar_t>(),
          sampled_acc.value().contiguous().data_ptr<scalar_t>(), sampled_aux.value().contiguous().data_ptr<scalar_t>(),
          // inputs
          (T2<scalar_t>*) means2D.contiguous().data_ptr<scalar_t>(),
          (T4<scalar_t>*) normal_opacity.contiguous().data_ptr<scalar_t>(), colors.contiguous().data_ptr<scalar_t>(),
          trans_mat.data_ptr<scalar_t>(), inverse_m.data_ptr<scalar_t>(),
          dL_dout_flow.has_value() ? trans_mat2.value().data_ptr<scalar_t>() : nullptr,
          // outputs
          images.contiguous().data<scalar_t>(), out_opacity.contiguous().data<scalar_t>(),
          dL_dothers.has_value() ? out_others.value().contiguous().data<scalar_t>() : nullptr,
          dL_dout_flow.has_value() ? out_flows.value().data_ptr<scalar_t>() : nullptr,
          // grad_outputs
          dL_dout_color.contiguous().data<scalar_t>(),
          dL_dothers.has_value() ? dL_dothers.value().contiguous().data<scalar_t>() : nullptr,
          dL_dout_opacity.contiguous().data<scalar_t>(),
          dL_dout_flow.has_value() ? dL_dout_flow.value().data_ptr<scalar_t>() : nullptr,
          // grad_inputs
          (T2<scalar_t>*) dL_dmeans2D.contiguous().data<scalar_t>(), dL_dnorm_op.contiguous().data<scalar_t>(),
          dL_dcolors.contiguous().data<scalar_t>(), dL_dtrans_m.data<scalar_t>(), dL_dinver_m.data_ptr<scalar_t>());
      break;
  }
  // cudaDeviceSynchronize();
  CHECK_CUDA_ERROR("gs_2d_render_backward_kernel");
  // });
  return {dL_dcolors, dL_dmeans2D, dL_dtrans_m, dL_dinver_m, dL_dnorm_op};
}

REGIST_PYTORCH_EXTENSION(gs_2d_fast_render, {
  m.def("GS_2D_fast_render_forward", &GS_2D_fast_render_forward, "GS_2D_fast_render_forward (CUDA)");
  m.def("GS_2D_fast_render_backward", &GS_2D_fast_render_backward, "GS_2D_fast_render_backward (CUDA)");
})

}  // namespace GaussianRasterizer
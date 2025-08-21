/*
paper: 2D Gaussian Splatting for Geometrically Accurate Radiance Fields, SIGGRAPH 2024
code: https://github.com/hbb1/2d-gaussian-splatting
 */

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
#define SUM_W_OFFSET 8
#define DETACH_WEIGHT 0

constexpr double FilterInvSquare = 2.0;

template <typename T, uint32_t CHANNELS, bool RENDER_AXUTILITY = false>
__global__ void __launch_bounds__(BLOCK_X* BLOCK_Y) gs_2d_render_forward_kernel(
    // scalar
    int W, int H, T near_n, T far_n,
    // inputs
    const T2<int64_t>* __restrict__ ranges, const int32_t* __restrict__ point_list,
    const T2<T>* __restrict__ points_xy_image, const T* __restrict__ features, const T* __restrict__ transMats,
    const T4<T>* __restrict__ normal_opacity, const T* __restrict__ transMats_t2,
    // outputs
    int32_t* __restrict__ n_contrib, T* __restrict__ out_color, T* __restrict__ out_opacity, T* __restrict__ out_others,
    T2<T>* __restrict__ out_flow, int32_t* __restrict__ accum_max_count = nullptr,
    T* __restrict__ accum_weights_p = nullptr, int32_t* __restrict__ accum_weights_count = nullptr) {
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
  __shared__ T collected_M[BLOCK_SIZE * 9];

  // Initialize helper variables
  T t                       = T(1.0);
  uint32_t contributor      = 0;
  uint32_t last_contributor = 0;
  T C[CHANNELS]             = {0};

  T weight_max          = 0;
  int32_t weight_max_id = -1;
  bool flag_update      = false;

  // render axutility ouput
  T N[3]         = {0};
  T D            = {0};
  T M1           = {0};
  T M2           = {0};
  T distortion   = {0};
  T median_depth = {0};
  T weight_sum   = 0;
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
      collected_Tu[block.thread_rank()]             = {
          transMats[9 * coll_id + 0], transMats[9 * coll_id + 1], transMats[9 * coll_id + 2]};
      collected_Tv[block.thread_rank()] = {
          transMats[9 * coll_id + 3], transMats[9 * coll_id + 4], transMats[9 * coll_id + 5]};
      collected_Tw[block.thread_rank()] = {
          transMats[9 * coll_id + 6], transMats[9 * coll_id + 7], transMats[9 * coll_id + 8]};
      if constexpr (RENDER_AXUTILITY) {
        if (transMats_t2 != nullptr) {
#pragma unroll
          for (int k = 0; k < 9; ++k) collected_M[block.thread_rank() * 9 + k] = transMats_t2[coll_id * 9 + k];
        }
      }
    }
    block.sync();

    // Iterate over current batch
    for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++) {
      // Keep track of current position in range
      contributor++;

      // Fisrt compute two homogeneous planes, See Eq. (8)
      const T2<T> xy   = collected_xy[j];
      const vec3<T> Tu = collected_Tu[j];
      const vec3<T> Tv = collected_Tv[j];
      const vec3<T> Tw = collected_Tw[j];
      vec3<T> k        = pixf.x * Tw - Tu;
      vec3<T> l        = pixf.y * Tw - Tv;
      vec3<T> p        = k.cross(l);
      if (p.z == 0.0) continue;
      // Perspective division to get the intersection (u,v), Eq. (10)
      T2<T> s = {p.x / p.z, p.y / p.z};
      T rho3d = (s.x * s.x + s.y * s.y);
      // Add low pass filter
      T2<T> d = {xy.x - pixf.x, xy.y - pixf.y};
      T rho2d = T(FilterInvSquare) * (d.x * d.x + d.y * d.y);

      // compute intersection and depth
      // T rho   = rho3d;
      T rho   = min(rho3d, rho2d);
      T depth = (s.x * Tw.x + s.y * Tw.y) + Tw.z;
      // T depth = (rho3d <= rho2d) ? (s.x * Tw.x + s.y * Tw.y) + Tw.z : Tw.z;
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
        if (transMats_t2 != nullptr) {
          T u_t2 = collected_M[j * 9 + 0] * s.x + collected_M[j * 9 + 1] * s.y + collected_M[j * 9 + 2];
          T v_t2 = collected_M[j * 9 + 3] * s.x + collected_M[j * 9 + 4] * s.y + collected_M[j * 9 + 5];
          T w_t2 = collected_M[j * 9 + 6] * s.x + collected_M[j * 9 + 7] * s.y + collected_M[j * 9 + 8];
          w_t2   = abs(w_t2) < 1e-5 ? 1e-5 : w_t2;  // avoid divide 0
          flow.x += (u_t2 / w_t2 - pixf.x) * w;
          flow.y += (v_t2 / w_t2 - pixf.y) * w;
        }
        weight_sum += w;
      }
      if (flag_update && accum_max_count) atomicAdd(&(accum_max_count[weight_max_id]), 1);

      // Eq. (3) from 3D Gaussian splatting paper.
      for (int ch = 0; ch < CHANNELS; ch++) C[ch] += features[collected_id[j] * CHANNELS + ch] * w;
      t = test_T;

      // Keep track of last range entry to update this pixel.
      last_contributor = contributor;
      // if (pix_id == 15) {
      //   printf("\033[31mgs=%d, xy=%f, %f, rho=%f, depth=%f, alpha=%f, sigma=%f\033[0m\n", collected_id[j], s.x, s.y,
      //       rho, depth, alpha, w);
      // }
    }
  }

  // All threads that treat valid pixel write out their final rendering data to the frame and auxiliary buffers.
  if (inside) {
    out_opacity[pix_id] = T(1) - t;
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
      out_others[pix_id + H * W * M1_OFFSET]    = M1;
      out_others[pix_id + H * W * M2_OFFSET]    = M2;
      out_others[pix_id + H * W * SUM_W_OFFSET] = weight_sum;

      if (out_flow != nullptr) {
        weight_sum       = weight_sum == 0 ? T(1e-7) : weight_sum;
        out_flow[pix_id] = {flow.x / weight_sum, flow.y / weight_sum};
      }
    }
  }
}

template <typename T = float, uint32_t C = NUM_CHANNELS, bool RENDER_AXUTILITY = true>
__global__ void __launch_bounds__(BLOCK_X* BLOCK_Y) gs_2d_render_backward_kernel(
    // scalaes
    int P, int W, int H, int64_t R, T near_n, T far_n,
    // aux
    const T2<int64_t>* __restrict__ ranges, const int32_t* __restrict__ point_list,
    const int32_t* __restrict__ n_contrib,
    // inputs
    const T2<T>* __restrict__ points_xy_image, const T4<T>* __restrict__ normal_opacity, const T* __restrict__ colors,
    const T* __restrict__ transMats, const T* __restrict__ transMats_t2,
    // outputs
    const T* __restrict__ out_opacity, const T* __restrict__ out_others, const T2<T>* __restrict__ out_flows,
    // grad outputs
    const T* __restrict__ dL_dpixels, const T* __restrict__ dL_dothers, const T* __restrict__ dL_dout_opacity,
    const T2<T>* __restrict__ dL_dout_flows,
    // grad inputs
    T2<T>* __restrict__ dL_dmean2D, T* __restrict__ dL_dnormal_dopacity, T* __restrict__ dL_dcolors,
    T* __restrict__ dL_dtransMat) {
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
  __shared__ vec3<T> collected_Tu[BLOCK_SIZE];
  __shared__ vec3<T> collected_Tv[BLOCK_SIZE];
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
  const T sum_w    = (RENDER_AXUTILITY & inside) ? out_others[pix_id + H * W * SUM_W_OFFSET] : 0;

  T2<T> accum_flow = {0}, last_flow = {0}, dL_dflow;
  T dL_dsum_w = 0, acc_w = 0;
  if (inside && dL_dout_flows != nullptr) {
    dL_dflow  = dL_dout_flows[pix_id];
    dL_dsum_w = -(dL_dflow.x * out_flows[pix_id].x + dL_dflow.y * out_flows[pix_id].y) / (sum_w * sum_w);
    dL_dflow  = {dL_dflow.x / sum_w, dL_dflow.y / sum_w};
  }

  if (inside) {
    for (int i = 0; i < C; i++) dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];
  }

  T last_alpha    = 0;
  T last_color[C] = {0};

  // Gradient of pixel coordinate w.r.t. normalized screen-space viewport corrdinates (-1 to 1)
  const T ddelx_dx = 0.5 * W;
  const T ddely_dy = 0.5 * H;

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
      collected_Tu[block.thread_rank()]             = {
          transMats[9 * coll_id + 0], transMats[9 * coll_id + 1], transMats[9 * coll_id + 2]};
      collected_Tv[block.thread_rank()] = {
          transMats[9 * coll_id + 3], transMats[9 * coll_id + 4], transMats[9 * coll_id + 5]};
      collected_Tw[block.thread_rank()] = {
          transMats[9 * coll_id + 6], transMats[9 * coll_id + 7], transMats[9 * coll_id + 8]};
      for (int c = 0; c < C; c++) collected_colors[c * BLOCK_SIZE + block.thread_rank()] = colors[coll_id * C + c];
      // collected_depths[block.thread_rank()] = depths[coll_id];
      if (dL_dout_flows != nullptr)
        for (int c = 0; c < 9; ++c) collected_M[block.thread_rank() * 9 + c] = transMats_t2[coll_id * 9 + c];
    }
    block.sync();

    // Iterate over Gaussians
    for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++) {
      // Keep track of current Gaussian ID. Skip, if this one is behind the last contributor for this pixel.
      contributor--;
      if (contributor >= last_contributor) continue;

      // compute ray-splat intersection as before Fisrt compute two homogeneous planes, See Eq. (8)
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
      T rho2d = FilterInvSquare * (d.x * d.x + d.y * d.y);

      // compute intersection and depth
      T rho = min(rho3d, rho2d);
      T c_d = (s.x * Tw.x + s.y * Tw.y) + Tw.z;
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
      T2<T> dL_ds  = {0, 0};
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
          atomicAdd((&dL_dnormal_dopacity[global_id * 4 + ch]), alpha * t * dL_dnormal2D[ch]);
        }
        dL_dz += alpha * t * dL_ddepth;
        dL_ds = {dL_dz * Tw.x, dL_dz * Tw.y};
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
        T gu = w / w_t2 * dL_dflow.x;
        T gv = w / w_t2 * dL_dflow.y;
        T gw = -w / (w_t2 * w_t2) * (u_t2 * dL_dflow.x + v_t2 * dL_dflow.y);
        dL_ds.x += collected_M[j * 9 + 0] * gu + collected_M[j * 9 + 3] * gv + collected_M[j * 9 + 6] * gw;
        dL_ds.y += collected_M[j * 9 + 1] * gu + collected_M[j * 9 + 4] * gv + collected_M[j * 9 + 7] * gw;
      }

      acc_w = last_alpha + (T(1.0) - last_alpha) * acc_w;
      dL_dalpha += (T(1) - acc_w) * dL_dsum_w;
      dL_dalpha *= t;
      // if (pix_id == 761) {
      //   printf("\033[31mgs=%d, xy=%f, %f, rho=%f, depth=%f, alpha=%f, sigma=%f, dL_dalpha=%f\033[0m\n",
      //   collected_id[j],
      //       xy.x, xy.y, rho, c_d, alpha, w, dL_dalpha);
      // }
      // Update last alpha (to be used in the next iteration)
      last_alpha = alpha;

      // Account for fact that alpha also influences how much of the background color is added if nothing left to blend
      //   T bg_dot_dpixel = 0;
      //   for (int i = 0; i < C; i++) bg_dot_dpixel += bg_color[i] * dL_dpixel[i];
      //   dL_dalpha += (-t_final / (T(1.)- alpha)) * bg_dot_dpixel;

      // Helpful reusable temporary variables
      const T dL_dG = nor_o.w * dL_dalpha;
      if (rho3d <= rho2d) {
        // Update gradients w.r.t. covariance of Gaussian 3x3 (T)
        dL_ds               = {dL_ds.x + dL_dG * -G * s.x, dL_ds.y + dL_dG * -G * s.y};
        const T3<T> dz_dTw  = {s.x, s.y, 1.0};
        const T dsx_pz      = dL_ds.x / p.z;
        const T dsy_pz      = dL_ds.y / p.z;
        const vec3<T> dL_dp = {dsx_pz, dsy_pz, -(dsx_pz * s.x + dsy_pz * s.y)};
        const T3<T> dL_dk   = l.cross(dL_dp);
        const T3<T> dL_dl   = dL_dp.cross(k);

        const T3<T> dL_dTu = {-dL_dk.x, -dL_dk.y, -dL_dk.z};
        const T3<T> dL_dTv = {-dL_dl.x, -dL_dl.y, -dL_dl.z};
        const T3<T> dL_dTw = {pixf.x * dL_dk.x + pixf.y * dL_dl.x + dL_dz * dz_dTw.x,
            pixf.x * dL_dk.y + pixf.y * dL_dl.y + dL_dz * dz_dTw.y,
            pixf.x * dL_dk.z + pixf.y * dL_dl.z + dL_dz * dz_dTw.z};

        // Update gradients w.r.t. 3D covariance (3x3 matrix)
        atomicAdd(&dL_dtransMat[global_id * 9 + 0], dL_dTu.x);
        atomicAdd(&dL_dtransMat[global_id * 9 + 1], dL_dTu.y);
        atomicAdd(&dL_dtransMat[global_id * 9 + 2], dL_dTu.z);
        atomicAdd(&dL_dtransMat[global_id * 9 + 3], dL_dTv.x);
        atomicAdd(&dL_dtransMat[global_id * 9 + 4], dL_dTv.y);
        atomicAdd(&dL_dtransMat[global_id * 9 + 5], dL_dTv.z);
        atomicAdd(&dL_dtransMat[global_id * 9 + 6], dL_dTw.x);
        atomicAdd(&dL_dtransMat[global_id * 9 + 7], dL_dTw.y);
        atomicAdd(&dL_dtransMat[global_id * 9 + 8], dL_dTw.z);
      } else {
        // Update gradients w.r.t. center of Gaussian 2D mean position
        const T dG_ddelx = -G * FilterInvSquare * d.x;
        const T dG_ddely = -G * FilterInvSquare * d.y;
        atomicAdd(&dL_dmean2D[global_id].x, dL_dG * dG_ddelx);  // not scaled
        atomicAdd(&dL_dmean2D[global_id].y, dL_dG * dG_ddely);  // not scaled
        atomicAdd(&dL_dtransMat[global_id * 9 + 6], dL_dz * s.x);
        atomicAdd(&dL_dtransMat[global_id * 9 + 7], dL_dz * s.y);
        atomicAdd(&dL_dtransMat[global_id * 9 + 8], dL_dz);  // propagate depth loss
      }

      // Update gradients w.r.t. opacity of the Gaussian
      atomicAdd(&(dL_dnormal_dopacity[global_id * 4 + 3]), G * dL_dalpha);
    }
  }
}

std::tuple<Tensor, Tensor, torch::optional<Tensor>, torch::optional<Tensor>, Tensor> GS_2D_render_forward(int width,
    int height, Tensor& means2D, const Tensor& colors, Tensor& normal_opacity, Tensor& trans_mat,
    const Tensor& point_list, const Tensor& ranges, double near_n, double far_n, bool only_image,
    torch::optional<Tensor>& trans_mat_t2, torch::optional<Tensor>& accum_max_count,
    torch::optional<Tensor>& accum_weights_p, torch::optional<Tensor>& accum_weights_count) {
  CHECK_INPUT(colors);
  CHECK_INPUT(trans_mat);
  CHECK_NDIM(colors, 2);
  int P = colors.size(0);  // num points
  CHECK_SHAPE(trans_mat, P, 3, 3);
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
  if (trans_mat_t2.has_value()) {
    CHECK_INPUT(trans_mat_t2.value());
    CHECK_SHAPE(trans_mat_t2.value(), {P, 3, 3});
  }
  Tensor pixel_colors  = torch::zeros({3, height, width}, colors.options());
  Tensor pixel_opacity = torch::zeros({height, width}, colors.options());
  Tensor n_contrib     = torch::zeros({2, height, width}, colors.options().dtype(torch::kInt32));
  torch::optional<Tensor> pixel_extras, pixel_flow;
  if (!only_image) pixel_extras = torch::zeros({9, height, width}, colors.options());
  if (!only_image && trans_mat_t2.has_value()) pixel_flow = torch::zeros({height, width, 2}, colors.options());
  if (P == 0) return std::make_tuple(pixel_colors, pixel_opacity, pixel_extras, pixel_flow, n_contrib);

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
      gs_2d_render_forward_kernel<scalar_t, NUM_CHANNELS, true> KERNEL_ARG(tile_grid, block)(
          // constant
          width, height, (scalar_t) near_n, (scalar_t) far_n,
          // inputs
          (T2<int64_t>*) ranges.contiguous().data_ptr<int64_t>(), point_list.contiguous().data_ptr<int32_t>(),
          (T2<scalar_t>*) means2D.contiguous().data_ptr<scalar_t>(), colors.contiguous().data_ptr<scalar_t>(),
          trans_mat.data_ptr<scalar_t>(), (T4<scalar_t>*) normal_opacity.contiguous().data_ptr<scalar_t>(),
          trans_mat_t2.has_value() ? trans_mat_t2.value().data_ptr<scalar_t>() : nullptr,
          // outputs
          n_contrib.data_ptr<int32_t>(), pixel_colors.data_ptr<scalar_t>(), pixel_opacity.data_ptr<scalar_t>(),
          pixel_extras.value().data_ptr<scalar_t>(),                                                       //
          (T2<scalar_t>*) (trans_mat_t2.has_value() ? pixel_flow.value().data_ptr<scalar_t>() : nullptr),  //
          accum_max_count.has_value() ? accum_max_count.value().data_ptr<int32_t>() : nullptr,             //
          accum_weights_p.has_value() ? accum_weights_p.value().data_ptr<scalar_t>() : nullptr,            //
          accum_weights_count.has_value() ? accum_weights_count.value().data_ptr<int32_t>() : nullptr      //
      );
      break;
    default:
      gs_2d_render_forward_kernel<scalar_t, NUM_CHANNELS, false> KERNEL_ARG(tile_grid, block)(
          // constant
          width, height, (scalar_t) near_n, (scalar_t) far_n,
          // inputs
          (T2<int64_t>*) ranges.contiguous().data_ptr<int64_t>(), point_list.contiguous().data_ptr<int32_t>(),
          (T2<scalar_t>*) means2D.contiguous().data_ptr<scalar_t>(), colors.contiguous().data_ptr<scalar_t>(),
          trans_mat.data_ptr<scalar_t>(), (T4<scalar_t>*) normal_opacity.contiguous().data_ptr<scalar_t>(), nullptr,
          // outputs
          n_contrib.data_ptr<int32_t>(), pixel_colors.data_ptr<scalar_t>(), pixel_opacity.data_ptr<scalar_t>(), nullptr,
          nullptr, nullptr, nullptr, nullptr);
      break;
  }
  // cudaDeviceSynchronize();
  CHECK_CUDA_ERROR("gs_2d_render_forward_kernel");
  //   });
  return std::make_tuple(pixel_colors, pixel_opacity, pixel_extras, pixel_flow, n_contrib);
}

std::tuple<Tensor, Tensor, Tensor, Tensor> GS_2D_render_backward(const Tensor& means2D, const Tensor& normal_opacity,
    const Tensor& colors, const Tensor& trans_mat, const torch::optional<Tensor>& trans_mat2, const Tensor& out_opacity,
    const torch::optional<Tensor>& out_others, const torch::optional<Tensor>& out_flows, const Tensor& dL_dout_color,
    const Tensor& dL_dout_opacity, const torch::optional<Tensor>& dL_dothers,
    const torch::optional<Tensor>& dL_dout_flows, const Tensor& ranges, const Tensor& point_list,
    const Tensor& n_contrib, double near_n, double far_n) {
  CHECK_CUDA(dL_dout_color);
  CHECK_CUDA(dL_dout_opacity);
  if (dL_dothers.has_value()) CHECK_CUDA(dL_dothers.value());
  if (dL_dout_flows.has_value()) CHECK_CUDA(dL_dout_flows.value());

  const int P = means2D.size(0);
  const int H = dL_dout_color.size(1);
  const int W = dL_dout_color.size(2);
  int64_t R   = point_list.size(0);

  auto options       = out_opacity.options();
  Tensor dL_dmeans2D = torch::zeros({P, 2}, options);
  Tensor dL_dnorm_op = torch::zeros({P, 4}, options);
  Tensor dL_dcolors  = torch::zeros({P, NUM_CHANNELS}, options);
  Tensor dL_dtrans_m = torch::zeros({P, 3, 3}, options);

  if (P == 0) return std::make_tuple(dL_dcolors, dL_dmeans2D, dL_dtrans_m, dL_dnorm_op);

  const dim3 tile_grid((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, 1);
  const dim3 block(BLOCK_X, BLOCK_Y, 1);

  // AT_DISPATCH_FLOATING_TYPES(means2D.scalar_type(), "gs_2d_render_backward_kernel", [&] {

  using scalar_t = float;
  switch (out_others.has_value() && dL_dothers.has_value()) {
    case true:
      gs_2d_render_backward_kernel<scalar_t, NUM_CHANNELS, true> KERNEL_ARG(tile_grid, block)(
          // connst
          P, W, H, R, (scalar_t) near_n, (scalar_t) far_n,
          // aux inputs
          (T2<int64_t>*) ranges.contiguous().data_ptr<int64_t>(), point_list.contiguous().data_ptr<int32_t>(),
          n_contrib.contiguous().data_ptr<int32_t>(),
          // inputs
          (T2<scalar_t>*) means2D.contiguous().data_ptr<scalar_t>(),
          (T4<scalar_t>*) normal_opacity.contiguous().data_ptr<scalar_t>(), colors.contiguous().data_ptr<scalar_t>(),
          trans_mat.data_ptr<scalar_t>(), trans_mat2.has_value() ? trans_mat2.value().data_ptr<scalar_t>() : nullptr,
          // outputs
          out_opacity.contiguous().data<scalar_t>(), out_others.value().data<scalar_t>(),
          dL_dout_flows.has_value() ? (T2<scalar_t>*) out_flows.value().data_ptr<scalar_t>() : nullptr,
          // grad_outputs
          dL_dout_color.contiguous().data<scalar_t>(),
          dL_dothers.has_value() ? dL_dothers.value().contiguous().data<scalar_t>() : nullptr,
          dL_dout_opacity.contiguous().data<scalar_t>(),
          (T2<scalar_t>*) (dL_dout_flows.has_value() ? dL_dout_flows.value().contiguous().data_ptr<scalar_t>()
                                                     : nullptr),
          // grad_inputs
          (T2<scalar_t>*) dL_dmeans2D.contiguous().data<scalar_t>(), dL_dnorm_op.contiguous().data<scalar_t>(),
          dL_dcolors.contiguous().data<scalar_t>(), dL_dtrans_m.data<scalar_t>());
      break;

    default:
      gs_2d_render_backward_kernel<scalar_t, NUM_CHANNELS, false> KERNEL_ARG(tile_grid, block)(
          // connst
          P, W, H, R, (scalar_t) near_n, (scalar_t) far_n,
          // aux inputs
          (T2<int64_t>*) ranges.contiguous().data_ptr<int64_t>(), point_list.contiguous().data_ptr<int32_t>(),
          n_contrib.contiguous().data_ptr<int32_t>(),
          // inputs
          (T2<scalar_t>*) means2D.contiguous().data_ptr<scalar_t>(),
          (T4<scalar_t>*) normal_opacity.contiguous().data_ptr<scalar_t>(), colors.contiguous().data_ptr<scalar_t>(),
          trans_mat.data_ptr<scalar_t>(), trans_mat2.has_value() ? trans_mat2.value().data_ptr<scalar_t>() : nullptr,
          // outputs
          out_opacity.contiguous().data<scalar_t>(), nullptr,
          dL_dout_flows.has_value() ? (T2<scalar_t>*) out_flows.value().data_ptr<scalar_t>() : nullptr,
          // grad_outputs
          dL_dout_color.contiguous().data<scalar_t>(),
          dL_dothers.has_value() ? dL_dothers.value().contiguous().data<scalar_t>() : nullptr,
          dL_dout_opacity.contiguous().data<scalar_t>(),
          (T2<scalar_t>*) (dL_dout_flows.has_value() ? dL_dout_flows.value().contiguous().data_ptr<scalar_t>()
                                                     : nullptr),
          // grad_inputs
          (T2<scalar_t>*) dL_dmeans2D.contiguous().data<scalar_t>(), dL_dnorm_op.contiguous().data<scalar_t>(),
          dL_dcolors.contiguous().data<scalar_t>(), dL_dtrans_m.data<scalar_t>());
      break;
  }

  // cudaDeviceSynchronize();
  CHECK_CUDA_ERROR("gs_2d_render_backward_kernel");
  // });
  return std::make_tuple(dL_dcolors, dL_dmeans2D, dL_dtrans_m, dL_dnorm_op);
}

REGIST_PYTORCH_EXTENSION(gs_2d_render, {
  m.def("GS_2D_render_forward", &GS_2D_render_forward, "GS_2D_render_forward (CUDA)");
  m.def("GS_2D_render_backward", &GS_2D_render_backward, "GS_2D_render_backward (CUDA)");
})

}  // namespace GaussianRasterizer
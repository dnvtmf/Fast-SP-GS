/*
based on
https://github.com/fatPeter/mini-splatting2/blob/5ec9202c2db7d900728bfc09733e85467cff885e/submodules/diff-gaussian-rasterization_ms/cuda_rasterizer/forward.cu#L616
*/
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include "gaussian_render.h"
#include "ops_3d.h"
#include "util.cuh"

namespace cg = cooperative_groups;
using namespace OPS_3D;
namespace GaussianRasterizer {

// Main rasterization method. Collaboratively works on one tile per block, each thread treats one pixel.
// Alternates between fetching and rasterizing data.
template <typename T, uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X* BLOCK_Y) render_mid_depth_kernel(const T2<int64_t>* __restrict__ ranges,
    const int32_t* __restrict__ point_list, int W, int H, const T2<T>* __restrict__ points_xy_image,
    const T* __restrict__ features, const T4<T>* __restrict__ conic_opacity, const T* __restrict__ bg_color,
    int32_t* __restrict__ accum_max_count, T* __restrict__ accum_weights_p, int32_t* __restrict__ accum_weights_count,
    T* __restrict__ out_color, T* __restrict__ out_pts, T* __restrict__ out_depth, T* __restrict__ accum_alpha,
    int* __restrict__ gidx, T* __restrict__ discriminants, const T* __restrict__ means3D,
    const vec3<T>* __restrict__ scales, const vec4<T>* __restrict__ rotations, const T* __restrict__ projmatrix_inv,
    const vec3<T>* __restrict__ cam_pos) {
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
  auto range       = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
  const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
  int toDo         = range.y - range.x;

  // Allocate storage for batches of collectively fetched data.
  __shared__ int collected_id[BLOCK_SIZE];
  __shared__ T2<T> collected_xy[BLOCK_SIZE];
  __shared__ T4<T> collected_conic_opacity[BLOCK_SIZE];

  // Initialize helper variables
  T t                 = T(1.0);
  int32_t contributor = 0;
  T C[CHANNELS]       = {0};

  T weight_max       = 0;
  T depth_max        = 0;
  T discriminant_max = 0;

  int idx_max = 0;

  vec3<T> ray_origin = *cam_pos;
  vec3<T> point_rec  = {0, 0, 0};
  T3<T> p_proj_r     = {(2 * pixf.x + 1) / W - 1, (2 * pixf.y + 1) / H - 1, 1};

  // inverse process of 'Transform point by projecting'
  T p_hom_x_r = p_proj_r.x * T(1.0000001);
  T p_hom_y_r = p_proj_r.y * T(1.0000001);
  // self.zfar = 100.0, self.znear = 0.01
  T p_hom_z_r = (100 - 100 * 0.01) / (100 - 0.01);
  T p_hom_w_r = 1;

  vec3<T> p_hom_r  = {p_hom_x_r, p_hom_y_r, p_hom_z_r};
  vec4<T> p_orig_r = xfm_p_4x4(p_hom_r, projmatrix_inv);

  vec3<T> ray_direction = {
      p_orig_r.x - ray_origin.x,
      p_orig_r.y - ray_origin.y,
      p_orig_r.z - ray_origin.z,
  };
  T len                            = sqrt(ray_direction.dot(ray_direction));
  vec3<T> normalized_ray_direction = ray_direction / len;
  mat3<T> R;

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

      // if (pix_id == 0) {
      //   printf("\033[31mgs=%d, xy=%.6f, %.6f, power=%.6f, alpha=%.6f, sigma=%.6f, last_sigma=%.6f\n\033[0m",
      //       collected_id[j], xy.x, xy.y, power, alpha, test_T, t);
      // }
      // Eq. (3) from 3D Gaussian splatting paper.
      T w = alpha * t;
      for (int ch = 0; ch < CHANNELS; ch++) C[ch] += features[collected_id[j] * CHANNELS + ch] * w;
      // compute Gaussian depth
      // Normalize quaternion to get valid rotation
      vec4<T> q = rotations[collected_id[j]];  // / glm::length(rot);
      quaternion_to_R(q, R._data);

      vec3<T> temp = {
          ray_origin.x - means3D[3 * collected_id[j] + 0],
          ray_origin.y - means3D[3 * collected_id[j] + 1],
          ray_origin.z - means3D[3 * collected_id[j] + 2],
      };

      vec3<T> rotated_ray_origin    = {R[0].x * temp.x + R[1].x * temp.y + R[2].x * temp.z,
             R[0].y * temp.x + R[1].y * temp.y + R[2].y * temp.z, R[0].z * temp.x + R[1].z * temp.y + R[2].z * temp.z};
      vec3<T> rotated_ray_direction = {R[0].x * normalized_ray_direction.x + R[1].x * normalized_ray_direction.y +
                                           R[2].x * normalized_ray_direction.z,
          R[0].y * normalized_ray_direction.x + R[1].y * normalized_ray_direction.y +
              R[2].y * normalized_ray_direction.z,
          R[0].z * normalized_ray_direction.x + R[1].z * normalized_ray_direction.y +
              R[2].z * normalized_ray_direction.z};

      vec3<T> a_t = rotated_ray_direction / (scales[collected_id[j]] * 3.0f) * rotated_ray_direction /
                    (scales[collected_id[j]] * 3.0f);
      T a = a_t.x + a_t.y + a_t.z;

      vec3<T> b_t = rotated_ray_direction / (scales[collected_id[j]] * 3.0f) * rotated_ray_origin /
                    (scales[collected_id[j]] * 3.0f);
      T b = 2 * (b_t.x + b_t.y + b_t.z);

      vec3<T> c_t =
          rotated_ray_origin / (scales[collected_id[j]] * 3.0f) * rotated_ray_origin / (scales[collected_id[j]] * 3.0f);
      T c = c_t.x + c_t.y + c_t.z - 1;

      T discriminant = b * b - 4 * a * c;
      T depth        = (-b / 2 / a) / len;
      // if (pix_id == 525181) {
      //   printf("\033[31mgs=%d; a=%f, b=%f, c=%f; discriminant=%f; depth=%f, len=%f\n\033[0m", collected_id[j], a, b,
      //   c,
      //       discriminant, depth, len);
      // }
      if (depth < 0) continue;

      if (weight_max < w) {
        weight_max       = w;
        depth_max        = depth;
        discriminant_max = discriminant;
        idx_max          = collected_id[j];
        point_rec        = ray_origin + (-b / 2 / a) * normalized_ray_direction;
      }

      t = test_T;
    }
  }

  // All threads that treat valid pixel write out their final rendering data to the frame and auxiliary buffers.
  if (inside) {
    // final_T[pix_id]   = t;
    // n_contrib[pix_id] = last_contributor;
    for (int ch = 0; ch < CHANNELS; ch++) out_color[ch * H * W + pix_id] = C[ch] + t * bg_color[ch];
    for (int ch = 0; ch < 3; ch++) out_pts[ch * H * W + pix_id] = point_rec[ch];

    out_depth[pix_id]     = depth_max;
    accum_alpha[pix_id]   = t;
    discriminants[pix_id] = discriminant_max;
    gidx[pix_id]          = idx_max;
  }
}

vector<Tensor> gaussian_mid_depth(int width, int height, Tensor& means2D, Tensor& conic_opacity, const Tensor& colors,
    const Tensor& point_list, const Tensor& ranges, Tensor& means3D, Tensor& scales, Tensor& rotations, Tensor& Tc2w,
    Tensor& campos, Tensor& bg) {
  CHECK_INPUT(colors);
  CHECK_NDIM(colors, 2);
  int P = colors.size(0);  // num points

  Tensor out_colors    = torch::zeros({3, height, width}, colors.options());
  Tensor out_points    = torch::zeros({3, height, width}, colors.options());
  Tensor out_depth     = torch::zeros({1, height, width}, colors.options());
  Tensor out_opacity   = torch::zeros({1, height, width}, colors.options());
  Tensor discriminants = torch::zeros({1, height, width}, colors.options());
  Tensor gidx          = torch::zeros({1, height, width}, colors.options().dtype(torch::kInt32));

  Tensor accum_max_count     = torch::zeros({P}, colors.options().dtype(torch::kInt32));
  Tensor accum_weights_p     = torch::zeros({P}, colors.options());
  Tensor accum_weights_count = torch::zeros({P}, colors.options().dtype(torch::kInt32));

  if (P == 0)
    return {out_colors, out_points, out_depth, out_opacity, gidx, discriminants, accum_max_count, accum_weights_p,
        accum_weights_count};

  const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
  const dim3 block(BLOCK_X, BLOCK_Y, 1);

  //   AT_DISPATCH_FLOATING_TYPES(means2D.scalar_type(), "render_mid_depth_kernel", [&] {
  using scalar_t = float;
  render_mid_depth_kernel<scalar_t, 3> KERNEL_ARG(tile_grid, block)(
      (T2<int64_t>*) ranges.contiguous().data_ptr<int64_t>(), point_list.contiguous().data_ptr<int32_t>(), width,
      height, (T2<scalar_t>*) means2D.contiguous().data_ptr<scalar_t>(), colors.contiguous().data_ptr<scalar_t>(),
      (T4<scalar_t>*) conic_opacity.contiguous().data_ptr<scalar_t>(), bg.data_ptr<scalar_t>(),
      accum_max_count.data_ptr<int32_t>(), accum_weights_p.data_ptr<scalar_t>(),
      accum_weights_count.data_ptr<int32_t>(), out_colors.data_ptr<scalar_t>(), out_points.data_ptr<scalar_t>(),
      out_depth.data_ptr<scalar_t>(), out_opacity.data_ptr<scalar_t>(), gidx.data_ptr<int32_t>(),
      discriminants.data_ptr<scalar_t>(),  //
      means3D.data_ptr<scalar_t>(), (vec3<scalar_t>*) scales.data_ptr<scalar_t>(),
      (vec4<scalar_t>*) rotations.data_ptr<scalar_t>(), Tc2w.data_ptr<scalar_t>(),
      (vec3<scalar_t>*) campos.data_ptr<scalar_t>());
  CHECK_CUDA_ERROR("render_mid_depth_kernel");
  //   });
  return {out_colors, out_points, out_depth, out_opacity, gidx, discriminants, accum_max_count, accum_weights_p,
      accum_weights_count};
}

REGIST_PYTORCH_EXTENSION(gs_mid_depth, { m.def("gs_mid_depth", &gaussian_mid_depth, "gaussian_mid_depth (CUDA)"); })
}  // namespace GaussianRasterizer
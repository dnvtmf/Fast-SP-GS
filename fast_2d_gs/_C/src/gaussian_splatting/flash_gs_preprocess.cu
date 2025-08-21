#if __INTELLISENSE__
#define __CUDACC__
#endif
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cuda.h>
#include <stdio.h>

#include "gaussian_render.h"
#include "ops_3d.h"
#include "spherical_harmonic.h"
#include "util.cuh"

namespace cg = cooperative_groups;

namespace GaussianRasterizer {
using namespace OPS_3D;

/*
// Forward version of 2D covariance matrix computation
template <typename T>
__device__ vec3<T> computeCov2D(
    const vec3<T>& mean, T focal_x, T focal_y, T tan_fovx, T tan_fovy, const T* cov3D, const T* viewmatrix);
// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care of quaternion normalization.
template <typename T>
__device__ void computeCov3D(const vec3<T>& s, const vec4<T>& rot, T* cov3D);
*/
template <typename T>
__device__ void computeCov3D(const vec3<T>& s, const vec4<T>& rot, T* cov3D) {
  // Create scaling matrix
  T R[9] = {0};
  quaternion_to_R(rot, R);  // Normalize quaternion to get valid rotation

  T sx2 = s.x * s.x;
  T sy2 = s.y * s.y;
  T sz2 = s.z * s.z;

  // Covariance is symmetric, only store upper right
  cov3D[0] = R[0] * R[0] * sx2 + R[1] * R[1] * sy2 + R[2] * R[2] * sz2;
  cov3D[1] = R[0] * R[3] * sx2 + R[1] * R[4] * sy2 + R[2] * R[5] * sz2;
  cov3D[2] = R[0] * R[6] * sx2 + R[1] * R[7] * sy2 + R[2] * R[8] * sz2;
  cov3D[3] = R[3] * R[3] * sx2 + R[4] * R[4] * sy2 + R[5] * R[5] * sz2;
  cov3D[4] = R[3] * R[6] * sx2 + R[4] * R[7] * sy2 + R[5] * R[8] * sz2;
  cov3D[5] = R[6] * R[6] * sx2 + R[7] * R[7] * sy2 + R[8] * R[8] * sz2;
}

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
    const T* dL_dcov2D, const T* dL_dconic, T* dL_dcov3D, T* dL_dvm);

// Backward pass for the conversion of scale and rotation to a 3D covariance matrix for each Gaussian.
template <typename T>
__device__ void computeCov3D_backward(const T* scale, const T* rot, const T* dL_dcov3D, T* dL_dscales, T* dL_drots);
uint32_t getHigherMsb(uint32_t n);

constexpr float log2e           = 1.4426950216293334961f;
constexpr float ln2             = 0.69314718055f;
constexpr int FLASHGS_WARP_SIZE = 32;

__forceinline__ __device__ float fast_max_f32(float a, float b) {
  float d;
  asm volatile("max.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
  return d;
}

__forceinline__ __device__ float fast_sqrt_f32(float x) {
  float y;
  asm volatile("sqrt.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__forceinline__ __device__ float fast_rsqrt_f32(float x) {
  float y;
  asm volatile("rsqrt.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__forceinline__ __device__ float fast_lg2_f32(float x) {
  float y;
  asm volatile("lg2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

template <typename T>
__forceinline__ __device__ void getRect(
    const T2<T> p, int width, int height, int2& rect_min, int2& rect_max, dim3 grid, int block_x, int block_y) {
  rect_min = {min((int) grid.x, max((int) 0, (int) ((p.x - width) / (T) block_x))),
      min((int) grid.y, max((int) 0, (int) ((p.y - height) / (T) block_y)))};
  rect_max = {min((int) grid.x, max((int) 0, (int) ((p.x + width) / (T) block_x) + 1)),
      min((int) grid.y, max((int) 0, (int) ((p.y + height) / (T) block_y) + 1))};
}

template <typename T>
union cov3d_t {
  T2<T> f2[3];
  T s[6];
};

template <typename T>
union shs_deg3_t {
  T4<T> f4[12];
  vec3<T> v3[16];
};

// Spherical harmonics coefficients
__device__ const float SH_C0   = 0.28209479177387814f;
__device__ const float SH_C1   = 0.4886025119029199f;
__device__ const float SH_C2[] = {
    1.0925484305920792f, -1.0925484305920792f, 0.31539156525252005f, -1.0925484305920792f, 0.5462742152960396f};
__device__ const float SH_C3[] = {-0.5900435899266435f, 2.890611442640554f, -0.4570457994644658f, 0.3731763325901154f,
    -0.4570457994644658f, 1.445305721320277f, -0.5900435899266435f};

template <typename T>
__forceinline__ __device__ vec3<T> computeColorFromSH(
    int idx, vec3<T> p_orig, vec3<T> campos, const shs_deg3_t<T>* shs) {
  // The implementation is loosely based on code for
  // "Differentiable Point-Based Radiance Fields for
  // Efficient View Synthesis" by Zhang et al. (2022)
  vec3<T> dir = p_orig - campos;
  T l2        = dir.x * dir.x + dir.y * dir.y + dir.z * dir.z;
  T rsqrt_l2  = fast_rsqrt_f32(l2);
  dir *= rsqrt_l2;

  auto sh        = shs + idx;
  vec3<T> result = SH_C0 * sh->v3[0] + T(0.5);

  T x    = dir.x;
  T y    = dir.y;
  T z    = dir.z;
  result = result - SH_C1 * y * sh->v3[1] + SH_C1 * z * sh->v3[2] - SH_C1 * x * sh->v3[3];

  T xx = x * x, yy = y * y, zz = z * z;
  T xy = x * y, yz = y * z, xz = x * z;
  result = result + SH_C2[0] * xy * sh->v3[4] + SH_C2[1] * yz * sh->v3[5] +
           SH_C2[2] * (T(2.0) * zz - xx - yy) * sh->v3[6] + SH_C2[3] * xz * sh->v3[7] +
           SH_C2[4] * (xx - yy) * sh->v3[8];

  result = result + SH_C3[0] * y * (T(3.0) * xx - yy) * sh->v3[9] + SH_C3[1] * xy * z * sh->v3[10] +
           SH_C3[2] * y * (T(4.0) * zz - xx - yy) * sh->v3[11] +
           SH_C3[3] * z * (T(2.0) * zz - T(3.0) * xx - T(3.0) * yy) * sh->v3[12] +
           SH_C3[4] * x * (T(4.0) * zz - xx - yy) * sh->v3[13] + SH_C3[5] * z * (xx - yy) * sh->v3[14] +
           SH_C3[6] * x * (xx - T(3.0) * yy) * sh->v3[15];

  result.x = fast_max_f32(result.x, T(0.0));
  result.y = fast_max_f32(result.y, T(0.0));
  result.z = fast_max_f32(result.z, T(0.0));
  return result;
}

template <typename T>
__forceinline__ __device__ bool segment_intersect_ellipse(T a, T b, T c, T d, T l, T r) {
  T delta = b * b - T(4.0) * a * c;
  // return delta >= T(0.0) && t1 <= sqrt(delta) && t2 >= -sqrt(delta)
  T t1 = (l - d) * (T(2.0) * a) + b;
  T t2 = (r - d) * (T(2.0) * a) + b;
  return delta >= T(0.0) && (t1 <= T(0.0) || t1 * t1 <= delta) && (t2 >= T(0.0) || t2 * t2 <= delta);
}

template <typename T>
__forceinline__ __device__ bool block_intersect_ellipse(
    int2 pix_min, int2 pix_max, T2<T> center, T3<T> conic, T power) {
  T a, b, c, dx, dy;
  T w = T(2.0) * power;

  if (center.x * T(2.0) < pix_min.x + pix_max.x) {
    dx = center.x - pix_min.x;
  } else {
    dx = center.x - pix_max.x;
  }
  a = conic.z;
  b = T(-2.0) * conic.y * dx;
  c = conic.x * dx * dx - w;

  if (segment_intersect_ellipse<T>(a, b, c, center.y, pix_min.y, pix_max.y)) {
    return true;
  }

  if (center.y * T(2.0) < pix_min.y + pix_max.y) {
    dy = center.y - pix_min.y;
  } else {
    dy = center.y - pix_max.y;
  }
  a = conic.x;
  b = T(-2.0) * conic.y * dy;
  c = conic.z * dy * dy - w;

  if (segment_intersect_ellipse<T>(a, b, c, center.x, pix_min.x, pix_max.x)) {
    return true;
  }
  return false;
}

template <typename T>
__forceinline__ __device__ bool block_contains_center(int2 pix_min, int2 pix_max, T2<T> center) {
  return center.x >= pix_min.x && center.x <= pix_max.x && center.y >= pix_min.y && center.y <= pix_max.y;
}

__forceinline__ __device__ float ndc2Pix(float v, int S) { return ((v + 1.0) * S - 1.0) * 0.5; }

template <typename T>
__global__ void flash_gs_prepocess_forward_kernel(int P, int M, const vec3<T>* __restrict__ positions,
    const T* __restrict__ opacities, const shs_deg3_t<T>* __restrict__ shs, mat4<T> viewmatrix, mat4<T> projmatrix,
    vec3<T> cam_position, const int W, const int H, int block_x, int block_y, const T tan_fovx, const T tan_fovy,
    const T focal_x, const T focal_y, T2<T>* __restrict__ points_xy, cov3d_t<T>* __restrict__ cov3Ds,
    T4<T>* __restrict__ rgb_depth, T4<T>* __restrict__ conic_opacity, int* __restrict__ curr_offset,
    uint64_t* __restrict__ gaussian_keys_unsorted, uint32_t* __restrict__ gaussian_values_unsorted, const dim3 grid,
    bool is_opengl) {
  int lane    = threadIdx.y * blockDim.x + threadIdx.x;
  int warp_id = blockIdx.x * blockDim.z + threadIdx.z;
  int idx_vec = warp_id * FLASHGS_WARP_SIZE + lane;

  // Initialize radius and touched tiles to 0. If this isn't changed,
  // this Gaussian will not be processed further.
  bool point_valid = false;
  vec3<T> p_orig;
  int width  = 0;
  int height = 0;
  T3<T> p_view;
  T2<T> point_xy;
  T3<T> conic;
  T opacity;
  T power;
  T log2_opacity;
  int2 rect_min;
  int2 rect_max;
  if (idx_vec < P) {
    do {
      // Perform near culling, quit if outside.
      p_orig = positions[idx_vec];
      p_view = xfm_p_4x3(p_orig, (T*) &viewmatrix);
      if (p_view.z <= T(0.2)) break;
      opacity = opacities[idx_vec];
      if (T(255.0) * opacity < T(1.0)) break;

      // Transform point by projecting
      T4<T> p_hom  = xfm_p_4x4(p_orig, (T*) &projmatrix);
      T p_w        = T(1.0) / (p_hom.w + T(0.0000001));
      T3<T> p_proj = {p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w};

      // Compute 2D screen-space covariance matrix
      T3<T> cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, (T*) &cov3Ds[idx_vec], (T*) &viewmatrix);

      // Invert covariance (EWA algorithm)
      T det     = (cov.x * cov.z - cov.y * cov.y);
      T det_inv = T(1.) / det;
      conic     = {cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv};

      log2_opacity = fast_lg2_f32(opacity);
      power        = ln2 * T(8.0) + ln2 * log2_opacity;
      width        = (int) (T(1.414214) * fast_sqrt_f32(cov.x * power) + T(1.0));
      height       = (int) (T(1.414214) * fast_sqrt_f32(cov.z * power) + T(1.0));

      point_xy = {ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H)};
      //   point_xy = {(T(1.0) + p_proj.x) * T(0.5) * W, (T(1.0) + (is_opengl ? -p_proj.y : p_proj.y)) * T(0.5) * H};
      getRect<T>(point_xy, width, height, rect_min, rect_max, grid, block_x, block_y);
      point_valid = (rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) > 0;
    } while (false);
  }
  bool single_tile = point_valid && (rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 1;
  if (single_tile) {
    int2 pix_min = {rect_min.x * block_x, rect_min.y * block_y};
    int2 pix_max = {pix_min.x + block_x - 1, pix_min.y + block_y - 1};
    bool valid   = block_contains_center<T>(pix_min, pix_max, point_xy) ||
                 block_intersect_ellipse(pix_min, pix_max, point_xy, conic, power);
    if (valid) {
      uint64_t key = rect_min.y * grid.x + rect_min.x;
      key <<= 32;
      key |= __float_as_uint(p_view.z);
      int offset = atomicAdd(curr_offset, 1);
      if (offset < M) {
        gaussian_keys_unsorted[offset]   = key;
        gaussian_values_unsorted[offset] = idx_vec;
      }
    }
    point_valid = false;
  }

  // Generate no key/value pair for invisible Gaussians
  int multi_tiles   = __ballot_sync(~0, point_valid);
  bool vertex_valid = single_tile;
  while (multi_tiles) {
    int i = __ffs(multi_tiles) - 1;
    multi_tiles &= multi_tiles - 1;
    // Find this Gaussian's offset in buffer for writing keys/values.
    T2<T> my_point_xy = {__shfl_sync(~0, point_xy.x, i), __shfl_sync(~0, point_xy.y, i)};
    T3<T> my_conic    = {
        __shfl_sync(~0, conic.x, i),
        __shfl_sync(~0, conic.y, i),
        __shfl_sync(~0, conic.z, i),
    };
    int2 my_rect_min = {__shfl_sync(~0, rect_min.x, i), __shfl_sync(~0, rect_min.y, i)};
    int2 my_rect_max = {__shfl_sync(~0, rect_max.x, i), __shfl_sync(~0, rect_max.y, i)};
    T my_depth       = __shfl_sync(~0, p_view.z, i);
    T my_power       = __shfl_sync(~0, power, i);
    int idx          = warp_id * FLASHGS_WARP_SIZE + i;

    // For each tile that the bounding rect overlaps, emit a
    // key/value pair. The key is |  tile ID  |      depth      |,
    // and the value is the ID of the Gaussian. Sorting the values
    // with this key yields Gaussian IDs in a list, such that they
    // are first sorted by tile and then by depth.
    for (int y0 = my_rect_min.y; y0 < my_rect_max.y; y0 += blockDim.y)  // 循环迭代tile范围，为每个tile生成键值对
    {
      int y = y0 + threadIdx.y;
      for (int x0 = my_rect_min.x; x0 < my_rect_max.x; x0 += blockDim.x) {
        int x      = x0 + threadIdx.x;
        bool valid = y < my_rect_max.y && x < my_rect_max.x;

        if (valid) {
          int2 pix_min = {x * block_x, y * block_y};
          int2 pix_max = {pix_min.x + block_x - 1, pix_min.y + block_y - 1};
          valid        = block_contains_center<T>(pix_min, pix_max, my_point_xy) ||
                  block_intersect_ellipse<T>(pix_min, pix_max, my_point_xy, my_conic, my_power);
        }

        int mask = __ballot_sync(~0, valid);
        if (mask == 0) {
          continue;
        }
        int my_offset;
        if (lane == 0) {
          my_offset = atomicAdd(curr_offset, __popc(mask));
        }
        vertex_valid = vertex_valid || i == lane;
        int count    = __popc(mask & ((1 << lane) - 1));
        uint64_t key = y * grid.x + x;
        key <<= 32;
        key |= __float_as_uint(my_depth);
        my_offset = __shfl_sync(~0, my_offset, 0);
        if (valid && my_offset + count < M) {
          gaussian_keys_unsorted[my_offset + count]   = key;
          gaussian_values_unsorted[my_offset + count] = idx;
        }
      }
    }
  }

  if (vertex_valid) {
    points_xy[idx_vec]     = point_xy;
    conic_opacity[idx_vec] = {(-T(0.5) * log2e) * conic.x, -log2e * conic.y, (-T(0.5) * log2e) * conic.z, log2_opacity};
    auto color             = computeColorFromSH(idx_vec, p_orig, cam_position, (const shs_deg3_t<T>*) shs);
    rgb_depth[idx_vec]     = {color.x, color.y, color.z, p_view.z};
  }
}

vector<Tensor> FlashGS_preprocess_forward(
    // scalar
    int W, int H, int sh_degree, bool is_opengl, int block_x, int block_y, int max_rendered,
    // Gaussians
    Tensor means3D, torch::optional<Tensor> scales, torch::optional<Tensor> rotations, Tensor opacities,
    torch::optional<Tensor> shs,
    // cameras
    Tensor Tw2v, Tensor Tv2c, Tensor cam_pos, Tensor focals,
    // pre-computed tensors
    torch::optional<Tensor> cov3D_precomp, torch::optional<Tensor> cov2D_precomp,
    torch::optional<Tensor> colors_precomp, torch::optional<Tensor> means2D, Tensor& gaussian_keys_unsorted,
    Tensor& gaussian_values_unsorted, Tensor& curr_offset) {
  CHECK_INPUT(means3D);
  BCNN_ASSERT(means3D.ndimension() == 2 && means3D.size(-1) == 3, "Error shape for means3D");
  int P = means3D.size(0);
  int M = 0;
  BCNN_ASSERT(cov3D_precomp.has_value(), "Flash need cvo3D");
  if (!(cov2D_precomp.has_value() || cov3D_precomp.has_value())) {
    BCNN_ASSERT(scales.has_value() && rotations.has_value(), "Need scales/rotations or cov2D or cov3D");
    CHECK_INPUT(scales.value());
    CHECK_INPUT(rotations.value());
    CHECK_SHAPE(scales.value(), P, 3);
    CHECK_SHAPE(rotations.value(), P, 4);
  }
  CHECK_INPUT(opacities);
  if (!colors_precomp.has_value()) {
    BCNN_ASSERT(shs.has_value(), "shs and colors must be None at same time")
    CHECK_INPUT(shs.value());
    BCNN_ASSERT(shs.value().ndimension() == 3 && shs.value().size(0) == P &&
                    shs.value().size(1) >= (1 + sh_degree) * (1 + sh_degree) && shs.value().size(2) == 3,
        "Error shape for sh features");
    M = shs.value().size(1);
  }
  BCNN_ASSERT(M == 16 && sh_degree == 3, "error sh features");

  //   CHECK_INPUT(Tw2v);
  //   CHECK_INPUT(Tv2c);
  //   CHECK_INPUT(cam_pos);
  if (means2D.has_value()) {
    CHECK_INPUT(means2D.value());
    CHECK_SHAPE(means2D.value(), P, 2);
  }

  CHECK_SHAPE(opacities, P, 1);
  CHECK_SHAPE(Tw2v, 4, 4);
  CHECK_SHAPE(Tv2c, 4, 4);
  CHECK_SHAPE(cam_pos, 3);

  Tensor colors_depth  = colors_precomp.has_value() ? colors_precomp.value() : torch::zeros_like(means3D);
  Tensor cov3Ds        = cov3D_precomp.has_value() ? cov3D_precomp.value() : torch::zeros({P, 6}, means3D.options());
  Tensor means2D_      = means2D.has_value() ? means2D.value() : torch::zeros({P, 2}, means3D.options());
  Tensor depths        = torch::zeros({P}, means3D.options());
  Tensor radii         = torch::zeros({P}, means3D.options().dtype(torch::kInt32));
  Tensor tiles_touched = torch::zeros({P}, means3D.options().dtype(torch::kInt32));
  Tensor conic_opacity = torch::zeros({P, 4}, means3D.options());
  dim3 grid((W + block_x - 1) / block_x, (H + block_y - 1) / block_y, 1);
  //   AT_DISPATCH_FLOATING_TYPES(means3D.scalar_type(), "FlashGS_preprocess_forward (CUDA)", [&] {
  typedef float scalar_t;
  scalar_t focal_x  = focals[0].item<scalar_t>();
  scalar_t focal_y  = focals[1].item<scalar_t>();
  scalar_t tan_fovx = W / (2.0 * focal_x);
  scalar_t tan_fovy = H / (2.0 * focal_y);

  flash_gs_prepocess_forward_kernel<scalar_t> KERNEL_ARG((P + 127) / 128, dim3(8, 4, 4))(P, max_rendered,
      (vec3<scalar_t>*) means3D.data_ptr<scalar_t>(), opacities.data_ptr<scalar_t>(),
      (shs_deg3_t<scalar_t>*) (shs.has_value() ? shs.value().data_ptr<scalar_t>() : nullptr),
      *((mat4<scalar_t>*) Tw2v.cpu().data_ptr<scalar_t>()), *((mat4<scalar_t>*) Tv2c.cpu().data_ptr<scalar_t>()),
      *((vec3<scalar_t>*) cam_pos.cpu().data_ptr<scalar_t>()), W, H, block_x, block_y, tan_fovx, tan_fovy, focal_x,
      focal_y, (T2<scalar_t>*) means2D_.data_ptr<scalar_t>(), (cov3d_t<scalar_t>*) cov3Ds.data_ptr<scalar_t>(),
      (T4<scalar_t>*) colors_depth.data_ptr<scalar_t>(), (T4<scalar_t>*) conic_opacity.data_ptr<scalar_t>(),
      curr_offset.data_ptr<int>(), (uint64_t*) gaussian_keys_unsorted.data_ptr<int64_t>(),
      (uint32_t*) gaussian_values_unsorted.data_ptr<int32_t>(), grid, is_opengl);
  //   cudaDeviceSynchronize();
  //   CHECK_CUDA_ERROR("flash_gs_prepocess_forward_kernel");
  //   });

  return {means2D_, colors_depth, radii, tiles_touched, cov3Ds, conic_opacity};
}

void sort_gaussian_torch(int num_rendered, int width, int height, int block_x, int block_y,
    torch::Tensor& list_sorting_space, torch::Tensor& gaussian_keys_unsorted, torch::Tensor& gaussian_values_unsorted,
    torch::Tensor& gaussian_keys_sorted, torch::Tensor& gaussian_values_sorted) {
  dim3 grid((width + block_x - 1) / block_x, (height + block_y - 1) / block_y, 1);
  size_t sorting_size    = list_sorting_space.size(0);
  int end_bit            = 32 + getHigherMsb(grid.x * grid.y);
  void* space_ptr        = (void*) list_sorting_space.contiguous().data_ptr();
  const uint64_t* ku_ptr = (uint64_t*) gaussian_keys_unsorted.contiguous().data_ptr<int64_t>();
  uint64_t* ks_ptr       = (uint64_t*) gaussian_keys_sorted.contiguous().data_ptr<int64_t>();
  const uint32_t* vu_ptr = (uint32_t*) gaussian_values_unsorted.contiguous().data_ptr<int32_t>();
  uint32_t* vs_ptr       = (uint32_t*) gaussian_values_sorted.contiguous().data_ptr<int32_t>();

  auto status = cub::DeviceRadixSort::SortPairs<uint64_t, uint32_t>(
      space_ptr, sorting_size, ku_ptr, ks_ptr, vu_ptr, vs_ptr, num_rendered, 0, end_bit);
  if (status != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(status));
  }
}

size_t get_sort_buffer_size(int num_rendered) {
  size_t sort_buffer_size = 0;
  cub::DeviceRadixSort::SortPairs<uint64_t, uint32_t>(
      nullptr, sort_buffer_size, nullptr, nullptr, nullptr, nullptr, num_rendered, 0, sizeof(uint64_t) * 8);
  return sort_buffer_size;
}

REGIST_PYTORCH_EXTENSION(gs_FlashGS_preprocess, {
  m.def("FlashGS_preprocess_forward", &FlashGS_preprocess_forward, "FlashGS preprocess forward (CUDA)");
  //   m.def("gs_preprocess_backward", &GS_preprocess_backward, "preprocess_backward (CUDA)");
  m.def("FlashGS_sort_gaussian", &sort_gaussian_torch, "FlashGS sort_gaussian_torch (CUDA)");
  m.def("FlashGS_get_sort_buffer_size", &get_sort_buffer_size, " FlashGS get_sort_buffer_size (CUDA)");
})
}  // namespace GaussianRasterizer
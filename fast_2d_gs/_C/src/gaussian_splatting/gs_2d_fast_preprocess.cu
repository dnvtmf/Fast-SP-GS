/*
paper: 2D Gaussian Splatting for Geometrically Accurate Radiance Fields, SIGGRAPH 2024
code: https://github.com/hbb1/2d-gaussian-splatting
 */

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>

#include "gaussian_render.h"
#include "ops_3d.h"
#include "spherical_harmonic.h"
#include "util.cuh"

namespace cg = cooperative_groups;
using namespace OPS_3D;
namespace GaussianRasterizer {
#define DUAL_VISIABLE 1
#define TIGHTBBOX 0

constexpr double FilterSize = 0.707106;  // sqrt(2) / 2;
// constexpr float log2e       = 1.4426950216293334961f;
constexpr float ln2 = 0.69314718055f;
const int debug_i   = 0;

uint32_t getHigherMsb(uint32_t n);

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

__forceinline__ __device__ float fast_lg2(float x) {
  float y;
  asm volatile("lg2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__forceinline__ __device__ double fast_lg2(double x) {
  double y;
  asm volatile("lg2.approx.f64 %0, %1;" : "=d"(y) : "d"(x));
  return y;
}
// Compute a 2D-to-2D mapping matrix from a tangent plane into a image plane given a 2D gaussian parameters.
template <typename T>
__device__ void compute_transmat(const T3<T>& p_orig, const T2<T>& scale, const T4<T>& rot, const mat4<T>& Tw2v,
    const mat3<T>& Tv2s, mat3<T>& matrices, vec3<T>& normal) {
  mat3<T> L;  // L = quat_to_mat(rot) * scale_to_mat(scale)
  vec3<T> tmp;
  T* data = L._data;

  data[0] = (1 - 2 * (rot.y * rot.y + rot.z * rot.z)) * scale.x;
  data[1] = 2 * (rot.x * rot.y - rot.z * rot.w) * scale.y;
  data[2] = p_orig.x;
  tmp.x   = 2 * (rot.y * rot.w + rot.x * rot.z);

  data[3] = (2 * (rot.x * rot.y + rot.z * rot.w)) * scale.x;
  data[4] = (1 - 2 * (rot.x * rot.x + rot.z * rot.z)) * scale.y;
  data[5] = p_orig.y;
  tmp.y   = 2 * (rot.y * rot.z - rot.x * rot.w);

  data[6] = (2 * (rot.x * rot.z - rot.y * rot.w)) * scale.x;
  data[7] = 2 * (rot.x * rot.w + rot.y * rot.z) * scale.y;
  data[8] = p_orig.z;
  tmp.z   = 1 - 2 * (rot.x * rot.x + rot.y * rot.y);

  mat3<T> t;
#pragma unroll
  for (int i = 0; i < 3; ++i) {
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      T res = 0;
#pragma unroll
      for (int k = 0; k < 3; ++k) {
        res += Tw2v._data[i * 4 + k] * data[k * 3 + j];
      }
      t._data[i * 3 + j] = res;
    }
  }
  t._data[8] += Tw2v._data[11];
  matmul<T, 3, 3, 3>(Tv2s._data, t._data, matrices._data);
  normal = xfm_v_4x3(tmp, Tw2v._data);
}

template <typename T>
__device__ void compute_transmat_backward(const T2<T>& scale, const vec4<T>& q, const mat4<T>& Tw2v,
    const mat3<T>& Tv2s, mat3<T>& grad_matrix, vec3<T>& grad_normal, vec3<T>& grad_point, T2<T>& grad_scale,
    vec4<T>& grad_rot) {
  mat3<T> grad_T, grad_L;
  zero_mat<T, 3, 3>(grad_L._data);
  matmul_tn<T, 3, 3, 3>(Tv2s._data, grad_matrix._data, grad_L._data);

  auto dR = grad_T._data;
#pragma unroll
  for (int i = 0; i < 3; ++i) {
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      T res = 0;
#pragma unroll
      for (int k = 0; k < 3; ++k) res += Tw2v._data[k * 4 + i] * grad_L._data[k * 3 + j];
      dR[i * 3 + j] = res;
    }
  }

  grad_point = {dR[2], dR[5], dR[8]};
  // normal = xfm_v_4x3(tmp, Tw2v._data);
  dR[2] = grad_normal.x * Tw2v._data[0] + grad_normal.y * Tw2v._data[4] + grad_normal.z * Tw2v._data[8];
  dR[5] = grad_normal.x * Tw2v._data[1] + grad_normal.y * Tw2v._data[5] + grad_normal.z * Tw2v._data[9];
  dR[8] = grad_normal.x * Tw2v._data[2] + grad_normal.y * Tw2v._data[6] + grad_normal.z * Tw2v._data[10];

  // L = R @ S
  grad_scale.x = dR[0] * (1 - 2 * (q.y * q.y + q.z * q.z)) + dR[3] * (2 * (q.x * q.y + q.z * q.w)) +
                 dR[6] * (2 * (q.x * q.z - q.y * q.w));
  grad_scale.y = dR[1] * (2 * (q.x * q.y - q.z * q.w)) + dR[4] * (1 - 2 * (q.x * q.x + q.z * q.z)) +
                 dR[7] * (2 * (q.x * q.w + q.y * q.z));
  dR[0] *= scale.x;
  dR[1] *= scale.y;
  dR[3] *= scale.x;
  dR[4] *= scale.y;
  dR[6] *= scale.x;
  dR[7] *= scale.y;

  grad_rot = dL_quaternion_to_R<T>(q, dR);
}

// Computing the bounding box of the 2D Gaussian and its center
// The center of the bounding box is used to create a low pass filter
template <typename T>
__device__ bool compute_aabb(mat3<T>& M, T cutoff, T2<T>& point_image, T2<T>& extent) {
  vec3<T> t = {cutoff * cutoff, cutoff * cutoff, -1.0f};
  T d       = t.dot(M[2] * M[2]);
  if (d == 0.0) return false;
  vec3<T> f = (1 / d) * t;

  T2<T> p = {f.dot(M[0] * M[2]), f.dot(M[1] * M[2])};

  T h_x = p.x * p.x - f.dot(M[0] * M[0]);
  T h_y = p.y * p.y - f.dot(M[1] * M[1]);

  point_image = {p.x, p.y};
  extent      = {sqrt(max(T(1e-4), h_x)), sqrt(max(T(1e-4), h_y))};
  return true;
}

template <typename T, int C = NUM_CHANNELS>
__global__ void gs_2d_fast_preprocess_forward_kernel(
    // constant
    int P, int sh_degree, int M, const int W, int H, const dim3 grid,
    // cameras
    const T* __restrict__ Tw2v, const T* __restrict__ Tv2c, const T* __restrict__ Tv2s,
    const T3<T>* __restrict__ cam_pos,
    // inputs
    const vec3<T>* __restrict__ orig_points, const T2<T>* __restrict__ scales, const T4<T>* __restrict__ rotations,
    const T* __restrict__ opacities, const vec3<T>* __restrict__ shs, const vec3<T>* __restrict__ sh_rest,
    const T* __restrict__ transMat_precomp,
    // outputs
    int* __restrict__ radii, T2<T>* __restrict__ points_xy_image, T* __restrict__ depths, T* __restrict__ transMats,
    T* __restrict__ inverse_M, vec3<T>* __restrict__ rgb, T4<T>* __restrict__ normal_opacity,
    int32_t* __restrict__ tiles_touched, const bool* __restrict__ culling) {
  auto idx = cg::this_grid().thread_rank();
  if (idx >= P) return;
  if (culling && culling[idx]) return;

  // Initialize radius and touched tiles to 0. If this isn't changed,
  // this Gaussian will not be processed further.
  radii[idx]         = 0;
  tiles_touched[idx] = 0;
  auto p_orig        = orig_points[idx];
  auto p_view        = xfm_p_4x3<T>(p_orig, Tw2v);
  // if (cg::this_grid().thread_rank() == debug_i) {
  //   printf("\033[31mp_view=%f %f %f\033[0m\n", p_view.x, p_view.y, p_view.z);
  // }
  if (p_view.z <= T(0.2)) return;  // in_frustum
  T opacity = opacities[idx];
  // if (T(255.0) * opacity < T(1.0)) return; // to reduce warp-dievery

  // Compute transformation matrix
  mat3<T> m;
  vec3<T> normal;
  if (transMat_precomp == nullptr) {
    compute_transmat<T>(p_orig, scales[idx], rotations[idx], *((mat4<T>*) Tw2v), *((mat3<T>*) Tv2s), m, normal);
  } else {
    m      = *((mat3<T>*) (transMat_precomp + idx * 9));
    normal = vec3<T>{0.0, 0.0, 1.0};
  }

#if DUAL_VISIABLE
  auto tmp = p_view * normal;
  T cos    = -(tmp.x + tmp.y + tmp.z);
  if (cos == 0) return;
  T multiplier = cos > 0 ? 1 : -1;
  normal       = multiplier * normal;
  // if (cg::this_grid().thread_rank() == debug_i) {
  //   printf("\033[31mcos=%f,multiplier=%f, normal=%f %f %f\033[0m\n", cos, multiplier, normal.x, normal.y, normal.z);
  // }
#endif

#if TIGHTBBOX  // no use in the paper, but it indeed help speeds.
  // the effective extent is now depended on the opacity of gaussian.
  T cutoff = sqrtf(max(9.f + 2.f * logf(opacities[idx]), 0.000001));
#else
  T cutoff = 3.0f;
#endif

  // Compute center and radius
  T2<T> point_image;
  T radius;
  {
    T2<T> extent;
    bool ok = compute_aabb(m, cutoff, point_image, extent);
    if (!ok) return;
    radius = ceil(max(max(extent.x, extent.y), cutoff * T(FilterSize)));
  }

  int2 rect_min, rect_max;
  getRect(point_image, radius, rect_min, rect_max, grid);
  // if (cg::this_grid().thread_rank() == debug_i) {
  //   printf("\033[31m\n");
  //   printf("radius=%f\n", radius);
  //   printf("rect_min=%u, %u, rect_max=%u, %u\n", rect_min.x, rect_min.y, rect_max.x, rect_max.y);
  //   printf("\033[0m\n");
  // }
  if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0) return;

  // Compute colors
  if (sh_degree >= 0) {
    rgb[idx] = SH_to_RGB<T>(sh_degree, p_orig, (vec3<T>*) cam_pos, shs + idx * (sh_rest == nullptr ? M : 1),
        sh_rest == nullptr ? nullptr : sh_rest + idx * M, true);
  }
#pragma unroll
  for (int i = 0; i < 9; ++i) transMats[idx * 9 + i] = m._data[i];
  inverse_M[idx * 9 + 0] = m(1, 1) * m(2, 2) - m(2, 1) * m(1, 2);
  inverse_M[idx * 9 + 1] = m(2, 1) * m(0, 2) - m(0, 1) * m(2, 2);
  inverse_M[idx * 9 + 2] = m(0, 1) * m(1, 2) - m(1, 1) * m(0, 2);
  inverse_M[idx * 9 + 3] = m(2, 0) * m(1, 2) - m(1, 0) * m(2, 2);
  inverse_M[idx * 9 + 4] = m(0, 0) * m(2, 2) - m(2, 0) * m(0, 2);
  inverse_M[idx * 9 + 5] = m(1, 0) * m(0, 2) - m(0, 0) * m(1, 2);
  inverse_M[idx * 9 + 6] = m(1, 0) * m(2, 1) - m(2, 0) * m(1, 1);
  inverse_M[idx * 9 + 7] = m(2, 0) * m(0, 1) - m(0, 0) * m(2, 1);
  inverse_M[idx * 9 + 8] = m(0, 0) * m(1, 1) - m(1, 0) * m(0, 1);
  depths[idx]            = p_view.z;
  radii[idx]             = (int) radius;
  points_xy_image[idx]   = point_image;
  // points_xy_image[idx] = {m(0, 2) / m(2, 2), m(1, 2) / m(2, 2)};
  normal_opacity[idx] = {normal.x, normal.y, normal.z, opacities[idx]};
  tiles_touched[idx]  = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}
// Generates one key/value pair for all Gaussian / tile overlaps. Run once per Gaussian (1:N mapping).
template <typename T, typename TK, typename TV>
__global__ void duplicateWithKeys(int P, const T2<T>* __restrict__ points_xy, const T* __restrict__ IM,
    const T* __restrict__ M, const T* __restrict__ depths, const int64_t* __restrict__ offsets,
    T* __restrict__ opacities, TK* __restrict__ gaussian_keys_unsorted, TV* gaussian_values_unsorted,
    int* __restrict__ radii, dim3 grid) {
  auto idx = cg::this_grid().thread_rank();
  if (idx >= P) return;
  // Generate no key/value pair for invisible Gaussians
  if (radii[idx] <= 0) return;
  // Find this Gaussian's offset in buffer for writing keys/values.
  int64_t off             = (idx == 0) ? 0 : offsets[idx - 1];
  const int64_t offset_to = offsets[idx];

  int2 rect_min, rect_max;
  getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);

  // depth = abs(depth); // use abs() to avoid negiative depth for OpenGL
  float depth = static_cast<float>(depths[idx]);
  T d         = -T(2) * ln2 * (T(7.9886846867721655) + fast_lg2(opacities[idx]));

  IM = IM + idx * 9;
  // if (idx == debug_i)
  //   printf("\033[31mm=%f %f %f %f %f %f %f %f %f; d=%f, opac=%f\033[0m\n", IM[0], IM[1], IM[2], IM[3], IM[4], IM[5],
  //       IM[6], IM[7], IM[8], d, opacities[idx]);
  T Wxx = IM[0] * IM[0] + IM[3] * IM[3] + d * IM[6] * IM[6];
  T Wyy = IM[1] * IM[1] + IM[4] * IM[4] + d * IM[7] * IM[7];
  T W__ = IM[2] * IM[2] + IM[5] * IM[5] + d * IM[8] * IM[8];
  T Wxy = 2 * (IM[0] * IM[1] + IM[3] * IM[4] + d * IM[6] * IM[7]);
  T Wx_ = 2 * (IM[0] * IM[2] + IM[3] * IM[5] + d * IM[6] * IM[8]);
  T Wy_ = 2 * (IM[1] * IM[2] + IM[4] * IM[5] + d * IM[7] * IM[8]);
  // if (idx == debug_i) printf("\033[31mW=%f %f %f %f %f %f\033[0m\n", Wxx, Wyy, Wxy, Wx_, Wy_, W__);

  Wxx        = Wxx * BLOCK_X * BLOCK_X;
  Wyy        = Wyy * BLOCK_Y * BLOCK_Y;
  Wxy        = Wxy * BLOCK_X * BLOCK_Y;
  Wx_        = Wx_ * BLOCK_X;
  Wy_        = Wy_ * BLOCK_Y;
  auto judge = [&](T u, T v) { return Wxx * u * u + Wyy * v * v + Wxy * u * v + Wx_ * u + Wy_ * v + W__ <= 0; };
  for (int x = rect_min.x; x < rect_max.x; ++x) {
    for (int y = rect_min.y; y < rect_max.y; ++y) {
      if (judge(x, y) || judge(x + 1, y) || judge(x, y + 1) || judge(x + 1, y + 1)) {
        TK key = y * grid.x + x;
        if constexpr (std::is_same<TK, int64_t>::value || std::is_same<TK, uint64_t>::value) {
          key <<= 32;
          key |= *((uint32_t*) &depth);
          // key |= reinterpret_cast<uint32_t>(&depth);
        } else {
          key += depth * 0.25 + 0.5;  // [key].[depth], depth: [-1, 1] -> [0.25, 0.75]
        }
        gaussian_keys_unsorted[off]   = key;
        gaussian_values_unsorted[off] = idx;
        off++;
      }
    }
  }
  assert(off <= offset_to);
  TV value = (TV) -1;
  TK key   = TK(grid.x * grid.y);
  if constexpr (std::is_same<TK, int64_t>::value || std::is_same<TK, uint64_t>::value) key = key << 32;
  for (; off < offset_to; ++off) {
    gaussian_keys_unsorted[off]   = key;
    gaussian_values_unsorted[off] = value;
  }
}

// Generates one key/value pair for all Gaussian / tile overlaps. Run once per Gaussian (1:N mapping).
template <typename T, typename TK, typename TV>
__global__ void duplicateWithKeys_v2(int P, const T2<T>* __restrict__ points_xy, const T* __restrict__ IM,
    const T* __restrict__ M, const T* __restrict__ depths, const int64_t* __restrict__ offsets,
    T* __restrict__ opacities, TK* __restrict__ gaussian_keys_unsorted, TV* gaussian_values_unsorted,
    int* __restrict__ radii, dim3 grid) {
  auto idx = cg::this_grid().thread_rank();
  if (idx >= P) return;
  // Generate no key/value pair for invisible Gaussians
  if (radii[idx] <= 0) return;
  // Find this Gaussian's offset in buffer for writing keys/values.
  int64_t off             = (idx == 0) ? 0 : offsets[idx - 1];
  const int64_t offset_to = offsets[idx];

  int2 rect_min, rect_max;
  getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);

  T2<T> c     = {M[idx * 9 + 2] / M[idx * 9 + 8], M[idx * 9 + 5] / M[idx * 9 + 8]};  // points_xy[idx];
  float depth = static_cast<float>(depths[idx]);
  T d         = -T(2) * ln2 * (T(7.9886846867721655) + fast_lg2(opacities[idx]));

  IM = IM + idx * 9;
  // if (idx == debug_i)
  //   printf("\033[31mm=%f %f %f %f %f %f %f %f %f; d=%f, opac=%f\033[0m\n", IM[0], IM[1], IM[2], IM[3], IM[4], IM[5],
  //       IM[6], IM[7], IM[8], d, opacities[idx]);
  T Wxx = IM[0] * IM[0] + IM[3] * IM[3] + d * IM[6] * IM[6];
  T Wyy = IM[1] * IM[1] + IM[4] * IM[4] + d * IM[7] * IM[7];
  T W__ = IM[2] * IM[2] + IM[5] * IM[5] + d * IM[8] * IM[8];
  T Wxy = 2 * (IM[0] * IM[1] + IM[3] * IM[4] + d * IM[6] * IM[7]);
  T Wx_ = 2 * (IM[0] * IM[2] + IM[3] * IM[5] + d * IM[6] * IM[8]);
  T Wy_ = 2 * (IM[1] * IM[2] + IM[4] * IM[5] + d * IM[7] * IM[8]);
  // if (idx == debug_i) printf("\033[31mW=%f %f %f %f %f %f\033[0m\n", Wxx, Wyy, Wxy, Wx_, Wy_, W__);

  bool march_x = abs(Wyy) > abs(Wxx);
  if (march_x) {
    T t = Wxx;
    Wxx = Wyy;
    Wyy = t;

    t   = Wx_;
    Wx_ = Wy_;
    Wy_ = t;

    c        = {c.y, c.x};
    rect_min = {rect_min.y, rect_min.x};
    rect_max = {rect_max.y, rect_max.x};
  }
  // int GX = march_x ? grid.x : grid.y;
  // int GY = march_x ? grid.y : grid.x;
  T BX = march_x ? BLOCK_X : BLOCK_Y;
  T BY = march_x ? BLOCK_Y : BLOCK_X;

  Wxx = Wxx * BX * BX;
  Wyy = Wyy * BY * BY;
  Wxy = Wxy * BX * BY;
  Wx_ = Wx_ * BX;
  Wy_ = Wy_ * BY;
  // if (Wxx < 0) {
  //   Wxx = -Wxx;
  //   Wyy = -Wyy;
  //   Wxy = -Wxy;
  //   Wx_ = -Wx_;
  //   Wy_ = -Wy_;
  //   W__ = -W__;
  // }
  // if (idx == debug_i) printf("\033[31mcenter: %f %f; Wxx=%f\033[0m\n", c.x, c.y, Wxx);
  T y0  = min(max(T(rect_min.y), floor(c.y / BY)), T(rect_max.y));
  T x0  = -(Wxy * y0 + Wx_) / (2 * Wxx);
  T dx  = -Wxy * BY / (2 * Wxx * BX);
  T dy  = 1;
  int l = min(max(rect_min.x - 1, int(floor(x0))), rect_max.x);
  int r = l + 1;

  auto judge = [&](T u, T v) { return Wxx * u * u + Wyy * v * v + Wxy * u * v + Wx_ * u + Wy_ * v + W__ <= 0; };
  while (l >= rect_min.x && judge(l, y0)) l--;
  while (r <= rect_max.x && judge(r, y0)) r++;
  // if (idx == debug_i) printf("\033[31m(%f %f), [l, r] = [%d %d]\033[0m\n", x0, y0, l, r);
  do {
    x0     = x0 + dx;
    y0     = y0 + dy;
    int l2 = max(rect_min.x - 1, min(l, int(floor(x0))));
    while (l2 >= rect_min.x && judge(l2, y0)) l2--;
    while (l2 + 1 < x0 && l2 <= rect_max.x && !judge(l2 + 1, y0)) l2++;
    int r2 = min(max(r, int(ceil(x0))), rect_max.x);
    while (r2 <= rect_max.x && judge(r2, y0)) r2++;
    while (x0 < r2 - 1 && r2 >= rect_min.x && !judge(r2 - 1, y0)) r2--;
    int y = int(y0) - 1;
    if (rect_min.y <= y && y < rect_max.y) {
      for (int x = max(rect_min.x, min(l, l2)); x < min(max(r, r2), rect_max.x); ++x) {
        TK key = march_x ? x * grid.x + y : y * grid.x + x;
        if constexpr (std::is_same<TK, int64_t>::value || std::is_same<TK, uint64_t>::value) {
          key <<= 32;
          key |= *((uint32_t*) &depth);
          // key |= reinterpret_cast<uint32_t>(&depth);
        } else {
          key += depth * 0.25 + 0.5;  // [key].[depth], depth: [-1, 1] -> [0.25, 0.75]
        }
        gaussian_keys_unsorted[off]   = key;
        gaussian_values_unsorted[off] = idx;
        off++;
      }
    }
    l = l2;
    r = r2;
    // if (idx == debug_i) printf("\033[31m(%f %f), [l, r] = [%d %d]\033[0m\n", x0, y0, l, r);
  } while (y0 < rect_max.y && judge(x0, y0));

  y0 = min(max((T) rect_min.y, floor(c.y / BY)), (T) rect_max.y);
  x0 = -(Wxy * y0 + Wx_) / (2 * Wxx);
  l  = min(max(-rect_min.x - 1, int(floor(x0))), rect_max.x);
  r  = l + 1;
  while (l >= rect_min.x && judge(l, y0)) l--;
  while (r <= rect_max.x && judge(r, y0)) r++;
  while (y0 >= dy && judge(x0, y0)) {
    x0     = x0 - dx;
    y0     = y0 - dy;
    int l2 = max(-1, min(l, int(floor(x0))));
    while (l2 >= rect_min.x && judge(l2, y0)) l2--;
    while (l2 + 1 < x0 && l2 <= rect_max.x && !judge(l2 + 1, y0)) l2++;
    int r2 = min(rect_max.x, max(r, int(ceil(x0))));
    while (r2 <= rect_max.x && judge(r2, y0)) r2++;
    while (x0 < r2 - 1 && r2 >= rect_min.x && !judge(r2 - 1, y0)) r2--;
    int y = int(y0);
    if (rect_min.y <= y && y < rect_max.y) {
      for (int x = max(min(l, l2), rect_min.x); x < min(max(r, r2), rect_max.x); ++x) {
        TK key = march_x ? x * grid.x + y : y * grid.x + x;
        if constexpr (std::is_same<TK, int64_t>::value || std::is_same<TK, uint64_t>::value) {
          key <<= 32;
          key |= *((uint32_t*) &depth);
          // key |= reinterpret_cast<uint32_t>(&depth);
        } else {
          key += depth * 0.25 + 0.5;  // [key].[depth], depth: [-1, 1] -> [0.25, 0.75]
        }
        gaussian_keys_unsorted[off]   = key;
        gaussian_values_unsorted[off] = idx;
        off++;
      }
    }
    l = l2;
    r = r2;
    // if (idx == debug_i) printf("\033[31m(%f %f), [l, r] = [%d %d]\033[0m\n", x0, y0, l, r);
  }
  // if (idx == debug_i) printf("num_tiles=%d, max=%d\n", (int) off, int(offset_to));
  // depth = abs(depth); // use abs() to avoid negiative depth for OpenGL
  assert(off <= offset_to);
  TV value = (TV) -1;
  TK key   = TK(grid.x * grid.y);
  if constexpr (std::is_same<TK, int64_t>::value || std::is_same<TK, uint64_t>::value) key = key << 32;
  for (; off < offset_to; ++off) {
    gaussian_keys_unsorted[off]   = key;
    gaussian_values_unsorted[off] = value;
  }
}
// Check keys to see if it is at the start/end of one tile's range in the full sorted list.
// If yes, write start/end of this tile. Run once per instanced (duplicated) Gaussian ID.
template <typename TK = uint64_t, typename T = int64_t>
__global__ void identifyTileRanges(int L, int num_tile, TK* point_list_keys, int32_t* values, T2<T>* ranges) {
  auto idx = cg::this_grid().thread_rank();
  if (idx >= L) return;

  // Read tile ID from key. Update start/end of tile range if at limit.
  TK key = point_list_keys[idx];
  uint32_t currtile, prevtile;
  if constexpr (std::is_same<TK, int64_t>::value || std::is_same<TK, uint64_t>::value) {
    currtile = key >> 32;
  } else {
    currtile = floor(key);
  }
  // if (values[idx] == debug_i) printf("\033[35mtile_id=%u\n\033[0m", currtile);

  bool valid_tile = currtile < num_tile;
  if (idx == 0) {
    if (valid_tile) ranges[currtile].x = 0;
  } else {
    if constexpr (std::is_same<TK, int64_t>::value || std::is_same<TK, uint64_t>::value) {
      prevtile = point_list_keys[idx - 1] >> 32;
    } else {
      prevtile = floor(point_list_keys[idx - 1]);
    }
    if (currtile != prevtile) {
      ranges[prevtile].y = idx;
      if (valid_tile) ranges[currtile].x = idx;
    }
  }
  if (idx == L - 1 && valid_tile) ranges[currtile].y = L;
}

template <typename T = float>
__device__ void compute_aabb_backward(
    T cut_off, const mat3<T>& M, const T2<T>& dL_dmean2D, const vec3<T>& p_orig, mat3<T>& dL_dM) {
  // if (dL_dmean2D.x == 0 && dL_dmean2D.y == 0) return;

  vec3<T> t_vec  = {cut_off * cut_off, cut_off * cut_off, T(-1.0)};
  T d            = t_vec.dot(M.value[2] * M.value[2]);
  vec3<T> f_vec  = t_vec * (T(1.0) / d);
  vec3<T> dL_dT0 = dL_dmean2D.x * f_vec * M.value[2];
  vec3<T> dL_dT1 = dL_dmean2D.y * f_vec * M.value[2];
  vec3<T> dL_dT2 = dL_dmean2D.x * f_vec * M.value[0] + dL_dmean2D.y * f_vec * M.value[1];
  vec3<T> dL_df  = dL_dmean2D.x * M.value[0] * M.value[2] + dL_dmean2D.y * M.value[1] * M.value[2];
  T dL_dd        = dL_df.dot(f_vec) * -(T(1.0) / d);
  vec3<T> dd_dT2 = t_vec * M.value[2] * 2.0f;
  dL_dT2 += dL_dd * dd_dT2;
  dL_dM[0] += dL_dT0;
  dL_dM[1] += dL_dT1;
  dL_dM[2] += dL_dT2;
}

template <typename T, int C>
__global__ void gs_2d_fast_preprocess_backward_kernel(
    // constant scalar
    int P, int D, int M, int W, int H,
    // cameras
    const T* Tw2v, const T* Tv2s, const T3<T>* campos,
    // inputs
    const vec3<T>* __restrict__ means3D, const T2<T>* __restrict__ scales, const vec4<T>* __restrict__ rotations,
    const vec3<T>* __restrict__ shs, const vec3<T>* __restrict__ sh_rest,
    // outputs
    const int* __restrict__ radii, const mat3<T>* __restrict__ transMats, const vec3<T>* __restrict__ colors,
    const vec4<T>* __restrict__ dnormal_opacity,
    // grad_outputs
    const mat3<T>* __restrict__ dL_dtransMats, const mat3<T>* __restrict__ dL_dinverse_M,
    const T* __restrict__ dL_dnormal_opacity, T* __restrict__ dL_dcolors, T2<T>* __restrict__ dL_dmean2Ds,
    // grad input
    vec3<T>* __restrict__ dL_dshs, vec3<T>* __restrict__ dL_dsh_rest, vec3<T>* __restrict__ dL_dmean3Ds,
    T2<T>* __restrict__ dL_dscales, vec4<T>* __restrict__ dL_drots, T* __restrict__ dL_dopacity) {
  auto idx = cg::this_grid().thread_rank();
  if (idx >= P || radii[idx] <= 0) return;
  dL_dopacity[idx] = dL_dnormal_opacity[idx * 4 + 3];

  mat3<T> m  = transMats[idx];
  mat3<T> gT = dL_dtransMats[idx];
  mat3<T> gM = dL_dinverse_M[idx];
  gT(0, 0) += gM(1, 1) * m(2, 2) + gM(2, 2) * m(1, 1) - gM(1, 2) * m(1, 2) - gM(2, 1) * m(2, 1);
  gT(0, 1) += gM(0, 2) * m(1, 2) + gM(2, 1) * m(2, 0) - gM(0, 1) * m(2, 2) - gM(2, 2) * m(1, 0);
  gT(0, 2) += gM(0, 1) * m(2, 1) + gM(1, 2) * m(1, 0) - gM(0, 2) * m(1, 1) - gM(1, 1) * m(2, 0);
  gT(1, 0) += gM(1, 2) * m(0, 2) + gM(2, 0) * m(2, 1) - gM(1, 0) * m(2, 2) - gM(2, 2) * m(0, 1);
  gT(1, 1) += gM(0, 0) * m(2, 2) + gM(2, 2) * m(0, 0) - gM(0, 2) * m(0, 2) - gM(2, 0) * m(2, 0);
  gT(1, 2) += gM(0, 2) * m(0, 1) + gM(1, 0) * m(2, 0) - gM(0, 0) * m(2, 1) - gM(1, 2) * m(0, 0);
  gT(2, 0) += gM(1, 0) * m(1, 2) + gM(2, 1) * m(0, 1) - gM(1, 1) * m(0, 2) - gM(2, 0) * m(1, 1);
  gT(2, 1) += gM(0, 1) * m(0, 2) + gM(2, 0) * m(1, 0) - gM(0, 0) * m(1, 2) - gM(2, 1) * m(0, 0);
  gT(2, 2) += gM(0, 0) * m(1, 1) + gM(1, 1) * m(0, 0) - gM(0, 1) * m(0, 1) - gM(1, 0) * m(1, 0);

  compute_aabb_backward<T>(T(3.0), m, dL_dmean2Ds[idx], means3D[idx], gT);

  vec3<T> dL_dnormal = *((vec3<T>*) (dL_dnormal_opacity + idx * 4));

#if DUAL_VISIABLE
  auto rot       = rotations[idx];
  vec3<T> normal = {2 * (rot.y * rot.w + rot.x * rot.z), 2 * (rot.y * rot.z - rot.x * rot.w),
      1 - 2 * (rot.x * rot.x + rot.y * rot.y)};
  normal         = xfm_v_4x3(normal, Tw2v);
  vec3<T> p_view = xfm_p_4x3(means3D[idx], Tw2v);
  vec3<T> tmp    = p_view * normal;
  T cos          = -(tmp.x + tmp.y + tmp.z);
  T multiplier   = cos > 0 ? 1 : -1;
  dL_dnormal     = multiplier * dL_dnormal;
#endif
  compute_transmat_backward<T>(scales[idx], rotations[idx], *((mat4<T>*) Tw2v), *((mat3<T>*) Tv2s), gT, dL_dnormal,
      dL_dmean3Ds[idx], dL_dscales[idx], dL_drots[idx]);
  // if (idx == 70) {
  //   printf("\033[31mdL_dM=%f %f %f;%f %f %f; %f %f %f\033[0m\n", gT(0, 0), gT(0, 1), gT(0, 2), gT(1, 0), gT(1, 1),
  //       gT(1, 2), gT(2, 0), gT(2, 1), gT(2, 2));
  //   printf("\033[31mdL_dmean3D=%f,%f,%f \033[0m\n", dL_dmean3Ds[idx].x, dL_dmean3Ds[idx].y, dL_dmean3Ds[idx].z);
  // }

  vec3<T> grad_dir;
  if (D >= 0) {
    int M1   = sh_rest == nullptr ? M : 1;
    grad_dir = SH_to_RGB_backward<T>(D, means3D[idx], (vec3<T>*) campos, colors[idx], shs + idx * M1,
        sh_rest == nullptr ? nullptr : sh_rest + idx * M, ((vec3<T>*) dL_dcolors)[idx],
        dL_dshs == nullptr ? nullptr : dL_dshs + idx * M1, dL_dsh_rest == nullptr ? nullptr : dL_dsh_rest + idx * M,
        true, true);

    dL_dmean3Ds[idx] = dL_dmean3Ds[idx] + grad_dir;
  }

  // hack the gradient here for densitification
  T depth            = m._data[8];
  dL_dmean2Ds[idx].x = gT._data[2] * depth * 0.5 * T(W);  // to ndc
  dL_dmean2Ds[idx].y = gT._data[5] * depth * 0.5 * T(H);  // to ndc
}  // namespace GaussianRasterizer

vector<Tensor> GS_2D_fast_preprocess_forward(int W, int H, int sh_degree, bool is_opengl,
    // Gaussians
    Tensor means3D, Tensor scales, Tensor rotations, Tensor opacities, Tensor shs, torch::optional<Tensor> sh_rest,
    // cameras
    Tensor Tw2v, Tensor Tv2c, Tensor Tv2s, Tensor cam_pos,
    // pre-computed tensors
    torch::optional<Tensor> trans_precomp, torch::optional<Tensor> means2D, torch::optional<Tensor> culling,
    bool debug) {
  CHECK_INPUT(means3D);
  BCNN_ASSERT(means3D.ndimension() == 2 && means3D.size(-1) == 3, "Error shape for means3D");
  int P = means3D.size(0);
  int M = 0;
  CHECK_INPUT(scales);
  CHECK_INPUT(rotations);
  CHECK_INPUT(opacities);
  CHECK_INPUT(shs);
  bool has_sh_rest = false;
  if (sh_rest.has_value() && sh_rest.value().numel() > 0) {
    CHECK_INPUT(sh_rest.value());
    BCNN_ASSERT(sh_rest.value().ndimension() == 3 && sh_rest.value().size(0) == P &&
                    sh_rest.value().size(1) >= (1 + sh_degree) * (1 + sh_degree) - 1 && sh_rest.value().size(2) == 3,
        "Error shape for sh features");
    M           = sh_rest.value().size(1);
    has_sh_rest = true;
  } else if (sh_degree >= 0) {
    BCNN_ASSERT(shs.ndimension() == 3 && shs.size(0) == P && shs.size(1) >= (1 + sh_degree) * (1 + sh_degree) &&
                    shs.size(2) == 3,
        "Error shape for sh features");
    M = shs.size(1);
  }

  CHECK_INPUT(Tw2v);
  CHECK_INPUT(Tv2c);
  CHECK_INPUT(Tv2s);
  CHECK_INPUT(cam_pos);
  if (means2D.has_value()) {
    CHECK_INPUT(means2D.value());
    CHECK_SHAPE(means2D.value(), P, 2);
  }
  if (culling.has_value()) {
    CHECK_INPUT(culling.value());
    CHECK_SHAPE(culling.value(), P);
  }

  CHECK_SHAPE(scales, P, 2);
  CHECK_SHAPE(rotations, P, 4);
  CHECK_SHAPE(opacities, P, 1);
  CHECK_SHAPE(Tw2v, 4, 4);
  CHECK_SHAPE(Tv2c, 4, 4);
  CHECK_SHAPE(Tv2s, 3, 3);
  CHECK_SHAPE(cam_pos, 3);
  if (sh_degree < 0) CHECK_SHAPE(shs, P, 3);

  Tensor colors        = sh_degree < 0 ? shs : torch::zeros_like(means3D);
  Tensor trans_mat     = trans_precomp.has_value() ? trans_precomp.value() : torch::zeros({P, 3, 3}, means3D.options());
  Tensor inverse_m     = torch::zeros_like(trans_mat);
  Tensor means2D_      = means2D.has_value() ? means2D.value() : torch::zeros({P, 2}, means3D.options());
  Tensor depths        = torch::zeros({P}, means3D.options());
  Tensor radii         = torch::zeros({P}, means3D.options().dtype(torch::kInt32));
  Tensor tiles_touched = torch::zeros({P}, means3D.options().dtype(torch::kInt32));
  Tensor normal_opacity = torch::zeros({P, 4}, means3D.options());

  dim3 tile_grid((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, 1);

  Tensor ranges         = torch::zeros({tile_grid.x * tile_grid.y, 2}, means3D.options().dtype(torch::kInt64));
  Tensor point_offsets  = torch::zeros(P, means3D.options().dtype(torch::kInt64));
  Tensor sorted_key     = torch::zeros(0, means3D.options().dtype(torch::kInt64));
  Tensor sorted_value   = torch::zeros(0, means3D.options().dtype(torch::kInt32));
  Tensor unsorted_key   = torch::zeros(0, means3D.options().dtype(torch::kInt64));
  Tensor unsorted_value = torch::zeros(0, means3D.options().dtype(torch::kInt32));
  Tensor temp_scan      = torch::zeros(0, means3D.options().dtype(torch::kInt8));
  Tensor temp_sort      = torch::zeros(0, means3D.options().dtype(torch::kInt8));
  Tensor bucket_offset  = torch::zeros(tile_grid.x * tile_grid.y, means3D.options().dtype(torch::kInt64));

  // dim3 block(BLOCK_X, BLOCK_Y, 1);
  //   AT_DISPATCH_FLOATING_TYPES(means3D.scalar_type(), "GS_2D_preprocess_forward (CUDA)", [&] {
  using scalar_t = float;
  gs_2d_fast_preprocess_forward_kernel<scalar_t, NUM_CHANNELS> KERNEL_ARG((P + 255) / 256, 256)(  //
      P, sh_degree, M, W, H, tile_grid,                                                           //
      Tw2v.data_ptr<scalar_t>(),                                                                  //
      Tv2c.data_ptr<scalar_t>(),                                                                  //
      Tv2s.data_ptr<scalar_t>(),                                                                  //
      (const T3<scalar_t>*) cam_pos.data_ptr<scalar_t>(),                                         //
      (vec3<scalar_t>*) means3D.data_ptr<scalar_t>(),                                             //
      (const T2<scalar_t>*) scales.data_ptr<scalar_t>(),                                          //
      (const T4<scalar_t>*) rotations.data_ptr<scalar_t>(),                                       //
      opacities.data_ptr<scalar_t>(),                                                             //
      (vec3<scalar_t>*) shs.data_ptr<scalar_t>(),                                                 //
      (vec3<scalar_t>*) (has_sh_rest ? sh_rest.value().data_ptr<scalar_t>() : nullptr),           //
      trans_precomp.has_value() ? trans_precomp.value().data_ptr<scalar_t>() : nullptr,           //
      radii.data_ptr<int>(),                                                                      //
      (T2<scalar_t>*) means2D_.data_ptr<scalar_t>(),                                              //
      depths.data_ptr<scalar_t>(),                                                                //
      trans_mat.data_ptr<scalar_t>(),                                                             //
      inverse_m.data_ptr<scalar_t>(),
      (vec3<scalar_t>*) colors.data_ptr<scalar_t>(),                    //
      (T4<scalar_t>*) normal_opacity.data_ptr<scalar_t>(),              //
      tiles_touched.data_ptr<int32_t>(),                                //
      culling.has_value() ? culling.value().data_ptr<bool>() : nullptr  //
  );
  if (debug) {
    cudaDeviceSynchronize();
    CHECK_CUDA_ERROR("gs_2d_preprocess_forward_kernel");
  }

  // Compute prefix sum over full list of touched tile counts by Gaussians
  // E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
  auto po_ptr = point_offsets.data_ptr<int64_t>();
  size_t scan_size;
  cub::DeviceScan::InclusiveSum(nullptr, scan_size, tiles_touched.data_ptr<int32_t>(), po_ptr, P);
  temp_scan.resize_(scan_size);
  cub::DeviceScan::InclusiveSum(temp_scan.data_ptr(), scan_size, tiles_touched.data_ptr<int32_t>(), po_ptr, P);
  if (debug) {
    cudaDeviceSynchronize();
    CHECK_CUDA_ERROR("InclusiveSum");
  }
  // Retrieve total number of Gaussian instances to launch and resize aux buffers
  int64_t num_rendered;
  cudaMemcpy(&num_rendered, po_ptr + P - 1, sizeof(int64_t), cudaMemcpyDeviceToHost);
  if (debug) {
    // printf("num rendered=%ld\n", num_rendered);
    cudaDeviceSynchronize();
    CHECK_CUDA_ERROR("cudaMemcpy");
  }

  unsorted_key.resize_(num_rendered);
  unsorted_value.resize_(num_rendered);
  sorted_key.resize_(num_rendered);
  sorted_value.resize_(num_rendered);
  typedef int64_t TK;
  TK* key_unsorted        = unsorted_key.data_ptr<TK>();
  int32_t* value_unsorted = unsorted_value.data_ptr<int32_t>();
  TK* key_sorted          = sorted_key.data_ptr<TK>();
  int32_t* value_sorted   = sorted_value.data_ptr<int32_t>();

  // For each instance to be rendered, produce adequate [ tile | depth ] key
  // and corresponding dublicated Gaussian indices to be sorted
  duplicateWithKeys<scalar_t, TK, int32_t> KERNEL_ARG((P + 255) / 256, 256)(P,
      (T2<scalar_t>*) (means2D_.data_ptr<scalar_t>()), inverse_m.data_ptr<scalar_t>(), trans_mat.data_ptr<scalar_t>(),
      depths.data_ptr<scalar_t>(), po_ptr, opacities.data_ptr<scalar_t>(), key_unsorted, value_unsorted,
      radii.data_ptr<int32_t>(), tile_grid);
  if (debug) {
    cudaDeviceSynchronize();
    CHECK_CUDA_ERROR("duplicateWithKeys");
  }

  int bit = getHigherMsb(tile_grid.x * tile_grid.y + 1);

  // Sort complete list of (duplicated) Gaussian indices by keys
  size_t sorting_size;
  cub::DeviceRadixSort::SortPairs(
      nullptr, sorting_size, key_unsorted, key_sorted, value_unsorted, value_sorted, num_rendered, 0, 32 + bit);
  temp_sort.resize_(sorting_size);
  cub::DeviceRadixSort::SortPairs(temp_sort.data_ptr(), sorting_size, key_unsorted, key_sorted, value_unsorted,
      value_sorted, num_rendered, 0, 32 + bit);
  if (debug) {
    cudaDeviceSynchronize();
    CHECK_CUDA_ERROR("SortPairs");
  }
  // Identify start and end of per-tile workloads in sorted list
  if (num_rendered > 0) {
    identifyTileRanges<TK, int64_t> KERNEL_ARG((num_rendered + 255) / 256, 256)(
        num_rendered, tile_grid.x * tile_grid.y, key_sorted, value_sorted, (T2<int64_t>*) ranges.data_ptr<int64_t>());
    if (debug) {
      cudaDeviceSynchronize();
      CHECK_CUDA_ERROR("identifyTileRanges");
    }
  }

  //   });
  return {means2D_, colors, trans_mat, inverse_m, normal_opacity, depths, radii, ranges, sorted_value};
}

void GS_2D_fast_preprocess_backward(
    // information
    int W, int H, int sh_degree, bool is_opengl,
    // geometry tensors
    Tensor means3D, Tensor scales, Tensor rotations, Tensor opacities, Tensor shs, torch::optional<Tensor> sh_rest,
    // camera informations
    Tensor Tw2v, Tensor Tv2s, Tensor cam_pos,
    // internal tensors
    torch::optional<Tensor> trans_mat, Tensor radii, torch::optional<Tensor> colors, Tensor normal_opacity,
    // grad_outputs
    Tensor grad_mean2D, Tensor grad_trans, Tensor grad_inv_m, torch::optional<Tensor> grad_depth,
    torch::optional<Tensor> grad_colors, Tensor grad_normals,
    // grad_inputs
    Tensor& grad_means3D, torch::optional<Tensor>& grad_scales, torch::optional<Tensor>& grad_rotations,
    Tensor& grad_opacities, torch::optional<Tensor>& grad_shs, torch::optional<Tensor>& grad_sh_rest,
    // grad cameras
    torch::optional<Tensor> grad_Tw2v, torch::optional<Tensor> grad_campos) {
  CHECK_INPUT(grad_mean2D);
  if (grad_colors.has_value()) CHECK_INPUT(grad_colors.value());
  CHECK_INPUT(grad_normals);
  CHECK_INPUT(grad_means3D);
  CHECK_INPUT(grad_trans);
  if (grad_scales.has_value()) CHECK_INPUT(grad_scales.value());
  if (grad_rotations.has_value()) CHECK_INPUT(grad_rotations.value());
  CHECK_INPUT(grad_opacities);
  if (grad_shs.has_value()) {
    BCNN_ASSERT(colors.has_value(), "shs and clamped must not be None when need grad_shs");
    CHECK_INPUT(grad_shs.value());
  }
  if (grad_sh_rest.has_value()) CHECK_INPUT(grad_sh_rest.value());
  bool has_sh_rest = sh_rest.has_value() && sh_rest.value().numel() > 0;
  int P            = means3D.size(0);
  int D            = sh_degree;
  int M            = has_sh_rest ? sh_rest.value().size(1) : shs.size(1);
  CHECK_SHAPE(grad_means3D, P, 3);
  if (grad_scales.has_value()) CHECK_SHAPE(grad_scales.value(), P, 2);
  if (grad_rotations.has_value()) CHECK_SHAPE(grad_rotations.value(), P, 4);
  CHECK_SHAPE(grad_opacities, P, 1);
  if (grad_Tw2v.has_value()) CHECK_SHAPE(grad_Tw2v.value(), 4, 4);
  if (grad_campos.has_value()) CHECK_SHAPE(grad_campos.value(), 4, 4);

  using scalar_t = float;
  //   AT_DISPATCH_FLOATING_TYPES(means3D.scalar_type(), "GS_2D_preprocess_backward(CUDA)", [&] {
  gs_2d_fast_preprocess_backward_kernel<scalar_t, NUM_CHANNELS> KERNEL_ARG((P + 255) / 256, 256)(          //
      P, D, M, W, H,                                                                                       //
      Tw2v.data_ptr<scalar_t>(), Tv2s.data_ptr<scalar_t>(), (T3<scalar_t>*) cam_pos.data_ptr<scalar_t>(),  //

      (vec3<scalar_t>*) means3D.data_ptr<scalar_t>(),                                              //
      (T2<scalar_t>*) scales.data_ptr<scalar_t>(),                                                 //
      (vec4<scalar_t>*) rotations.data_ptr<scalar_t>(),                                            //
      (vec3<scalar_t>*) shs.data_ptr<scalar_t>(),                                                  //
      (vec3<scalar_t>*) (has_sh_rest ? sh_rest.value().data_ptr<scalar_t>() : nullptr),            //
      radii.data_ptr<int>(),                                                                       //
      trans_mat.has_value() ? (mat3<scalar_t>*) trans_mat.value().data_ptr<scalar_t>() : nullptr,  //
      colors.has_value() ? (vec3<scalar_t>*) colors.value().data_ptr<scalar_t>() : nullptr,        //
      (vec4<scalar_t>*) normal_opacity.data_ptr<scalar_t>(),                                       //

      (mat3<scalar_t>*) grad_trans.data_ptr<scalar_t>(),                             //
      (mat3<scalar_t>*) grad_inv_m.data_ptr<scalar_t>(),                             //
      grad_normals.data_ptr<scalar_t>(),                                             //
      grad_colors.has_value() ? grad_colors.value().data_ptr<scalar_t>() : nullptr,  //
      (T2<scalar_t>*) grad_mean2D.data_ptr<scalar_t>(),                              //

      grad_shs.has_value() ? (vec3<scalar_t>*) grad_shs.value().data_ptr<scalar_t>() : nullptr,              //
      grad_sh_rest.has_value() ? (vec3<scalar_t>*) grad_sh_rest.value().data_ptr<scalar_t>() : nullptr,      //
      (vec3<scalar_t>*) grad_means3D.data_ptr<scalar_t>(),                                                   //
      grad_scales.has_value() ? (T2<scalar_t>*) grad_scales.value().data_ptr<scalar_t>() : nullptr,          //
      grad_rotations.has_value() ? (vec4<scalar_t>*) grad_rotations.value().data_ptr<scalar_t>() : nullptr,  //
      grad_opacities.data_ptr<scalar_t>()                                                                    //
      //   grad_cov.has_value() ? grad_cov.value().data_ptr<scalar_t>() : nullptr,                                //
      //   grad_Tw2v.has_value() ? grad_Tw2v.value().data_ptr<scalar_t>() : nullptr,                              //
      //   grad_campos.has_value() ? grad_campos.value().data_ptr<scalar_t>() : nullptr
  );
  CHECK_CUDA_ERROR("gs_2d_preprocess_backward_kernel");
  //   });
}

template <typename T>
__global__ void gs_2d_fast_compute_trans_mat_forward_kernel(int P, const vec3<T>* __restrict__ means3D,
    const T2<T>* __restrict__ scales, const vec4<T>* __restrict__ rotations, const mat4<T> Tw2v, const mat3<T> Tv2s,
    T* __restrict__ trans_mat, vec3<T>* __restrict__ normals) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < P) {
    compute_transmat(means3D[idx], scales[idx], rotations[idx], Tw2v, Tv2s, ((mat3<T>*) trans_mat)[idx], normals[idx]);
  }
}

std::tuple<Tensor, Tensor> gs_2d_fast_compute_trans_mat_forward(
    Tensor means3D, Tensor scales, Tensor rotations, Tensor Tw2v, Tensor Tv2s) {
  int P = means3D.size(0);
  CHECK_INPUT(means3D);
  CHECK_INPUT(scales);
  CHECK_INPUT(rotations);
  CHECK_INPUT(Tw2v);
  CHECK_INPUT(Tv2s);
  CHECK_SHAPE(means3D, P, 3);
  CHECK_SHAPE(scales, P, 2);
  CHECK_SHAPE(rotations, P, 4);
  CHECK_SHAPE(Tw2v, 4, 4);
  CHECK_SHAPE(Tv2s, 3, 3);
  Tv2s = Tv2s.cpu();
  Tw2v = Tw2v.cpu();

  Tensor trans_mat = torch::zeros({P, 3, 3}, means3D.options());
  Tensor normals   = torch::zeros({P, 3}, means3D.options());
  AT_DISPATCH_FLOATING_TYPES(means3D.scalar_type(), "GS_2D_preprocess_backward(CUDA)", [&] {
    auto p_means3D = (vec3<scalar_t>*) means3D.data_ptr<scalar_t>();
    auto p_scales  = (T2<scalar_t>*) scales.data_ptr<scalar_t>();
    auto p_rot     = (vec4<scalar_t>*) rotations.data_ptr<scalar_t>();
    auto p_matrix  = trans_mat.data_ptr<scalar_t>();
    auto p_normal  = (vec3<scalar_t>*) normals.data_ptr<scalar_t>();
    auto p_Tw2v    = (mat4<scalar_t>*) Tw2v.data_ptr<scalar_t>();
    auto p_Tv2s    = (mat3<scalar_t>*) Tv2s.data_ptr<scalar_t>();
    if (means3D.is_cuda()) {
      gs_2d_fast_compute_trans_mat_forward_kernel<scalar_t> KERNEL_ARG(div_round_up(P, 256), 256)(
          P, p_means3D, p_scales, p_rot, *p_Tw2v, *p_Tv2s, p_matrix, p_normal);
    } else {
      // for (int p = 0; p < P; ++p) {
      //   compute_transmat(
      //       W, H, p_means3D[p], p_scales[p], p_rot[p], *p_Tw2c, *p_Tw2v, ((mat3<scalar_t>*) p_matrix)[p],
      //       p_normal[p]);
      // }
    }
  });
  return std::make_tuple(trans_mat, normals);
}

template <typename T>
__global__ void gs_2d_fast_compute_trans_mat_backward_kernel(int P, const vec3<T>* __restrict__ means3D,
    const T2<T>* __restrict__ scales, const vec4<T>* __restrict__ rotations, const mat4<T> Tw2v, const mat3<T> Tv2s,
    const T* __restrict__ grad_trans_mat, vec3<T>* __restrict__ grad_normals, vec3<T>* __restrict__ grad_means3D,
    T2<T>* __restrict__ grad_scales, vec4<T>* __restrict__ grad_rotations) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < P) {
    compute_transmat_backward(scales[idx], rotations[idx], Tw2v, Tv2s, ((mat3<T>*) grad_trans_mat)[idx],
        grad_normals[idx], grad_means3D[idx], grad_scales[idx], grad_rotations[idx]);
  }
}

vector<Tensor> gs_2d_fast_compute_trans_mat_backward(Tensor means3D, Tensor scales, Tensor rotations, Tensor Tw2v,
    Tensor Tv2s, Tensor grad_matrix, Tensor grad_normals) {
  int P = means3D.size(0);
  CHECK_INPUT(grad_matrix);
  CHECK_INPUT(grad_normals);
  CHECK_SHAPE(grad_matrix, P, 3, 3);
  CHECK_SHAPE(grad_normals, P, 3);
  Tv2s = Tv2s.cpu();
  Tw2v = Tw2v.cpu();

  Tensor grad_means3D   = torch::zeros({P, 3}, means3D.options());
  Tensor grad_scales    = torch::zeros({P, 2}, means3D.options());
  Tensor grad_rotations = torch::zeros({P, 4}, means3D.options());
  AT_DISPATCH_FLOATING_TYPES(means3D.scalar_type(), "gs_2d_compute_trans_mat_backward(CUDA)", [&] {
    auto p_means3D = (vec3<scalar_t>*) means3D.data_ptr<scalar_t>();
    auto p_scales  = (T2<scalar_t>*) scales.data_ptr<scalar_t>();
    auto p_rot     = (vec4<scalar_t>*) rotations.data_ptr<scalar_t>();
    auto p_Tw2v    = (mat4<scalar_t>*) Tw2v.data_ptr<scalar_t>();
    auto p_Tv2s    = (mat3<scalar_t>*) Tv2s.data_ptr<scalar_t>();
    auto g_matrix  = grad_matrix.data_ptr<scalar_t>();
    auto g_normal  = (vec3<scalar_t>*) grad_normals.data_ptr<scalar_t>();
    auto g_means3D = (vec3<scalar_t>*) grad_means3D.data_ptr<scalar_t>();
    auto g_scales  = (T2<scalar_t>*) grad_scales.data_ptr<scalar_t>();
    auto g_rot     = (vec4<scalar_t>*) grad_rotations.data_ptr<scalar_t>();
    if (means3D.is_cuda()) {
      gs_2d_fast_compute_trans_mat_backward_kernel<scalar_t> KERNEL_ARG(div_round_up(P, 256), 256)(
          P, p_means3D, p_scales, p_rot, *p_Tw2v, *p_Tv2s, g_matrix, g_normal, g_means3D, g_scales, g_rot);
    } else {
      // for (int p = 0; p < P; ++p) {
      //   compute_transmat_backward(W, H, p_scales[p], p_rot[p], *p_Tw2c, *p_Tw2v, ((mat3<scalar_t>*) g_matrix)[p],
      //       g_normal[p], g_means3D[p], g_scales[p], g_rot[p]);
      // }
    }
  });
  return {grad_means3D, grad_scales, grad_rotations};
}

REGIST_PYTORCH_EXTENSION(gs_2d_fast_preprocess, {
  m.def("gs_2d_fast_preprocess_forward", &GS_2D_fast_preprocess_forward, "gs 2d fast_preprocess_forward (CUDA)");
  m.def("gs_2d_fast_preprocess_backward", &GS_2D_fast_preprocess_backward, "gs 2d fast_preprocess_backward (CUDA)");
  m.def("gs_2d_fast_compute_transmat_forward", &gs_2d_fast_compute_trans_mat_forward,
      "gs 2d fast_compute_trans_mat_forward (CUDA, CPU)");
  m.def("gs_2d_fast_compute_trans_mat_backward", &gs_2d_fast_compute_trans_mat_backward,
      "gs 2d fast_compute_trans_mat_backward (CUDA, CPU)");
})

}  // namespace GaussianRasterizer
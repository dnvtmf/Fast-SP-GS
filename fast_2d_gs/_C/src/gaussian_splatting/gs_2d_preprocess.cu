/*
paper: 2D Gaussian Splatting for Geometrically Accurate Radiance Fields, SIGGRAPH 2024
code: https://github.com/hbb1/2d-gaussian-splatting
 */

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include "gaussian_render.h"
#include "ops_3d.h"
#include "spherical_harmonic.h"
#include "util.cuh"

namespace cg = cooperative_groups;
using namespace OPS_3D;
namespace GaussianRasterizer {
#define DUAL_VISIABLE 1
#define TIGHTBBOX 0

constexpr int debug_i       = 100821;
constexpr double FilterSize = 0.707106;  // sqrt(2) / 2;

// Compute a 2D-to-2D mapping matrix from a tangent plane into a image plane given a 2D gaussian parameters.
template <typename T>
__device__ void compute_transmat(const int W, const int H, const T3<T>& p_orig, const T2<T>& scale, const T4<T>& rot,
    const mat4<T>& Tw2c, const mat4<T>& Tw2v, mat3<T>& matrices, vec3<T>& normal) {
  mat4<T> L;  // L = quat_to_mat(rot) * scale_to_mat(scale)
  vec3<T> tmp;
  T* data = L._data;

  data[0] = (1 - 2 * (rot.y * rot.y + rot.z * rot.z)) * scale.x;
  data[1] = 2 * (rot.x * rot.y - rot.z * rot.w) * scale.y;
  data[2] = p_orig.x;
  tmp.x   = 2 * (rot.y * rot.w + rot.x * rot.z);

  data[4] = (2 * (rot.x * rot.y + rot.z * rot.w)) * scale.x;
  data[5] = (1 - 2 * (rot.x * rot.x + rot.z * rot.z)) * scale.y;
  data[6] = p_orig.y;
  tmp.y   = 2 * (rot.y * rot.z - rot.x * rot.w);

  data[8]  = (2 * (rot.x * rot.z - rot.y * rot.w)) * scale.x;
  data[9]  = 2 * (rot.x * rot.w + rot.y * rot.z) * scale.y;
  data[10] = p_orig.z;
  tmp.z    = 1 - 2 * (rot.x * rot.x + rot.y * rot.y);

  data[14] = 1;

  mat4<T> ndc2pix = {T(0.5 * W), T(0), T(0), T(0.5 * (W - 1)), T(0), T(0.5 * H), T(0), T(0.5 * (H - 1)), T(0), T(0),
      T(0), T(1), T(0), T(0), T(0), T(1)};
  mat4<T> tm      = ndc2pix * (Tw2c * L);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) matrices._data[i * 3 + j] = tm._data[i * 4 + j];
  normal = xfm_v_4x3(tmp, Tw2v._data);
  // if (cg::this_grid().thread_rank() == debug_i) {
  //   printf("\033[34m\n");
  //   printf("splat2world=torch.tensor([\n");
  //   for (int i = 0; i < 4; ++i) printf("  [%f, %f, %f, %f], \n", L[i].x, L[i].y, L[i].z, L[i].w);
  //   printf("])\n");
  //   printf("world2ndc=torch.tensor([\n");
  //   for (int i = 0; i < 4; ++i)
  //     printf("  [%f, %f, %f, %f],\n", Tw2c.value[i].x, Tw2c.value[i].y, Tw2c.value[i].z, Tw2c.value[i].w);
  //   printf("])\n");
  //   printf("ndc2pix=torch.tensor([\n");
  //   for (int i = 0; i < 4; ++i)
  //     printf("  [%f, %f, %f, %f],\n", ndc2pix.value[i].x, ndc2pix.value[i].y, ndc2pix.value[i].z,
  //     ndc2pix.value[i].w);
  //   printf("])\n");
  //   printf("T=torch.tensor([\n");
  //   for (int i = 0; i < 3; ++i)
  //     printf("  [%f, %f, %f],\n", matrices.value[i].x, matrices.value[i].y, matrices.value[i].z);
  //   printf("])\n");
  //   printf("normal=torch.tensor([%f, %f, %f])\n", normal.x, normal.y, normal.z);
  //   printf("tmp=%f %f %f\n", tmp.x, tmp.y, tmp.z);
  //   printf("Tw2v=torch.tensor([\n");
  //   for (int i = 0; i < 3; ++i) printf("  [%f, %f, %f],\n", Tw2v.value[i].x, Tw2v.value[i].y, Tw2v.value[i].z);
  //   printf("])\n");
  //   printf("\033[0m");
  // }
}

template <typename T>
__device__ void compute_transmat_backward(const int W, const int H, const T2<T>& scale, const T4<T>& q,
    const mat4<T>& Tw2c, const mat4<T>& Tw2v, mat3<T>& grad_matrix, vec3<T>& grad_normal, vec3<T>& grad_point,
    T2<T>& grad_scale, vec4<T>& grad_rot) {
  mat4<T> grad_T, grad_L;
  mat4<T> ndc2pix = {T(0.5 * W), T(0), T(0), T(0.5 * (W - 1)), T(0), T(0.5 * H), T(0), T(0.5 * (H - 1)), T(0), T(0),
      T(0), T(1), T(0), T(0), T(0), T(1)};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) grad_T._data[i * 4 + j] = grad_matrix._data[i * 3 + j];
  zero_mat<T, 4, 4>(grad_L._data);
  matmul_tn<T, 4, 4, 4>(ndc2pix._data, grad_T._data, grad_L._data);
  zero_mat<T, 4, 4>(grad_T._data);
  matmul_tn<T, 4, 4, 4>(Tw2c._data, grad_L._data, grad_T._data);

  auto dR = grad_T._data;

  grad_point = {dR[2], dR[6], dR[10]};
  // normal = xfm_v_4x3(tmp, Tw2v._data);
  dR[2]  = grad_normal.x * Tw2v._data[0] + grad_normal.y * Tw2v._data[4] + grad_normal.z * Tw2v._data[8];
  dR[6]  = grad_normal.x * Tw2v._data[1] + grad_normal.y * Tw2v._data[5] + grad_normal.z * Tw2v._data[9];
  dR[10] = grad_normal.x * Tw2v._data[2] + grad_normal.y * Tw2v._data[6] + grad_normal.z * Tw2v._data[10];

  // L = R @ S
  grad_scale.x = dR[0] * (1 - 2 * (q.y * q.y + q.z * q.z)) + dR[4] * (2 * (q.x * q.y + q.z * q.w)) +
                 dR[8] * (2 * (q.x * q.z - q.y * q.w));
  grad_scale.y = dR[1] * (2 * (q.x * q.y - q.z * q.w)) + dR[5] * (1 - 2 * (q.x * q.x + q.z * q.z)) +
                 dR[9] * (2 * (q.x * q.w + q.y * q.z));
  dR[0] *= scale.x;
  dR[1] *= scale.y;
  dR[4] *= scale.x;
  dR[5] *= scale.y;
  dR[8] *= scale.x;
  dR[9] *= scale.y;

  // dL_quaternion_to_R
  vec4<T>& dq = grad_rot;
  dq.x = 2 * (-2 * q.x * (dR[5] + dR[10]) + q.y * (dR[1] + dR[4]) + q.z * (dR[2] + dR[8]) + q.w * (dR[9] - dR[6]));
  dq.y = 2 * (q.x * (dR[1] + dR[4]) - 2 * q.y * (dR[0] + dR[10]) + q.z * (dR[6] + dR[9]) + q.w * (dR[2] - dR[8]));
  dq.z = 2 * (q.x * (dR[2] + dR[8]) + q.y * (dR[6] + dR[9]) - 2 * q.z * (dR[0] + dR[5]) + q.w * (dR[4] - dR[1]));
  dq.w = 2 * (q.x * (dR[9] - dR[6]) + q.y * (dR[2] - dR[8]) + q.z * (dR[4] - dR[1]));
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
  // if (cg::this_grid().thread_rank() == debug_i) {
  //   printf("\033[34m\n");
  //   // printf("t=%f, %f, %f\n", t.x, t.y, t.z);
  //   // printf("d=%f\n", d);
  //   // printf("f=%f, %f, %f\n", f.x, f.y, f.z);
  //   printf("p=%f %f; h=%f %f\n", p.x, p.y, extent.x, extent.y);
  //   // auto pp = p * p;
  //   // printf("p*p=%f %f\n", pp.x, pp.y);
  //   // auto tmp = T[0] * T[0];
  //   // printf("T[0] * T[0]=%f %f %f\n", tmp.x, tmp.y, tmp.z);
  //   // tmp = T[1] * T[1];
  //   // printf("T[1] * T[1]=%f %f %f\n", tmp.x, tmp.y, tmp.z);
  //   // printf("T[0]=%f %f %f; %f %f %f; %f %f %f\n", T[0].x, T[0].y, T[0].z, T[1].x, T[1].y, T[1].z, T[2].x, T[2].y,
  //   // T[2].z);
  //   printf("\033[0m\n");
  // }
  return true;
}

template <typename T, int C = NUM_CHANNELS>
__global__ void gs_2d_preprocess_forward_kernel(
    // constant
    int P, int sh_degree, int M, const int W, int H, const dim3 grid,
    // cameras
    const T* __restrict__ Tw2v, const T* __restrict__ Tv2c, const T* __restrict__ Tw2c,
    const T3<T>* __restrict__ cam_pos,
    // inputs
    const vec3<T>* __restrict__ orig_points, const T2<T>* __restrict__ scales, const T4<T>* __restrict__ rotations,
    const T* __restrict__ opacities, const vec3<T>* __restrict__ shs, const vec3<T>* __restrict__ sh_rest,
    const T* __restrict__ transMat_precomp,
    // outputs
    int* __restrict__ radii, T2<T>* __restrict__ points_xy_image, T* __restrict__ depths, T* __restrict__ transMats,
    vec3<T>* __restrict__ rgb, T4<T>* __restrict__ normal_opacity, int32_t* __restrict__ tiles_touched,
    const bool* __restrict__ culling) {
  auto idx = cg::this_grid().thread_rank();
  if (idx >= P) return;
  if (culling && culling[idx]) return;

  // Initialize radius and touched tiles to 0. If this isn't changed,
  // this Gaussian will not be processed further.
  radii[idx]         = 0;
  tiles_touched[idx] = 0;
  auto p_orig        = orig_points[idx];
  auto p_view        = xfm_p_4x3<T>(p_orig, Tw2v);
  if (p_view.z <= T(0.2)) return;  // in_frustum

  // Compute transformation matrix
  mat3<T> m;
  vec3<T> normal;
  if (transMat_precomp == nullptr) {
    compute_transmat<T>(W, H, p_orig, scales[idx], rotations[idx], *((mat4<T>*) Tw2c), *((mat4<T>*) Tw2v), m, normal);
  } else {
    T3<T>* T_ptr = (T3<T>*) transMat_precomp;
    m            = *((mat3<T>*) (T_ptr + idx * 9));
    normal       = vec3<T>{0.0, 0.0, 1.0};
  }

#if DUAL_VISIABLE
  auto tmp = p_view * normal;
  T cos    = -(tmp.x + tmp.y + tmp.z);
  if (cos == 0) return;
  T multiplier = cos > 0 ? 1 : -1;
  normal       = multiplier * normal;
  // if (cg::this_grid().thread_rank() == debug_i) {
  //   printf("\033[34mcos=%f,multiplier=%f, normal=%f %f %f\033[0m\n", cos, multiplier, normal.x, normal.y, normal.z);
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
  //   printf("\033[34m\n");
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
  if (transMat_precomp == nullptr)
    for (int i = 0; i < 9; ++i) transMats[idx * 9 + i] = m._data[i];
  depths[idx]          = p_view.z;
  radii[idx]           = (int) radius;
  points_xy_image[idx] = point_image;
  normal_opacity[idx]  = {normal.x, normal.y, normal.z, opacities[idx]};
  tiles_touched[idx]   = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
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
__global__ void gs_2d_preprocess_backward_kernel(
    // constant scalar
    int P, int D, int M, int W, int H,
    // cameras
    const T* Tw2v, const T* Tw2c, const T3<T>* campos,
    // inputs
    const vec3<T>* __restrict__ means3D, const T2<T>* __restrict__ scales, const T4<T>* __restrict__ rotations,
    const vec3<T>* __restrict__ shs, const vec3<T>* __restrict__ sh_rest,
    // outputs
    const int* __restrict__ radii, const mat3<T>* __restrict__ transMats, const vec3<T>* __restrict__ colors,
    const vec4<T>* __restrict__ dnormal_opacity,
    // grad_outputs
    const mat3<T>* __restrict__ dL_dtransMats, const T* __restrict__ dL_dnormal_opacity, T* __restrict__ dL_dcolors,
    T2<T>* __restrict__ dL_dmean2Ds,
    // grad input
    vec3<T>* __restrict__ dL_dshs, vec3<T>* __restrict__ dL_dsh_rest, vec3<T>* __restrict__ dL_dmean3Ds,
    T2<T>* __restrict__ dL_dscales, vec4<T>* __restrict__ dL_drots, T* __restrict__ dL_dopacity) {
  auto idx = cg::this_grid().thread_rank();
  if (idx >= P || radii[idx] <= 0) return;
  dL_dopacity[idx] = dL_dnormal_opacity[idx * 4 + 3];

  mat3<T> dL_dT = dL_dtransMats[idx];
  compute_aabb_backward<T>(T(3.0), transMats[idx], dL_dmean2Ds[idx], means3D[idx], dL_dT);

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
  compute_transmat_backward<T>(W, H, scales[idx], rotations[idx], *((mat4<T>*) Tw2c), *((mat4<T>*) Tw2v), dL_dT,
      dL_dnormal, dL_dmean3Ds[idx], dL_dscales[idx], dL_drots[idx]);
  // if (idx == 70) {
  //   auto gT = dL_dT;
  //   printf("\033[32mdL_dM=%f %f %f;%f %f %f; %f %f %f\033[0m\n", gT(0, 0), gT(0, 1), gT(0, 2), gT(1, 0), gT(1, 1),
  //       gT(1, 2), gT(2, 0), gT(2, 1), gT(2, 2));
  //   printf("\033[32mdL_dmean3D=%f,%f,%f \033[0m\n", dL_dmean3Ds[idx].x, dL_dmean3Ds[idx].y, dL_dmean3Ds[idx].z);
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
  T depth            = transMats[idx]._data[8];
  dL_dmean2Ds[idx].x = dL_dtransMats[idx]._data[2] * depth * 0.5 * T(W);  // to ndc
  dL_dmean2Ds[idx].y = dL_dtransMats[idx]._data[5] * depth * 0.5 * T(H);  // to ndc
}  // namespace GaussianRasterizer

vector<Tensor> GS_2D_preprocess_forward(int W, int H, int sh_degree, bool is_opengl,
    // Gaussians
    Tensor means3D, Tensor scales, Tensor rotations, Tensor opacities, Tensor shs, torch::optional<Tensor> sh_rest,
    // cameras
    Tensor Tw2v, Tensor Tv2c, Tensor cam_pos,
    // pre-computed tensors
    torch::optional<Tensor> trans_precomp, torch::optional<Tensor> means2D, torch::optional<Tensor> culling) {
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
  CHECK_SHAPE(cam_pos, 3);
  if (sh_degree < 0) CHECK_SHAPE(shs, P, 3);

  Tensor Tw2c = torch::matmul(Tv2c, Tw2v);

  Tensor colors        = sh_degree < 0 ? shs : torch::zeros_like(means3D);
  Tensor trans_mat     = trans_precomp.has_value() ? trans_precomp.value() : torch::zeros({P, 3, 3}, means3D.options());
  Tensor means2D_      = means2D.has_value() ? means2D.value() : torch::zeros({P, 2}, means3D.options());
  Tensor depths        = torch::zeros({P}, means3D.options());
  Tensor radii         = torch::zeros({P}, means3D.options().dtype(torch::kInt32));
  Tensor tiles_touched = torch::zeros({P}, means3D.options().dtype(torch::kInt32));
  Tensor normal_opacity = torch::zeros({P, 4}, means3D.options());

  dim3 tile_grid((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, 1);
  // dim3 block(BLOCK_X, BLOCK_Y, 1);
  //   AT_DISPATCH_FLOATING_TYPES(means3D.scalar_type(), "GS_2D_preprocess_forward (CUDA)", [&] {
  using scalar_t = float;
  gs_2d_preprocess_forward_kernel<scalar_t, NUM_CHANNELS> KERNEL_ARG((P + 255) / 256, 256)(  //
      P, sh_degree, M, W, H, tile_grid,                                                      //
      Tw2v.data_ptr<scalar_t>(),                                                             //
      Tv2c.data_ptr<scalar_t>(),                                                             //
      Tw2c.data_ptr<scalar_t>(),                                                             //
      (const T3<scalar_t>*) cam_pos.data_ptr<scalar_t>(),                                    //
      (vec3<scalar_t>*) means3D.data_ptr<scalar_t>(),                                        //
      (const T2<scalar_t>*) scales.data_ptr<scalar_t>(),                                     //
      (const T4<scalar_t>*) rotations.data_ptr<scalar_t>(),                                  //
      opacities.data_ptr<scalar_t>(),                                                        //
      (vec3<scalar_t>*) shs.data_ptr<scalar_t>(),                                            //
      (vec3<scalar_t>*) (has_sh_rest ? sh_rest.value().data_ptr<scalar_t>() : nullptr),      //
      trans_precomp.has_value() ? trans_precomp.value().data_ptr<scalar_t>() : nullptr,      //
      radii.data_ptr<int>(),                                                                 //
      (T2<scalar_t>*) means2D_.data_ptr<scalar_t>(),                                         //
      depths.data_ptr<scalar_t>(),                                                           //
      trans_mat.data_ptr<scalar_t>(),                                                        //
      (vec3<scalar_t>*) colors.data_ptr<scalar_t>(),                                         //
      (T4<scalar_t>*) normal_opacity.data_ptr<scalar_t>(),                                   //
      tiles_touched.data_ptr<int32_t>(),                                                     //
      culling.has_value() ? culling.value().data_ptr<bool>() : nullptr                       //
  );
  // cudaDeviceSynchronize();
  CHECK_CUDA_ERROR("gs_2d_preprocess_forward_kernel");
  //   });
  return {means2D_, colors, trans_mat, normal_opacity, depths, radii, tiles_touched};
}

void GS_2D_preprocess_backward(
    // information
    int W, int H, int sh_degree, bool is_opengl,
    // geometry tensors
    Tensor means3D, Tensor scales, Tensor rotations, Tensor opacities, Tensor shs, torch::optional<Tensor> sh_rest,
    // camera informations
    Tensor Tw2v, Tensor Tv2c, Tensor cam_pos,
    // internal tensors
    torch::optional<Tensor> trans_mat, Tensor radii, torch::optional<Tensor> colors, Tensor normal_opacity,
    // grad_outputs
    Tensor grad_mean2D, Tensor grad_trans, torch::optional<Tensor> grad_depth, torch::optional<Tensor> grad_colors,
    Tensor grad_normals,
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

  Tensor Tw2c = torch::matmul(Tv2c, Tw2v);

  using scalar_t = float;
  //   AT_DISPATCH_FLOATING_TYPES(means3D.scalar_type(), "GS_2D_preprocess_backward(CUDA)", [&] {
  gs_2d_preprocess_backward_kernel<scalar_t, NUM_CHANNELS> KERNEL_ARG((P + 255) / 256, 256)(               //
      P, D, M, W, H,                                                                                       //
      Tw2v.data_ptr<scalar_t>(), Tw2c.data_ptr<scalar_t>(), (T3<scalar_t>*) cam_pos.data_ptr<scalar_t>(),  //

      (vec3<scalar_t>*) means3D.data_ptr<scalar_t>(),                                              //
      (T2<scalar_t>*) scales.data_ptr<scalar_t>(),                                                 //
      (T4<scalar_t>*) rotations.data_ptr<scalar_t>(),                                              //
      (vec3<scalar_t>*) shs.data_ptr<scalar_t>(),                                                  //
      (vec3<scalar_t>*) (has_sh_rest ? sh_rest.value().data_ptr<scalar_t>() : nullptr),            //
      radii.data_ptr<int>(),                                                                       //
      trans_mat.has_value() ? (mat3<scalar_t>*) trans_mat.value().data_ptr<scalar_t>() : nullptr,  //
      colors.has_value() ? (vec3<scalar_t>*) colors.value().data_ptr<scalar_t>() : nullptr,        //
      (vec4<scalar_t>*) normal_opacity.data_ptr<scalar_t>(),                                       //

      (mat3<scalar_t>*) grad_trans.data_ptr<scalar_t>(),                             //
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
__global__ void gs_2d_compute_trans_mat_forward_kernel(int W, int H, int P, const vec3<T>* __restrict__ means3D,
    const T2<T>* __restrict__ scales, const vec4<T>* __restrict__ rotations, const mat4<T> Tw2v, const mat4<T> Tw2c,
    T* __restrict__ trans_mat, vec3<T>* __restrict__ normals) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < P) {
    compute_transmat(
        W, H, means3D[idx], scales[idx], rotations[idx], Tw2c, Tw2v, ((mat3<T>*) trans_mat)[idx], normals[idx]);
  }
}

std::tuple<Tensor, Tensor> gs_2d_compute_trans_mat_forward(
    int W, int H, Tensor means3D, Tensor scales, Tensor rotations, Tensor Tw2v, Tensor Tw2c) {
  int P = means3D.size(0);
  CHECK_INPUT(means3D);
  CHECK_INPUT(scales);
  CHECK_INPUT(rotations);
  CHECK_INPUT(Tw2v);
  CHECK_INPUT(Tw2c);
  CHECK_SHAPE(means3D, P, 3);
  CHECK_SHAPE(scales, P, 2);
  CHECK_SHAPE(rotations, P, 4);
  CHECK_SHAPE(Tw2v, 4, 4);
  CHECK_SHAPE(Tw2c, 4, 4);
  Tw2c = Tw2c.cpu();
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
    auto p_Tw2c    = (mat4<scalar_t>*) Tw2c.data_ptr<scalar_t>();
    if (means3D.is_cuda()) {
      gs_2d_compute_trans_mat_forward_kernel<scalar_t> KERNEL_ARG(div_round_up(P, 256), 256)(
          W, H, P, p_means3D, p_scales, p_rot, *p_Tw2v, *p_Tw2c, p_matrix, p_normal);
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
__global__ void gs_2d_compute_trans_mat_backward_kernel(int W, int H, int P, const vec3<T>* __restrict__ means3D,
    const T2<T>* __restrict__ scales, const vec4<T>* __restrict__ rotations, const mat4<T> Tw2v, const mat4<T> Tw2c,
    const T* __restrict__ grad_trans_mat, vec3<T>* __restrict__ grad_normals, vec3<T>* __restrict__ grad_means3D,
    T2<T>* __restrict__ grad_scales, vec4<T>* __restrict__ grad_rotations) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < P) {
    compute_transmat_backward(W, H, scales[idx], rotations[idx], Tw2c, Tw2v, ((mat3<T>*) grad_trans_mat)[idx],
        grad_normals[idx], grad_means3D[idx], grad_scales[idx], grad_rotations[idx]);
  }
}

vector<Tensor> gs_2d_compute_trans_mat_backward(int W, int H, Tensor means3D, Tensor scales, Tensor rotations,
    Tensor Tw2v, Tensor Tw2c, Tensor grad_matrix, Tensor grad_normals) {
  int P = means3D.size(0);
  CHECK_INPUT(grad_matrix);
  CHECK_INPUT(grad_normals);
  CHECK_SHAPE(grad_matrix, P, 3, 3);
  CHECK_SHAPE(grad_normals, P, 3);
  Tw2c = Tw2c.cpu();
  Tw2v = Tw2v.cpu();

  Tensor grad_means3D   = torch::zeros({P, 3}, means3D.options());
  Tensor grad_scales    = torch::zeros({P, 2}, means3D.options());
  Tensor grad_rotations = torch::zeros({P, 4}, means3D.options());
  AT_DISPATCH_FLOATING_TYPES(means3D.scalar_type(), "gs_2d_compute_trans_mat_backward(CUDA)", [&] {
    auto p_means3D = (vec3<scalar_t>*) means3D.data_ptr<scalar_t>();
    auto p_scales  = (T2<scalar_t>*) scales.data_ptr<scalar_t>();
    auto p_rot     = (vec4<scalar_t>*) rotations.data_ptr<scalar_t>();
    auto p_Tw2v    = (mat4<scalar_t>*) Tw2v.data_ptr<scalar_t>();
    auto p_Tw2c    = (mat4<scalar_t>*) Tw2c.data_ptr<scalar_t>();
    auto g_matrix  = grad_matrix.data_ptr<scalar_t>();
    auto g_normal  = (vec3<scalar_t>*) grad_normals.data_ptr<scalar_t>();
    auto g_means3D = (vec3<scalar_t>*) grad_means3D.data_ptr<scalar_t>();
    auto g_scales  = (T2<scalar_t>*) grad_scales.data_ptr<scalar_t>();
    auto g_rot     = (vec4<scalar_t>*) grad_rotations.data_ptr<scalar_t>();
    if (means3D.is_cuda()) {
      gs_2d_compute_trans_mat_backward_kernel<scalar_t> KERNEL_ARG(div_round_up(P, 256), 256)(
          W, H, P, p_means3D, p_scales, p_rot, *p_Tw2v, *p_Tw2c, g_matrix, g_normal, g_means3D, g_scales, g_rot);
    } else {
      // for (int p = 0; p < P; ++p) {
      //   compute_transmat_backward(W, H, p_scales[p], p_rot[p], *p_Tw2c, *p_Tw2v, ((mat3<scalar_t>*) g_matrix)[p],
      //       g_normal[p], g_means3D[p], g_scales[p], g_rot[p]);
      // }
    }
  });
  return {grad_means3D, grad_scales, grad_rotations};
}
REGIST_PYTORCH_EXTENSION(gs_2d_preprocess, {
  m.def("gs_2d_preprocess_forward", &GS_2D_preprocess_forward, "gs 2d preprocess_forward (CUDA)");
  m.def("gs_2d_preprocess_backward", &GS_2D_preprocess_backward, "gs 2d preprocess_backward (CUDA)");
  m.def("gs_2d_compute_transmat_forward", &gs_2d_compute_trans_mat_forward,
      "gs 2d compute_trans_mat_forward (CUDA, CPU)");
  m.def("gs_2d_compute_trans_mat_backward", &gs_2d_compute_trans_mat_backward,
      "gs 2d compute_trans_mat_backward (CUDA, CPU)");
})

}  // namespace GaussianRasterizer
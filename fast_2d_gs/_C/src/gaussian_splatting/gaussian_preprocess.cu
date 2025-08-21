#if __INTELLISENSE__
#define __CUDACC__
#endif
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda.h>
#include <stdio.h>

#include "gaussian_render.h"
#include "ops_3d.h"
#include "spherical_harmonic.h"
#include "util.cuh"

namespace cg = cooperative_groups;

namespace GaussianRasterizer {
using namespace OPS_3D;

// Forward version of 2D covariance matrix computation
template <typename T = float>
__device__ vec3<T> computeCov2D(
    const vec3<T>& mean, T focal_x, T focal_y, T tan_fovx, T tan_fovy, const T* cov3D, const T* viewmatrix);
// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care of quaternion normalization.
template <typename T>
__device__ void computeCov3D(const vec3<T>& s, const vec4<T>& rot, T* cov3D);

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

// Perform initial steps for each Gaussian prior to rasterization.
template <typename T, int C = NUM_CHANNELS>
__global__ void preprocess_forward_kernel(
    //
    const int P, const int sh_degree, const int M, const bool is_opengl,
    // inputs
    const T* __restrict__ orig_points, const vec3<T>* __restrict__ scales, const vec4<T>* __restrict__ rotations,
    const T* __restrict__ opacities, const vec3<T>* __restrict__ shs, const vec3<T>* __restrict__ sh_rest,
    const T* __restrict__ cov3D_precomp, const T* __restrict__ cov2D_precomp,
    // camera
    const T* __restrict__ Tw2v, const T* __restrict__ Tv2c, const vec3<T>* __restrict__ cam_pos, const int W, int H,
    const T tan_fovx, T tan_fovy, const T focal_x, T focal_y,
    // outputs
    int* __restrict__ radii, T2<T>* __restrict__ points_xy_image, T* __restrict__ depths, T* __restrict__ cov3Ds,
    vec3<T>* __restrict__ rgb, vec4<T>* __restrict__ conic_opacity, const dim3 grid,
    int32_t* __restrict__ tiles_touched, const bool* __restrict__ culling) {
  auto idx = cg::this_grid().thread_rank();
  if (idx >= P) return;

  // Initialize radius and touched tiles to 0. If this isn't changed, this Gaussian will not be processed further.
  radii[idx]         = 0;
  tiles_touched[idx] = 0;
  if (culling && culling[idx]) return;

  // Perform near culling, quit if outside.
  vec3<T> p_orig = {orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2]};
  vec3<T> p_view = xfm_p_4x3(p_orig, Tw2v);
  // if (p_view.z <= T(0.2)) return;  // in_frustum

  // Transform point by projecting
  vec4<T> p_hom  = xfm_p_4x4(p_view, Tv2c);
  T p_w          = T(1.0) / (p_hom.w + T(0.0000001));
  vec3<T> p_proj = {p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w};
  if (is_opengl ? (p_proj.z < T(-1.0) || p_proj.z > T(1.)) : p_view.z <= T(0.2)) return;  // in_frustum

  // If 3D covariance matrix is precomputed, use it, otherwise compute from scaling and rotation parameters.
  vec3<T> cov;
  if (cov2D_precomp != nullptr) {
    cov = {cov2D_precomp[idx * 3 + 0], cov2D_precomp[idx * 3 + 1], cov2D_precomp[idx * 3 + 2]};
  } else {
    const T* cov3D;
    if (cov3D_precomp != nullptr) {
      cov3D = cov3D_precomp + idx * 6;
    } else {
      computeCov3D<T>(scales[idx], rotations[idx], cov3Ds + idx * 6);
      cov3D = cov3Ds + idx * 6;
    }

    // Compute 2D screen-space covariance matrix
    cov = computeCov2D<T>(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, Tw2v);
  }

  // Invert covariance (EWA algorithm)
  T det = (cov.x * cov.z - cov.y * cov.y);
  if (det == T(0.0)) return;

  // Compute extent in screen space (by finding eigenvalues of 2D covariance matrix).
  // Use extent to compute a bounding rectangle of screen-space tiles that this Gaussian overlaps with.
  // Quit if rectangle covers 0 tiles.
  T mid       = T(0.5) * (cov.x + cov.z);
  T lambda1   = mid + sqrt(max(T(0.1), mid * mid - det));
  T lambda2   = mid - sqrt(max(T(0.1), mid * mid - det));
  T my_radius = ceil(T(3.) * sqrt(max(lambda1, lambda2)));

  T2<T> point_image;
  if (is_opengl)
    point_image = {(T(1.0) + p_proj.x) * T(0.5) * W, (T(1.0) + (is_opengl ? -p_proj.y : p_proj.y)) * T(0.5) * H};
  else
    // point_image = {(T(1.0) + p_proj.x) * T(0.5) * W, (T(1.0) + p_proj.y) * T(0.5) * H};
    point_image = {(T(1.0) + p_proj.x) * T(0.5) * W - T(0.5), (T(1.0) + p_proj.y) * T(0.5) * H - T(0.5)};
  int2 rect_min, rect_max;
  getRect(point_image, my_radius, rect_min, rect_max, grid);
  if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0) return;

  // If colors have been precomputed, use them, otherwise convert spherical harmonics coefficients to RGB color.
  if (sh_degree >= 0) {
    rgb[idx] = SH_to_RGB<T>(sh_degree, p_orig, (vec3<T>*) cam_pos, shs + idx * (sh_rest == nullptr ? M : 1),
        sh_rest == nullptr ? nullptr : sh_rest + idx * M, true);
  }

  // Store some useful helper data for the next steps.
  depths[idx]          = is_opengl ? p_proj.z : p_view.z;
  radii[idx]           = my_radius;
  points_xy_image[idx] = point_image;
  // Inverse 2D covariance and opacity neatly pack into one float4
  T det_inv          = T(1.) / det;
  conic_opacity[idx] = {cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv, opacities[idx]};
  tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

// Perform initial steps for each Gaussian prior to rasterization.
template <typename T, int C = NUM_CHANNELS>
__global__ void preprocess_forward_kernel_v2(
    //
    const int P, const int sh_degree, const int M, const bool is_opengl,
    // inputs
    const T* __restrict__ orig_points, const vec3<T>* __restrict__ scales, const vec4<T>* __restrict__ rotations,
    const T* __restrict__ opacities, const vec3<T>* __restrict__ shs, const vec3<T>* __restrict__ sh_rest,
    const T* __restrict__ cov3D_precomp, const T* __restrict__ cov2D_precomp,
    // camera
    const T* __restrict__ Tw2v, const T* __restrict__ Tv2c, const vec3<T>* __restrict__ cam_pos, const int W, int H,
    const T tan_fovx, T tan_fovy, const T focal_x, T focal_y,
    // outputs
    int* __restrict__ radii, T2<T>* __restrict__ rects, T2<T>* __restrict__ points_xy_image, T* __restrict__ depths,
    T* __restrict__ cov3Ds, vec3<T>* __restrict__ rgb, vec4<T>* __restrict__ conic_opacity, const dim3 grid,
    int32_t* __restrict__ tiles_touched, const bool* __restrict__ culling) {
  auto idx = cg::this_grid().thread_rank();
  if (idx >= P) return;

  // Initialize radius and touched tiles to 0. If this isn't changed, this Gaussian will not be processed further.
  radii[idx]         = 0;
  tiles_touched[idx] = 0;
  if (culling && culling[idx]) return;

  // Perform near culling, quit if outside.
  vec3<T> p_orig = {orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2]};
  vec3<T> p_view = xfm_p_4x3(p_orig, Tw2v);
  // if (p_view.z <= T(0.2)) return;  // in_frustum

  // Transform point by projecting
  vec4<T> p_hom  = xfm_p_4x4(p_view, Tv2c);
  T p_w          = T(1.0) / (p_hom.w + T(0.0000001));
  vec3<T> p_proj = {p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w};
  if (is_opengl ? (p_proj.z < T(-1.0) || p_proj.z > T(1.)) : p_view.z <= T(0.2)) return;  // in_frustum

  // If 3D covariance matrix is precomputed, use it, otherwise compute from scaling and rotation parameters.
  vec3<T> cov;
  if (cov2D_precomp != nullptr) {
    cov = {cov2D_precomp[idx * 3 + 0], cov2D_precomp[idx * 3 + 1], cov2D_precomp[idx * 3 + 2]};
  } else {
    const T* cov3D;
    if (cov3D_precomp != nullptr) {
      cov3D = cov3D_precomp + idx * 6;
    } else {
      computeCov3D<T>(scales[idx], rotations[idx], cov3Ds + idx * 6);
      cov3D = cov3Ds + idx * 6;
    }

    // Compute 2D screen-space covariance matrix
    cov = computeCov2D<T>(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, Tw2v);
  }
  // Invert covariance (EWA algorithm)
  T det = (cov.x * cov.z - cov.y * cov.y);
  if (det == T(0.0)) return;
  T det_inv   = 1.f / det;
  T3<T> conic = {cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv};

  T opacity                       = opacities[idx];
  constexpr T alpha_threshold     = 1.0f / 255.0f;
  const T opacity_power_threshold = log(opacity / alpha_threshold);
  const T extent                  = min(3.33, sqrt(2.0f * opacity_power_threshold));

  T mid       = 0.5f * (cov.x + cov.z);
  T lambda    = mid + sqrt(max(0.01f, mid * mid - det));
  T my_radius = extent * sqrt(lambda);
  if (my_radius <= 0.0f) return;

  T2<T> point_image;
  if (is_opengl)
    point_image = {(T(1.0) + p_proj.x) * T(0.5) * W, (T(1.0) + (is_opengl ? -p_proj.y : p_proj.y)) * T(0.5) * H};
  else
    // point_image = {(T(1.0) + p_proj.x) * T(0.5) * W, (T(1.0) + p_proj.y) * T(0.5) * H};
    point_image = {(T(1.0) + p_proj.x) * T(0.5) * W - T(0.5), (T(1.0) + p_proj.y) * T(0.5) * H - T(0.5)};

  int2 rect_min, rect_max;
  // getRect(point_image, my_radius, rect_min, rect_max, grid);
  const T extent_x      = min(extent * sqrt(cov.x), my_radius);
  const T extent_y      = min(extent * sqrt(cov.z), my_radius);
  const T2<T> rect_dims = {extent_x, extent_y};
  getRect(point_image, rect_dims, rect_min, rect_max, grid);

  const int tile_count_rect = (rect_max.x - rect_min.x) * (rect_max.y - rect_min.y);
  if (tile_count_rect == 0) return;

  // If colors have been precomputed, use them, otherwise convert spherical harmonics coefficients to RGB color.
  if (sh_degree >= 0) {
    rgb[idx] = SH_to_RGB<T>(sh_degree, p_orig, (vec3<T>*) cam_pos, shs + idx * (sh_rest == nullptr ? M : 1),
        sh_rest == nullptr ? nullptr : sh_rest + idx * M, true);
  }
  // Store some useful helper data for the next steps.
  depths[idx]          = is_opengl ? p_proj.z : p_view.z;
  radii[idx]           = (int) ceil(my_radius);
  rects[idx]           = rect_dims;
  points_xy_image[idx] = point_image;
  conic_opacity[idx]   = {conic.x, conic.y, conic.z, opacity};
  tiles_touched[idx]   = tile_count_rect;
}

// Backward version of INVERSE 2D covariance matrix computation
// (due to length launched as separate kernel before other backward steps contained in preprocess)
template <typename T>
__global__ void compute_Cov2D_backward_kernel(int P, const vec3<T>* means, const int* radii, const T* cov3Ds,
    const T fx, const T fy, const T tan_fovx, const T tan_fovy, const T* view_matrix, const T* dL_dconics,
    vec3<T>* dL_dmeans, T* dL_dcov, T* dL_dvm) {
  auto idx = cg::this_grid().thread_rank();

  T temp[12] = {0};
  if (idx < P && radii[idx] > 0) {
    dL_dmeans[idx] = computeCov2DBackward<T>(means[idx], cov3Ds + idx * 6, fx, fy, tan_fovx, tan_fovy, view_matrix,
        nullptr, dL_dconics + idx * 4, dL_dcov + idx * 6, dL_dvm == nullptr ? nullptr : temp);
  }
  if (dL_dvm == nullptr) return;
  for (int i = 0; i < 12; ++i) {
    T W = temp[i];
    reduce_sum_block<T, false>(W);
    if (threadIdx.x == 0) atomicAdd(dL_dvm + i, W);
  }
}

// Backward pass of the preprocessing steps, except for the covariance computation and inversion
// (those are handled by a previous kernel call)
template <typename T, int C = NUM_CHANNELS>
__global__ void preprocess_backward_kernel(const int P, const int sh_degree, const int M, const int W, const int H,
    const bool is_opengl,
    // inputs
    const vec3<T>* __restrict__ means, const vec3<T>* __restrict__ scales, const vec4<T>* __restrict__ rotations,
    const vec3<T>* __restrict__ shs, const vec3<T>* __restrict__ sh_rest,
    // camera inputs
    const T* __restrict__ Tw2v, const T* __restrict__ Tv2c, const T* __restrict__ campos, const T fx, const T fy,
    const T tan_fovx, const T tan_fovy,
    // outputs
    const int* __restrict__ radii, const vec3<T>* __restrict__ colors, const T* __restrict__ cov3Ds,
    const T* __restrict__ cov2Ds,
    // grad outputs
    T2<T>* __restrict__ dL_dmean2D, T* __restrict__ dL_dcolor, const T* __restrict__ dL_dconics,
    // grad inputs
    vec3<T>* __restrict__ dL_dmeans3D, vec3<T>* __restrict__ dL_dscale, vec4<T>* __restrict__ dL_drot,
    T* __restrict__ dL_dopacity, vec3<T>* __restrict__ dL_dsh, vec3<T>* __restrict__ dL_dsh_rest,
    T* __restrict__ dL_dcov,
    // grad camera
    T* __restrict__ dL_dTw2v, T* __restrict__ dL_dcampos) {
  auto idx = cg::this_grid().thread_rank();
  // vec3<T> p_orig, dL_dp_view;

  T temp[12] = {0};
  vec3<T> grad_dir;

  if (idx < P && radii[idx] > 0) {
    vec3<T> p_orig = means[idx];
    vec3<T> p_view = xfm_p_4x3(p_orig, Tw2v);
    vec4<T> p_clip = xfm_p_4x4(p_view, Tv2c);
    T m_w          = 1.0f / (p_clip.w + 0.0000001f);

    // Taking care of gradients from the screenspace means3D
    T du = T(0.5 * W) * dL_dmean2D[idx].x, dv = T((is_opengl ? -0.5 : 0.5) * H) * dL_dmean2D[idx].y;
    dL_dmean2D[idx] = {du, dv};

    // Compute loss gradient w.r.t. 3D means due to gradients of 2D means from rendering procedure
    T tmp1             = (p_clip.x * du + p_clip.y * dv) * m_w * m_w;
    vec3<T> dL_dp_view = {
        (Tv2c[0] * du + Tv2c[4] * dv) * m_w - Tv2c[12] * tmp1,
        (Tv2c[1] * du + Tv2c[5] * dv) * m_w - Tv2c[13] * tmp1,
        (Tv2c[2] * du + Tv2c[6] * dv) * m_w - Tv2c[14] * tmp1,
    };

    vec3<T> dL_dmean = {
        Tw2v[0] * dL_dp_view.x + Tw2v[4] * dL_dp_view.y + Tw2v[8] * dL_dp_view.z,
        Tw2v[1] * dL_dp_view.x + Tw2v[5] * dL_dp_view.y + Tw2v[9] * dL_dp_view.z,
        Tw2v[2] * dL_dp_view.x + Tw2v[6] * dL_dp_view.y + Tw2v[10] * dL_dp_view.z,
    };

    if (dL_dopacity != nullptr) dL_dopacity[idx] = dL_dconics[idx * 4 + 3];
    // Compute loss gradient w.r.t. cov2D
    if (cov2Ds != nullptr && dL_dcov != nullptr) {  // cov2D is pre-computed
      T a = cov2Ds[idx * 3 + 0], b = cov2Ds[idx * 3 + 1], c = cov2Ds[idx * 3 + 2];
      T denom              = a * c - b * b;
      T denom2inv          = 1. / ((denom * denom) + 0.0000001f);
      T t                  = c * dL_dconics[idx * 4 + 0] - b * dL_dconics[idx * 4 + 1] + a * dL_dconics[idx * 4 + 2];
      dL_dcov[idx * 3 + 0] = denom2inv * (-c * t + denom * dL_dconics[idx * 4 + 2]);
      dL_dcov[idx * 3 + 2] = denom2inv * (-a * t + denom * dL_dconics[idx * 4 + 0]);
      dL_dcov[idx * 3 + 1] = denom2inv * (T(2) * b * t - denom * dL_dconics[idx * 4 + 1]);
    }
    if (cov2Ds == nullptr) {
      dL_dmean += computeCov2DBackward<T>(p_orig, cov3Ds + idx * 6, fx, fy, tan_fovx, tan_fovy, Tw2v, nullptr,
          dL_dconics + idx * 4, dL_dcov == nullptr ? nullptr : dL_dcov + idx * 6, dL_dTw2v == nullptr ? nullptr : temp);
    }

    if (dL_dTw2v != nullptr) {
      temp[0] += p_orig.x * dL_dp_view.x;
      temp[1] += p_orig.y * dL_dp_view.x;
      temp[2] += p_orig.z * dL_dp_view.x;
      temp[3] += dL_dp_view.x;
      temp[4] += p_orig.x * dL_dp_view.y;
      temp[5] += p_orig.y * dL_dp_view.y;
      temp[6] += p_orig.z * dL_dp_view.y;
      temp[7] += dL_dp_view.y;
      temp[8] += p_orig.x * dL_dp_view.z;
      temp[9] += p_orig.y * dL_dp_view.z;
      temp[10] += p_orig.z * dL_dp_view.z;
      temp[11] += dL_dp_view.z;
    }

    // Compute gradient updates due to computing colors from SHs
    if (sh_degree >= 0 && dL_dcolor != nullptr) {
      int M1   = sh_rest == nullptr ? M : 1;
      grad_dir = SH_to_RGB_backward<T>(sh_degree, means[idx], (vec3<T>*) campos, colors[idx], shs + idx * M1,
          sh_rest == nullptr ? nullptr : sh_rest + idx * M, ((vec3<T>*) dL_dcolor)[idx],
          dL_dsh == nullptr ? nullptr : dL_dsh + idx * M1, dL_dsh_rest == nullptr ? nullptr : dL_dsh_rest + idx * M,
          true, true);

      dL_dmean += grad_dir;
    }
    dL_dmeans3D[idx] = dL_dmean;

    // Compute gradient updates due to computing covariance from scale/rotation
    if (dL_dscale != nullptr || dL_drot != nullptr) {
      computeCov3D_backward<T>((const T*) (scales + idx), (const T*) (rotations + idx), dL_dcov + idx * 6,
          dL_dscale == nullptr ? nullptr : (T*) (dL_dscale + idx), dL_drot == nullptr ? nullptr : (T*) (dL_drot + idx));
    }
  }
  if (dL_dTw2v != nullptr) {
    for (int i = 0; i < 12; ++i) {
      T W = temp[i];
      reduce_sum_block<T, false>(W);
      if (threadIdx.x == 0) atomicAdd(dL_dTw2v + i, W);
    }
  }
  if (dL_dcampos != nullptr) {
    reduce_sum_block<T, false>(grad_dir.x);
    reduce_sum_block<T, false>(grad_dir.y);
    reduce_sum_block<T, false>(grad_dir.z);
    if (threadIdx.x == 0) {
      atomicAdd(dL_dcampos + 0, -grad_dir.x);
      atomicAdd(dL_dcampos + 1, -grad_dir.y);
      atomicAdd(dL_dcampos + 2, -grad_dir.z);
    }
  }
}

vector<Tensor> GS_preprocess_forward(int W, int H, int sh_degree, bool is_opengl,
    // Gaussians
    Tensor means3D, Tensor scales, Tensor rotations, Tensor opacities, Tensor shs, torch::optional<Tensor> sh_rest,
    // cameras
    Tensor Tw2v, Tensor Tv2c, Tensor cam_pos, Tensor tanFoV,
    // pre-computed tensors
    torch::optional<Tensor> cov3D_precomp, torch::optional<Tensor> cov2D_precomp, torch::optional<Tensor> means2D,
    torch::optional<Tensor> culling) {
  CHECK_INPUT(means3D);
  BCNN_ASSERT(means3D.ndimension() == 2 && means3D.size(-1) == 3, "Error shape for means3D");
  int P = means3D.size(0);
  int M = 0;
  CHECK_INPUT(scales);
  CHECK_INPUT(rotations);
  CHECK_INPUT(opacities);
  CHECK_INPUT(shs);
  bool has_sh_rest = false;
  // NOTE: when sh_rest.value().numel() == 0, sh_rest.value().data_ptr<scalar_t>() = nullptr
  if (sh_rest.has_value() && sh_rest.value().numel() > 0) {
    BCNN_ASSERT(sh_rest.has_value(), "shs and colors must be None at same time")
    CHECK_INPUT(sh_rest.value());
    BCNN_ASSERT(sh_rest.value().ndimension() == 3 && sh_rest.value().size(0) == P &&
                    sh_rest.value().size(1) >= (1 + sh_degree) * (1 + sh_degree) - 1 && sh_rest.value().size(2) == 3,
        "Error shape for sh features");
    has_sh_rest = true;
    M           = sh_rest.value().size(1);
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

  CHECK_SHAPE(scales, P, 3);
  CHECK_SHAPE(rotations, P, 4);
  CHECK_SHAPE(opacities, P, 1);
  if (sh_degree < 0) CHECK_SHAPE(shs, P, 3);
  CHECK_SHAPE(Tw2v, 4, 4);
  CHECK_SHAPE(Tv2c, 4, 4);
  CHECK_SHAPE(cam_pos, 3);
  CHECK_SHAPE(tanFoV, 2);

  Tensor colors        = sh_degree >= 0 ? torch::zeros_like(means3D) : shs;
  Tensor cov3Ds        = cov3D_precomp.has_value() ? cov3D_precomp.value() : torch::zeros({P, 6}, means3D.options());
  Tensor means2D_      = means2D.has_value() ? means2D.value() : torch::zeros({P, 2}, means3D.options());
  Tensor depths        = torch::zeros({P}, means3D.options());
  Tensor radii         = torch::zeros({P}, means3D.options().dtype(torch::kInt32));
  Tensor tiles_touched = torch::zeros({P}, means3D.options().dtype(torch::kInt32));
  Tensor conic_opacity = torch::zeros({P, 4}, means3D.options());

  dim3 tile_grid((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, 1);
  // dim3 block(BLOCK_X, BLOCK_Y, 1);
  // AT_DISPATCH_FLOATING_TYPES(means3D.scalar_type(), "GS_preprocess_forward (CUDA)", [&] {
  using scalar_t          = float;
  const scalar_t tan_fovx = tanFoV[0].item<scalar_t>();
  const scalar_t tan_fovy = tanFoV[1].item<scalar_t>();
  const scalar_t focal_x  = 0.5 * W / tan_fovx;
  const scalar_t focal_y  = 0.5 * H / tan_fovy;
  preprocess_forward_kernel<scalar_t, NUM_CHANNELS> KERNEL_ARG((P + 255) / 256, 256)(  //
      P, sh_degree, M, is_opengl,
      means3D.data_ptr<scalar_t>(),                                                            //
      (const vec3<scalar_t>*) scales.data_ptr<scalar_t>(),                                     //
      (const vec4<scalar_t>*) rotations.data_ptr<scalar_t>(),                                  //
      opacities.data_ptr<scalar_t>(),                                                          //
      (const vec3<scalar_t>*) shs.data_ptr<scalar_t>(),                                        //
      (const vec3<scalar_t>*) (has_sh_rest ? sh_rest.value().data_ptr<scalar_t>() : nullptr),  //
      cov3D_precomp.has_value() ? cov3D_precomp.value().data_ptr<scalar_t>() : nullptr,        //
      cov2D_precomp.has_value() ? cov2D_precomp.value().data_ptr<scalar_t>() : nullptr,        //
      Tw2v.data_ptr<scalar_t>(),                                                               //
      Tv2c.data_ptr<scalar_t>(),                                                               //
      (const vec3<scalar_t>*) cam_pos.data_ptr<scalar_t>(),                                    //
      W, H, tan_fovx, tan_fovy, focal_x, focal_y,                                              //
      radii.data_ptr<int>(),                                                                   //
      (T2<scalar_t>*) means2D_.data_ptr<scalar_t>(),                                           //
      depths.data_ptr<scalar_t>(),                                                             //
      cov3Ds.data_ptr<scalar_t>(),                                                             //
      (vec3<scalar_t>*) colors.data_ptr<scalar_t>(),                                           //
      (vec4<scalar_t>*) conic_opacity.data_ptr<scalar_t>(),                                    //
      tile_grid,                                                                               //
      tiles_touched.data_ptr<int32_t>(),                                                       //
      culling.has_value() ? culling.value().data_ptr<bool>() : nullptr                         //
  );
  // });
  return {means2D_, depths, colors, radii, tiles_touched, cov3Ds, conic_opacity};
}

vector<Tensor> GS_preprocess_forward_v2(int W, int H, int sh_degree, bool is_opengl,
    // Gaussians
    Tensor means3D, Tensor scales, Tensor rotations, Tensor opacities, Tensor shs, torch::optional<Tensor> sh_rest,
    // cameras
    Tensor Tw2v, Tensor Tv2c, Tensor cam_pos, Tensor tanFoV,
    // pre-computed tensors
    torch::optional<Tensor> cov3D_precomp, torch::optional<Tensor> cov2D_precomp, torch::optional<Tensor> means2D,
    torch::optional<Tensor> culling) {
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
    BCNN_ASSERT(sh_rest.has_value(), "shs and colors must be None at same time")
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

  CHECK_SHAPE(scales, P, 3);
  CHECK_SHAPE(rotations, P, 4);
  CHECK_SHAPE(opacities, P, 1);
  if (sh_degree < 0) CHECK_SHAPE(shs, P, 3);
  CHECK_SHAPE(Tw2v, 4, 4);
  CHECK_SHAPE(Tv2c, 4, 4);
  CHECK_SHAPE(cam_pos, 3);
  CHECK_SHAPE(tanFoV, 2);

  Tensor colors        = sh_degree >= 0 ? torch::zeros_like(means3D) : shs;
  Tensor cov3Ds        = cov3D_precomp.has_value() ? cov3D_precomp.value() : torch::zeros({P, 6}, means3D.options());
  Tensor means2D_      = means2D.has_value() ? means2D.value() : torch::zeros({P, 2}, means3D.options());
  Tensor depths        = torch::zeros({P}, means3D.options());
  Tensor radii         = torch::zeros({P}, means3D.options().dtype(torch::kInt32));
  Tensor tiles_touched = torch::zeros({P}, means3D.options().dtype(torch::kInt32));
  Tensor conic_opacity = torch::zeros({P, 4}, means3D.options());
  Tensor rects         = torch::zeros({P, 2}, means3D.options());

  dim3 tile_grid((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, 1);
  // dim3 block(BLOCK_X, BLOCK_Y, 1);
  // AT_DISPATCH_FLOATING_TYPES(means3D.scalar_type(), "GS_preprocess_forward (CUDA)", [&] {
  using scalar_t          = float;
  const scalar_t tan_fovx = tanFoV[0].item<scalar_t>();
  const scalar_t tan_fovy = tanFoV[1].item<scalar_t>();
  const scalar_t focal_x  = 0.5 * W / tan_fovx;
  const scalar_t focal_y  = 0.5 * H / tan_fovy;
  preprocess_forward_kernel_v2<scalar_t, NUM_CHANNELS> KERNEL_ARG((P + 255) / 256, 256)(  //
      P, sh_degree, M, is_opengl,
      means3D.data_ptr<scalar_t>(),                                                            //
      (const vec3<scalar_t>*) scales.data_ptr<scalar_t>(),                                     //
      (const vec4<scalar_t>*) rotations.data_ptr<scalar_t>(),                                  //
      opacities.data_ptr<scalar_t>(),                                                          //
      (const vec3<scalar_t>*) shs.data_ptr<scalar_t>(),                                        //
      (const vec3<scalar_t>*) (has_sh_rest ? sh_rest.value().data_ptr<scalar_t>() : nullptr),  //
      cov3D_precomp.has_value() ? cov3D_precomp.value().data_ptr<scalar_t>() : nullptr,        //
      cov2D_precomp.has_value() ? cov2D_precomp.value().data_ptr<scalar_t>() : nullptr,        //
      Tw2v.data_ptr<scalar_t>(),                                                               //
      Tv2c.data_ptr<scalar_t>(),                                                               //
      (const vec3<scalar_t>*) cam_pos.data_ptr<scalar_t>(),                                    //
      W, H, tan_fovx, tan_fovy, focal_x, focal_y,                                              //
      radii.data_ptr<int>(),                                                                   //
      (T2<scalar_t>*) rects.data_ptr<scalar_t>(),                                              //
      (T2<scalar_t>*) means2D_.data_ptr<scalar_t>(),                                           //
      depths.data_ptr<scalar_t>(),                                                             //
      cov3Ds.data_ptr<scalar_t>(),                                                             //
      (vec3<scalar_t>*) colors.data_ptr<scalar_t>(),                                           //
      (vec4<scalar_t>*) conic_opacity.data_ptr<scalar_t>(),                                    //
      tile_grid,                                                                               //
      tiles_touched.data_ptr<int32_t>(),                                                       //
      culling.has_value() ? culling.value().data_ptr<bool>() : nullptr                         //
  );
  // });
  return {means2D_, depths, colors, radii, rects, tiles_touched, cov3Ds, conic_opacity};
}

void GS_preprocess_backward(
    // information
    int W, int H, int sh_degree, bool is_opengl,
    // geometry tensors
    Tensor means3D, Tensor scales, Tensor rotations, Tensor opacities, Tensor shs, torch::optional<Tensor> sh_rest,
    // camera informations
    Tensor Tw2v, Tensor Tv2c, Tensor cam_pos, Tensor tanFoV,
    // internal tensors
    torch::optional<Tensor> cov3Ds, torch::optional<Tensor> cov2Ds, Tensor radii, Tensor colors,
    // grad_outputs
    Tensor grad_mean2D, torch::optional<Tensor> grad_depth, torch::optional<Tensor> grad_colors, Tensor grad_conic,
    // grad_inputs
    Tensor& grad_means3D, torch::optional<Tensor>& grad_scales, torch::optional<Tensor>& grad_rotations,
    Tensor& grad_opacities, torch::optional<Tensor>& grad_shs, torch::optional<Tensor>& grad_sh_rest,
    torch::optional<Tensor> grad_cov,
    // grad cameras
    torch::optional<Tensor> grad_Tw2v, torch::optional<Tensor> grad_campos) {
  CHECK_INPUT(grad_mean2D);
  if (grad_colors.has_value()) CHECK_INPUT(grad_colors.value());
  CHECK_INPUT(grad_conic);
  CHECK_INPUT(grad_means3D);
  if (grad_scales.has_value()) CHECK_INPUT(grad_scales.value());
  if (grad_rotations.has_value()) CHECK_INPUT(grad_rotations.value());
  CHECK_INPUT(grad_opacities);
  if (grad_shs.has_value()) {
    CHECK_INPUT(grad_shs.value());
  }
  if (grad_sh_rest.has_value()) {
    CHECK_INPUT(grad_sh_rest.value());
  }
  bool has_sh_rest = sh_rest.has_value() && sh_rest.value().numel() > 0;
  int P            = means3D.size(0);
  int D            = sh_degree;
  int M            = has_sh_rest ? sh_rest.value().size(1) : shs.size(1);

  using scalar_t = float;
  // AT_DISPATCH_FLOATING_TYPES(means3D.scalar_type(), "preprocessCUDA_backward(CUDA)", [&] {
  const scalar_t tan_fovx = tanFoV[0].item<scalar_t>();
  const scalar_t tan_fovy = tanFoV[1].item<scalar_t>();
  const scalar_t focal_x  = 0.5 * W / tan_fovx;
  const scalar_t focal_y  = 0.5 * H / tan_fovy;
  preprocess_backward_kernel<scalar_t, NUM_CHANNELS> KERNEL_ARG((P + 255) / 256, 256)(                       //
      P, D, M, W, H, is_opengl,                                                                              //
      (vec3<scalar_t>*) means3D.data_ptr<scalar_t>(),                                                        //
      (vec3<scalar_t>*) scales.data_ptr<scalar_t>(),                                                         //
      (vec4<scalar_t>*) rotations.data_ptr<scalar_t>(),                                                      //
      (vec3<scalar_t>*) shs.data_ptr<scalar_t>(),                                                            //
      (vec3<scalar_t>*) (has_sh_rest ? sh_rest.value().data_ptr<scalar_t>() : nullptr),                      //
      Tw2v.data_ptr<scalar_t>(), Tv2c.data_ptr<scalar_t>(), (scalar_t*) cam_pos.data_ptr<scalar_t>(),        //
      (scalar_t) focal_x, (scalar_t) focal_y, (scalar_t) tan_fovx, (scalar_t) tan_fovy,                      //
      radii.data_ptr<int>(),                                                                                 //
      (vec3<scalar_t>*) colors.data_ptr<scalar_t>(),                                                         //
      cov3Ds.has_value() ? cov3Ds.value().data_ptr<scalar_t>() : nullptr,                                    //
      cov2Ds.has_value() ? cov2Ds.value().data_ptr<scalar_t>() : nullptr,                                    //
      (T2<scalar_t>*) grad_mean2D.data_ptr<scalar_t>(),                                                      //
      grad_colors.has_value() ? grad_colors.value().data_ptr<scalar_t>() : nullptr,                          //
      grad_conic.data_ptr<scalar_t>(),                                                                       //
      (vec3<scalar_t>*) grad_means3D.data_ptr<scalar_t>(),                                                   //
      grad_scales.has_value() ? (vec3<scalar_t>*) grad_scales.value().data_ptr<scalar_t>() : nullptr,        //
      grad_rotations.has_value() ? (vec4<scalar_t>*) grad_rotations.value().data_ptr<scalar_t>() : nullptr,  //
      grad_opacities.data_ptr<scalar_t>(),                                                                   //
      (vec3<scalar_t>*) (grad_shs.has_value() ? grad_shs.value().data_ptr<scalar_t>() : nullptr),            //
      (vec3<scalar_t>*) (grad_sh_rest.has_value() ? grad_sh_rest.value().data_ptr<scalar_t>() : nullptr),    //
      grad_cov.has_value() ? grad_cov.value().data_ptr<scalar_t>() : nullptr,                                //
      grad_Tw2v.has_value() ? grad_Tw2v.value().data_ptr<scalar_t>() : nullptr,                              //
      grad_campos.has_value() ? grad_campos.value().data_ptr<scalar_t>() : nullptr);
  CHECK_CUDA_ERROR("preprocessCUDA_backward");
  // });
}

REGIST_PYTORCH_EXTENSION(gs_gaussian_preprocess, {
  m.def("gs_preprocess_forward", &GS_preprocess_forward, "gs preprocess_forward (CUDA)");
  m.def("gs_preprocess_forward_v2", &GS_preprocess_forward_v2, "gs preprocess_forward (CUDA)");
  m.def("gs_preprocess_backward", &GS_preprocess_backward, "preprocess_backward (CUDA)");
})

#define INSTANCE_FUN(T, C)                                                                                             \
  template __global__ void preprocess_forward_kernel<T, C>(const int P, const int sh_degree, const int M,              \
      const bool is_opengl, const T* orig_points, const vec3<T>* scales, const vec4<T>* rotations, const T* opacities, \
      const vec3<T>* shs, const vec3<T>* sh_rest, const T* cov3D_precomp, const T* cov2D_precomp, const T* Tw2v,       \
      const T* Tv2c, const vec3<T>* cam_pos, const int W, int H, const T tan_fovx, T tan_fovy, const T focal_x,        \
      T focal_y, int* radii, T2<T>* points_xy_image, T* depths, T* cov3Ds, vec3<T>* rgb, vec4<T>* conic_opacity,       \
      const dim3 grid, int32_t* tiles_touched, const bool* culling);                                                   \
  template __global__ void preprocess_backward_kernel<T, C>(const int P, const int sh_degree, const int M,             \
      const int W, const int H, const bool is_opengl, const vec3<T>* means, const vec3<T>* scales,                     \
      const vec4<T>* rotations, const vec3<T>* shs, const vec3<T>* sh_rest, const T* Tw2v, const T* Tv2c,              \
      const T* campos, const T fx, const T fy, const T tan_fovx, const T tan_fovy, const int* radii,                   \
      const vec3<T>* colors, const T* cov3Ds, const T* cov2Ds, T2<T>* dL_dmean2D, T* dL_dcolor, const T* dL_dconics,   \
      vec3<T>* dL_dmeans3D, vec3<T>* dL_dscale, vec4<T>* dL_drot, T* dL_dopacity, vec3<T>* dL_dsh,                     \
      vec3<T>* dL_dsh_rest, T* dL_dcov, T* dL_dTw2v, T* dL_dcampos);

INSTANCE_FUN(float, NUM_CHANNELS);
INSTANCE_FUN(double, NUM_CHANNELS);

}  // namespace GaussianRasterizer
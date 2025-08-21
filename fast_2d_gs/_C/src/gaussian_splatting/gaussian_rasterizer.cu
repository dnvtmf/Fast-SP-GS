/*
paper: 3D Gaussian Splatting for Real-Time Radiance Field Rendering, SIGGRAPH 2023
code:  https://github.com/graphdeco-inria/diff-gaussian-rasterization
*/
/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>

#include "gaussian_render.h"
#include "ops_3d.h"
#include "util.cuh"

namespace cg = cooperative_groups;
using namespace OPS_3D;

namespace GaussianRasterizer {

template <typename T, int C = NUM_CHANNELS>
__global__ void preprocess_forward_kernel(  //
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
    int32_t* __restrict__ tiles_touched, const bool* __restrict__ culling);

template <typename T>
void render_forward(const dim3 grid, dim3 block, const T2<int64_t>* ranges, const int32_t* point_list, int W, int H,
    int E, const T2<T>* means2D, const T* colors, const T4<T>* conic_opacity, const T* extra, int32_t* n_contrib,
    T* out_color, T* out_opacity, T* out_extra, int32_t* accum_max_count, T* accum_weights_p,
    int32_t* accum_weights_count, const int64_t* per_tile_bucket_offset, int32_t* bucket_to_tile, T* sampled_T,
    T* sampled_ar, int32_t* max_contrib);

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
    T* __restrict__ dL_dTw2v, T* __restrict__ dL_dcampos);

template <typename T>
void render_backward(int P, int W, int H, int E, int64_t R, const dim3 grid, const dim3 block,
    // inputs
    const T2<T>* means2D, const T4<T>* conic_opacity, const T* colors, const T* extras,
    // aux inputs
    const T2<int64_t>* ranges, const int32_t* point_list, const int32_t* n_contrib,
    // outputs & grad outputs
    const T* out_opacity, const T* dL_dpixels, const T* dL_dout_extras, const T* dL_dout_opacity,
    // grad inputs
    T2<T>* dL_dmean2D, T4<T>* dL_dconic_opacity, T* dL_dcolors, T* dL_dextras);

template <typename T, typename TK>
int64_t prepare(int P, int width, int height, bool debug, T2<T>* means2D, T* depths, int* radii, int32_t* tiles_touched,
    int64_t* point_offsets, Tensor& scanning_space, T2<int64_t>* ranges, Tensor& point_list_unsorted,
    Tensor& point_list_keys_unsorted, Tensor& point_list, Tensor& point_list_keys, Tensor& list_sorting_space);

std::tuple<int, Tensor, Tensor, torch::optional<Tensor>, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>
    RasterizeGaussiansCUDA(
        // const parameters
        const int image_height, const int image_width, const int sh_degree, bool debug, bool is_opengl,
        // GS parameters
        const Tensor& means3D, const Tensor& opacities, const Tensor& scales, const Tensor& rotations,
        const Tensor& shs, const torch::optional<Tensor>& sh_rest, const at::optional<Tensor>& extras,
        // camera parameters
        const Tensor& Tw2v, const Tensor& Tv2c, const Tensor& cam_pos, const Tensor& tanFoV,
        // pre-computed tensors
        torch::optional<Tensor> cov3D_precomp, torch::optional<Tensor> cov2D_precomp, Tensor& means2D) {
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }
  CHECK_INPUT(means3D);
  BCNN_ASSERT(means3D.ndimension() == 2 && means3D.size(-1) == 3, "Error shape for means3D");
  const int P = means3D.size(0);
  int M       = 0;
  CHECK_INPUT(scales);
  CHECK_INPUT(rotations);
  CHECK_INPUT(opacities);
  CHECK_INPUT(shs);
  if (sh_degree >= 0) {
    if (sh_rest.has_value()) {
      BCNN_ASSERT(sh_rest.has_value(), "shs and colors must be None at same time")
      CHECK_INPUT(sh_rest.value());
      BCNN_ASSERT(sh_rest.value().ndimension() == 3 && sh_rest.value().size(0) == P &&
                      sh_rest.value().size(1) >= (1 + sh_degree) * (1 + sh_degree) - 1 && sh_rest.value().size(2) == 3,
          "Error shape for sh_rest features");
      CHECK_SHAPE(shs, P, 1, 3);
      M = sh_rest.value().size(1);
    } else {
      BCNN_ASSERT(shs.ndimension() == 3 && shs.size(0) == P && shs.size(1) >= (1 + sh_degree) * (1 + sh_degree) &&
                      shs.size(2) == 3,
          "Error shape for sh features");
      M = shs.size(1);
    }
  }
  CHECK_INPUT(means2D);

  CHECK_INPUT(Tw2v);
  CHECK_INPUT(Tv2c);
  CHECK_INPUT(cam_pos);

  CHECK_SHAPE(scales, P, 3);
  CHECK_SHAPE(rotations, P, 4);
  CHECK_SHAPE(opacities, P, 1);
  CHECK_SHAPE(Tw2v, 4, 4);
  CHECK_SHAPE(Tv2c, 4, 4);
  CHECK_SHAPE(cam_pos, 3);
  CHECK_SHAPE(means2D, P, 2);
  if (cov3D_precomp.has_value()) {
    CHECK_INPUT(cov3D_precomp.value());
    CHECK_SHAPE(cov3D_precomp.value(), P, 6);
  }
  if (sh_degree < 0) CHECK_SHAPE(shs, P, 3);
  CHECK_SHAPE(tanFoV, 2);

  const int H = image_height;
  const int W = image_width;
  const int E = extras.has_value() ? extras.value().size(-1) : 0;

  dim3 tile_grid((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, 1);
  dim3 block(BLOCK_X, BLOCK_Y, 1);

  //   Tensor means2D        = torch::zeros({P, 2}, means3D.options());
  Tensor colors         = sh_degree < 0 ? shs : torch::zeros_like(means3D);
  Tensor cov3Ds         = cov3D_precomp.has_value() ? cov3D_precomp.value() : torch::zeros({P, 6}, means3D.options());
  Tensor depths         = torch::zeros({P}, means3D.options());
  Tensor radii          = torch::zeros({P}, means3D.options().dtype(torch::kInt32));
  Tensor tiles_touched  = torch::zeros({P}, means3D.options().dtype(torch::kInt32));
  Tensor conic_opacity  = torch::zeros({P, 4}, means3D.options());
  Tensor out_color      = torch::zeros({H, W, NUM_CHANNELS}, means3D.options());
  Tensor out_opacity    = torch::zeros({H, W}, means3D.options());
  Tensor n_contrib      = torch::zeros({H, W}, means3D.options().dtype(torch::kInt32));
  Tensor ranges         = torch::zeros({tile_grid.x * tile_grid.y, 2}, means3D.options().dtype(torch::kInt64));
  Tensor point_offsets  = torch::zeros(P, means3D.options().dtype(torch::kInt64));
  Tensor sorted_key     = torch::zeros(0, means3D.options().dtype(torch::kInt64));
  Tensor sorted_value   = torch::zeros(0, means3D.options().dtype(torch::kInt32));
  Tensor unsorted_key   = torch::zeros(0, means3D.options().dtype(torch::kInt64));
  Tensor unsorted_value = torch::zeros(0, means3D.options().dtype(torch::kInt32));
  Tensor temp_scan      = torch::zeros(0, means3D.options().dtype(torch::kInt8));
  Tensor temp_sort      = torch::zeros(0, means3D.options().dtype(torch::kInt8));
  torch::optional<Tensor> out_extras;
  if (extras.has_value()) out_extras = torch::zeros({H, W, E}, means3D.options());
  int64_t num_rendered;

  AT_DISPATCH_FLOATING_TYPES(means3D.scalar_type(), "GS_preprocess_forward (CUDA)", [&] {
    //   using scalar_t = float;
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
        (vec3<scalar_t>*) shs.data_ptr<scalar_t>(),                                              //
        sh_rest.has_value() ? (vec3<scalar_t>*) sh_rest.value().data_ptr<scalar_t>() : nullptr,  //
        cov3D_precomp.has_value() ? cov3D_precomp.value().data_ptr<scalar_t>() : nullptr,        //
        cov2D_precomp.has_value() ? cov2D_precomp.value().data_ptr<scalar_t>() : nullptr,        //
        Tw2v.data_ptr<scalar_t>(),                                                               //
        Tv2c.data_ptr<scalar_t>(),                                                               //
        (const vec3<scalar_t>*) cam_pos.data_ptr<scalar_t>(),                                    //
        W, H, (scalar_t) tan_fovx, tan_fovy, focal_x, focal_y,                                   //
        radii.data_ptr<int>(),                                                                   //
        (T2<scalar_t>*) means2D.data_ptr<scalar_t>(),                                            //
        depths.data_ptr<scalar_t>(),                                                             //
        cov3Ds.data_ptr<scalar_t>(),                                                             //
        (vec3<scalar_t>*) colors.data_ptr<scalar_t>(),                                           //
        (vec4<scalar_t>*) conic_opacity.data_ptr<scalar_t>(),                                    //
        tile_grid,                                                                               //
        tiles_touched.data_ptr<int32_t>(), nullptr                                               //
    );
    if (debug) cudaDeviceSynchronize();
    CHECK_CUDA_ERROR("preprocess_forward_kernel");

    num_rendered = prepare<scalar_t, int64_t>(P, W, H, debug, (T2<scalar_t>*) means2D.data_ptr<scalar_t>(),
        depths.data_ptr<scalar_t>(), radii.data_ptr<int>(), tiles_touched.data_ptr<int32_t>(),
        point_offsets.data_ptr<int64_t>(), temp_scan, (T2<int64_t>*) ranges.data_ptr<int64_t>(), unsorted_value,
        unsorted_key, sorted_value, sorted_key, temp_sort);

    render_forward<scalar_t>(tile_grid, block, (T2<int64_t>*) ranges.contiguous().data_ptr<int64_t>(),
        sorted_value.contiguous().data_ptr<int32_t>(), W, H, E,
        (T2<scalar_t>*) means2D.contiguous().data_ptr<scalar_t>(), colors.contiguous().data_ptr<scalar_t>(),
        (T4<scalar_t>*) conic_opacity.contiguous().data_ptr<scalar_t>(),
        extras.has_value() ? extras.value().contiguous().data_ptr<scalar_t>() : nullptr, n_contrib.data_ptr<int32_t>(),
        out_color.data_ptr<scalar_t>(), out_opacity.data_ptr<scalar_t>(),
        extras.has_value() ? out_extras.value().data_ptr<scalar_t>() : nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr);
    if (debug) cudaDeviceSynchronize();
    CHECK_CUDA_ERROR("render_forward");
  });
  return std::make_tuple(num_rendered, out_color, out_opacity, out_extras, cov3Ds, colors, conic_opacity, radii, ranges,
      sorted_value, n_contrib);
}

void RasterizeGaussiansBackwardCUDA(
    // scalar parameters
    const int sh_degree, const bool debug, bool is_opengl,
    // tensor parameters
    const Tensor& Tw2v, const Tensor& Tv2c, const Tensor& campos, const Tensor& tanFoV,
    // inputs
    const Tensor& means3D, const Tensor& scales, const Tensor& rotations, const Tensor& shs,
    const torch::optional<Tensor>& sh_rest, const at::optional<Tensor> extras, const Tensor& colors,
    const torch::optional<Tensor>& cov3Ds, const torch::optional<Tensor>& cov2Ds,
    // internal auxiliary tensors
    const Tensor& ranges, const Tensor& point_list, const Tensor& n_contrib,
    // outputs
    const Tensor& means2D, const Tensor& conic_opacity, const Tensor& radii, const Tensor& out_opacity,
    // grad_outputs
    const Tensor& dL_dout_color, const Tensor& dL_dout_opacity, const at::optional<Tensor>& dL_dout_extra,
    // grad internal tensors
    torch::optional<Tensor>& grad_means2D, torch::optional<Tensor>& grad_conic_o,
    // grad_inputs
    Tensor& grad_means3D, torch::optional<Tensor>& grad_scales, torch::optional<Tensor>& grad_rotations,
    torch::optional<Tensor>& grad_opacity, torch::optional<Tensor>& grad_shs, torch::optional<Tensor>& grad_sh_rest,
    torch::optional<Tensor> grad_extras, torch::optional<Tensor>& grad_Tw2v, torch::optional<Tensor>& grad_campos,
    torch::optional<Tensor>& grad_colors, torch::optional<Tensor>& grad_cov) {
  const int P = means3D.size(0);
  const int H = dL_dout_color.size(0);
  const int W = dL_dout_color.size(1);
  const int E = (extras.has_value() && dL_dout_extra.has_value()) ? extras.value().size(-1) : 0;
  const int M = sh_rest.has_value() ? sh_rest.value().size(1) : shs.size(1);
  int64_t R   = point_list.size(0);

  CHECK_SHAPE(Tw2v, 4, 4);
  CHECK_SHAPE(Tv2c, 4, 4);
  CHECK_SHAPE(campos, 3);
  CHECK_SHAPE(means3D, P, 3);
  CHECK_SHAPE(scales, P, 3);
  CHECK_SHAPE(rotations, P, 4);
  if (sh_degree >= 0) {
    CHECK_SHAPE(shs, P, sh_rest.has_value() ? 1 : M, 3);
    if (sh_rest.has_value()) CHECK_SHAPE(sh_rest.value(), P, M, 3);
  }
  if (extras.has_value()) CHECK_SHAPE(extras.value(), P, E);
  CHECK_SHAPE(colors, P, 3);
  if (cov3Ds.has_value()) CHECK_SHAPE(cov3Ds.value(), P, 6);
  if (cov2Ds.has_value()) CHECK_SHAPE(cov2Ds.value(), P, 3);
  CHECK_INPUT(point_list);
  CHECK_SHAPE(point_list, R);
  CHECK_INPUT(n_contrib);
  CHECK_SHAPE(n_contrib, H, W);
  CHECK_INPUT(means2D);
  CHECK_SHAPE(means2D, P, 2);
  CHECK_INPUT(out_opacity);
  CHECK_SHAPE(out_opacity, H, W);
  CHECK_INPUT(conic_opacity);
  CHECK_SHAPE(conic_opacity, P, 4);
  CHECK_INPUT(radii);
  CHECK_SHAPE(radii, P);
  if (dL_dout_extra.has_value()) {
    CHECK_INPUT(dL_dout_extra.value());
    CHECK_SHAPE(dL_dout_extra.value(), H, W, E);
  }

  CHECK_INPUT(dL_dout_color);
  CHECK_SHAPE(dL_dout_color, H, W, 3);
  CHECK_INPUT(dL_dout_opacity);
  CHECK_SHAPE(dL_dout_opacity, H, W);
  CHECK_INPUT(grad_means3D);
  CHECK_SHAPE(grad_means3D, P, 3);
  if (grad_scales.has_value()) {
    CHECK_INPUT(grad_scales.value());
    CHECK_SHAPE(grad_scales.value(), P, 3);
  }
  if (grad_rotations.has_value()) {
    CHECK_INPUT(grad_rotations.value());
    CHECK_SHAPE(grad_rotations.value(), P, 4);
  }
  if (grad_opacity.has_value()) {
    CHECK_INPUT(grad_opacity.value());
    CHECK_SHAPE(grad_opacity.value(), P, 1);
  }
  if (grad_colors.has_value()) {
    CHECK_INPUT(grad_colors.value());
    CHECK_SHAPE(grad_colors.value(), P, 3);
  }
  if (grad_shs.has_value()) {
    CHECK_INPUT(grad_shs.value());
    CHECK_SHAPE(grad_shs.value(), P, shs.size(1), 3);
  }
  if (grad_sh_rest.has_value()) {
    CHECK_INPUT(grad_sh_rest.value());
    CHECK_SHAPE(grad_sh_rest.value(), P, M, 3);
  }
  if (grad_extras.has_value()) {
    CHECK_INPUT(grad_extras.value());
    CHECK_SHAPE(grad_extras.value(), P, E);
  }
  if (grad_means2D.has_value()) {
    CHECK_INPUT(grad_means2D.value());
    CHECK_SHAPE(grad_means2D.value(), P, 2);
  }
  if (grad_Tw2v.has_value()) {
    CHECK_INPUT(grad_Tw2v.value());
    CHECK_SHAPE(grad_Tw2v.value(), 4, 4);
  }
  if (grad_campos.has_value()) {
    CHECK_INPUT(grad_campos.value());
    CHECK_SHAPE(grad_campos.value(), 3);
  }
  if (grad_cov.has_value()) {
    CHECK_INPUT(grad_cov.value());
    CHECK_SHAPE(grad_cov.value(), P, 6);
  }

  if (grad_conic_o.has_value()) {
    CHECK_INPUT(grad_conic_o.value());
    CHECK_SHAPE(grad_conic_o.value(), P, 4);
  }

  auto options = means3D.options();
  //   Tensor dL_dmeans3D = torch::zeros({P, 3}, options);
  Tensor dL_dmeans2D = grad_means2D.has_value() ? grad_means2D.value() : torch::zeros({P, 2}, options);
  Tensor dL_dcolors  = grad_colors.has_value() ? grad_colors.value() : torch::zeros({P, NUM_CHANNELS}, options);
  Tensor dL_dconic_o = grad_conic_o.has_value() ? grad_conic_o.value() : torch::zeros({P, 4}, options);

  const dim3 tile_grid((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, 1);
  const dim3 block(BLOCK_X, BLOCK_Y, 1);
  CHECK_SHAPE(ranges, tile_grid.x * tile_grid.y, 2);
  if (debug) {
    cudaDeviceSynchronize();
    CHECK_CUDA_ERROR("before render_backward");
  }
  if (P == 0) return;

  AT_DISPATCH_FLOATING_TYPES(means3D.scalar_type(), "gaussian_rasterize_backward", [&] {
    //   using scalar_t = float;
    const scalar_t tan_fovx = tanFoV[0].item<scalar_t>();
    const scalar_t tan_fovy = tanFoV[1].item<scalar_t>();
    const scalar_t focal_x  = 0.5 * W / tan_fovx;
    const scalar_t focal_y  = 0.5 * H / tan_fovy;
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
        (T2<scalar_t>*) dL_dmeans2D.data<scalar_t>(), (T4<scalar_t>*) dL_dconic_o.data_ptr<scalar_t>(),
        dL_dcolors.data_ptr<scalar_t>(), grad_extras.has_value() ? grad_extras.value().data<scalar_t>() : nullptr);
    if (debug) cudaDeviceSynchronize();
    CHECK_CUDA_ERROR("render_backward");

    preprocess_backward_kernel<scalar_t, NUM_CHANNELS> KERNEL_ARG((P + 255) / 256, 256)(                       //
        P, sh_degree, M, W, H, is_opengl,                                                                      //
        (vec3<scalar_t>*) means3D.data_ptr<scalar_t>(),                                                        //
        (vec3<scalar_t>*) scales.data_ptr<scalar_t>(),                                                         //
        (vec4<scalar_t>*) rotations.data_ptr<scalar_t>(),                                                      //
        (vec3<scalar_t>*) shs.data_ptr<scalar_t>(),                                                            //
        sh_rest.has_value() ? (vec3<scalar_t>*) sh_rest.value().data_ptr<scalar_t>() : nullptr,                //
        Tw2v.data_ptr<scalar_t>(), Tv2c.data_ptr<scalar_t>(), (scalar_t*) campos.data_ptr<scalar_t>(),         //
        focal_x, focal_y, tan_fovx, tan_fovy,                                                                  //
        radii.data_ptr<int>(),                                                                                 //
        (vec3<scalar_t>*) colors.data_ptr<scalar_t>(),                                                         //
        cov3Ds.has_value() ? cov3Ds.value().data_ptr<scalar_t>() : nullptr,                                    //
        cov2Ds.has_value() ? cov2Ds.value().data_ptr<scalar_t>() : nullptr,                                    //
        (T2<scalar_t>*) dL_dmeans2D.data_ptr<scalar_t>(),                                                      //
        grad_shs.has_value() ? dL_dcolors.data_ptr<scalar_t>() : nullptr,                                      //
        dL_dconic_o.data_ptr<scalar_t>(),                                                                      //
        (vec3<scalar_t>*) grad_means3D.data_ptr<scalar_t>(),                                                   //
        grad_scales.has_value() ? (vec3<scalar_t>*) grad_scales.value().data_ptr<scalar_t>() : nullptr,        //
        grad_rotations.has_value() ? (vec4<scalar_t>*) grad_rotations.value().data_ptr<scalar_t>() : nullptr,  //
        grad_opacity.has_value() ? grad_opacity.value().data_ptr<scalar_t>() : nullptr,                        //
        grad_shs.has_value() ? (vec3<scalar_t>*) grad_shs.value().data_ptr<scalar_t>() : nullptr,              //
        grad_sh_rest.has_value() ? (vec3<scalar_t>*) grad_sh_rest.value().data_ptr<scalar_t>() : nullptr,      //
        grad_cov.has_value() ? grad_cov.value().data_ptr<scalar_t>() : nullptr,                                //
        grad_Tw2v.has_value() ? grad_Tw2v.value().data_ptr<scalar_t>() : nullptr,                              //
        grad_campos.has_value() ? grad_campos.value().data_ptr<scalar_t>() : nullptr);
    if (debug) cudaDeviceSynchronize();
    CHECK_CUDA_ERROR("preprocessCUDA_backward");
  });
  return;
}

REGIST_PYTORCH_EXTENSION(gs_gaussian_rasterizer, {
  m.def("rasterize_gaussians", &RasterizeGaussiansCUDA);
  m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA);
})
}  // namespace GaussianRasterizer
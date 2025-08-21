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

// Helper function to find the next-highest bit of the MSB on the CPU.
uint32_t getHigherMsb(uint32_t n) {
  uint32_t msb  = sizeof(n) * 4;
  uint32_t step = msb;
  while (step > 1) {
    step /= 2;
    if (n >> msb)
      msb += step;
    else
      msb -= step;
  }
  if (n >> msb) msb++;
  return msb;
}

// Generates one key/value pair for all Gaussian / tile overlaps. Run once per Gaussian (1:N mapping).
template <typename T, typename TK, typename TV>
__global__ void duplicateWithKeys(int P, const T2<T>* points_xy, const T* depths, const int64_t* offsets,
    TK* gaussian_keys_unsorted, TV* gaussian_values_unsorted, int* radii, dim3 grid) {
  auto idx = cg::this_grid().thread_rank();
  if (idx >= P) return;

  // Generate no key/value pair for invisible Gaussians
  if (radii[idx] > 0) {
    // Find this Gaussian's offset in buffer for writing keys/values.
    int64_t off = (idx == 0) ? 0 : offsets[idx - 1];
    int2 rect_min, rect_max;

    getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);

    // For each tile that the bounding rect overlaps, emit a key/value pair.
    // The key is |  tile ID  |      depth      |, and the value is the ID of the Gaussian.
    // Sorting the values with this key yields Gaussian IDs in a list,
    // such that they are first sorted by tile and then by depth.
    float depth = static_cast<float>(depths[idx]);
    // depth = abs(depth); // use abs() to avoid negiative depth for OpenGL
    for (int y = rect_min.y; y < rect_max.y; y++) {
      for (int x = rect_min.x; x < rect_max.x; x++) {
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
}

__device__ inline float evaluate_opacity_factor(const float dx, const float dy, const float4 co) {
  return 0.5f * (co.x * dx * dx + co.z * dy * dy) + co.y * dx * dy;
}

template <typename T, uint32_t PATCH_WIDTH, uint32_t PATCH_HEIGHT>
__device__ inline float max_contrib_power_rect_gaussian_float(
    const T4<T> co, const T2<T> mean, const T2<T> rect_min, const T2<T> rect_max, T2<T>& max_pos) {
  const T x_min_diff = rect_min.x - mean.x;
  const T x_left     = x_min_diff > 0.0f;
  // const float x_left = mean.x < rect_min.x;
  const T not_in_x_range = x_left + (mean.x > rect_max.x);

  const T y_min_diff = rect_min.y - mean.y;
  const T y_above    = y_min_diff > 0.0f;
  // const float y_above = mean.y < rect_min.y;
  const T not_in_y_range = y_above + (mean.y > rect_max.y);

  max_pos             = {mean.x, mean.y};
  T max_contrib_power = 0.0f;

  if ((not_in_y_range + not_in_x_range) > 0.0f) {
    const T px = x_left * rect_min.x + (1.0f - x_left) * rect_max.x;
    const T py = y_above * rect_min.y + (1.0f - y_above) * rect_max.y;

    const T dx = copysign(T(PATCH_WIDTH), x_min_diff);
    const T dy = copysign(T(PATCH_HEIGHT), y_min_diff);

    const T diffx = mean.x - px;
    const T diffy = mean.y - py;

    const T rcp_dxdxcox = __frcp_rn(PATCH_WIDTH * PATCH_WIDTH * co.x);    // = 1.0 / (dx*dx*co.x)
    const T rcp_dydycoz = __frcp_rn(PATCH_HEIGHT * PATCH_HEIGHT * co.z);  // = 1.0 / (dy*dy*co.z)

    const T tx = not_in_y_range * __saturatef((dx * co.x * diffx + dx * co.y * diffy) * rcp_dxdxcox);
    const T ty = not_in_x_range * __saturatef((dy * co.y * diffx + dy * co.z * diffy) * rcp_dydycoz);
    max_pos    = {px + tx * dx, py + ty * dy};

    const T2<T> max_pos_diff = {mean.x - max_pos.x, mean.y - max_pos.y};
    max_contrib_power        = evaluate_opacity_factor(max_pos_diff.x, max_pos_diff.y, co);
  }
  return max_contrib_power;
}

// Generates one key/value pair for all Gaussian / tile overlaps. Run once per Gaussian (1:N mapping).
__global__ void duplicateWithKeys(int P, const float2* points_xy, const float4* __restrict__ conic_opacity,
    const float* depths, const int64_t* offsets, uint64_t* gaussian_keys_unsorted, uint32_t* gaussian_values_unsorted,
    int* radii, dim3 grid, float2* rects) {
  auto idx = cg::this_grid().thread_rank();
  if (idx >= P) return;

  // Generate no key/value pair for invisible Gaussians
  if (radii[idx] > 0) {
    // Find this Gaussian's offset in buffer for writing keys/values.
    int64_t off             = (idx == 0) ? 0 : offsets[idx - 1];
    const int64_t offset_to = offsets[idx];
    int2 rect_min, rect_max;

    if (rects == nullptr)
      getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);
    else
      getRect(points_xy[idx], rects[idx], rect_min, rect_max, grid);

    const float2 xy                      = points_xy[idx];
    const float4 co                      = conic_opacity[idx];
    const float opacity_threshold        = 1.0f / 255.0f;
    const float opacity_factor_threshold = logf(co.w / opacity_threshold);

    // For each tile that the bounding rect overlaps, emit a
    // key/value pair. The key is |  tile ID  |      depth      |,
    // and the value is the ID of the Gaussian. Sorting the values
    // with this key yields Gaussian IDs in a list, such that they
    // are first sorted by tile and then by depth.
    for (int y = rect_min.y; y < rect_max.y; y++) {
      for (int x = rect_min.x; x < rect_max.x; x++) {
        const T2<float> tile_min = {float(x * BLOCK_X), float(y * BLOCK_Y)};
        const T2<float> tile_max = {float((x + 1) * BLOCK_X - 1), float((y + 1) * BLOCK_Y - 1)};

        T2<float> max_pos;
        float max_opac_factor = 0.0f;
        max_opac_factor =
            max_contrib_power_rect_gaussian_float<float, BLOCK_X - 1, BLOCK_Y - 1>(co, xy, tile_min, tile_max, max_pos);

        uint64_t key = y * grid.x + x;
        key <<= 32;
        key |= *((uint32_t*) &depths[idx]);
        if (max_opac_factor <= opacity_factor_threshold) {
          gaussian_keys_unsorted[off]   = key;
          gaussian_values_unsorted[off] = idx;

          off++;
        }
      }
    }

    for (; off < offset_to; ++off) {
      uint64_t key = (uint32_t) -1;
      key <<= 32;
      const float depth = FLT_MAX;
      key |= *((uint32_t*) &depth);
      gaussian_values_unsorted[off] = static_cast<uint32_t>(-1);
      gaussian_keys_unsorted[off]   = key;
    }
  }
}

// Check keys to see if it is at the start/end of one tile's range in the full sorted list.
// If yes, write start/end of this tile. Run once per instanced (duplicated) Gaussian ID.
template <typename TK = uint64_t, typename T = int64_t>
__global__ void identifyTileRanges(int L, TK* point_list_keys, T2<T>* ranges) {
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
  bool valid_tile = currtile != (uint32_t) -1;
  if (idx == 0)
    ranges[currtile].x = 0;
  else {
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

template <typename T, typename TK>
int64_t prepare(int P, int width, int height, bool debug, T2<T>* means2D, T* depths, int* radii, int32_t* tiles_touched,
    int64_t* point_offsets, Tensor& scanning_space, T2<int64_t>* ranges, Tensor& point_list_unsorted,
    Tensor& point_list_keys_unsorted, Tensor& point_list, Tensor& point_list_keys, Tensor& list_sorting_space) {
  dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
  dim3 block(BLOCK_X, BLOCK_Y, 1);

  // Compute prefix sum over full list of touched tile counts by Gaussians
  // E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
  size_t scan_size;
  cub::DeviceScan::InclusiveSum(nullptr, scan_size, tiles_touched, point_offsets, P);  // get scan size
  scanning_space.resize_(scan_size);
  cub::DeviceScan::InclusiveSum(scanning_space.data_ptr(), scan_size, tiles_touched, point_offsets, P);
  if (debug) cudaDeviceSynchronize();
  CHECK_CUDA_ERROR("InclusiveSum");
  // Retrieve total number of Gaussian instances to launch and resize aux buffers
  int64_t num_rendered;
  cudaMemcpy(&num_rendered, point_offsets + P - 1, sizeof(int64_t), cudaMemcpyDeviceToHost);
  if (debug) cudaDeviceSynchronize();
  CHECK_CUDA_ERROR("cudaMemcpy");

  point_list_keys_unsorted.resize_(num_rendered);
  point_list_unsorted.resize_(num_rendered);
  point_list_keys.resize_(num_rendered);
  point_list.resize_(num_rendered);
  TK* key_unsorted        = point_list_keys_unsorted.data_ptr<TK>();
  int32_t* value_unsorted = point_list_unsorted.data_ptr<int32_t>();
  TK* key_sorted          = point_list_keys.data_ptr<TK>();
  int32_t* value_sorted   = point_list.data_ptr<int32_t>();

  // For each instance to be rendered, produce adequate [ tile | depth ] key
  // and corresponding dublicated Gaussian indices to be sorted
  duplicateWithKeys<T, TK, int32_t> KERNEL_ARG((P + 255) / 256, 256)(
      P, means2D, depths, point_offsets, key_unsorted, value_unsorted, radii, tile_grid);
  if (debug) cudaDeviceSynchronize();
  CHECK_CUDA_ERROR("duplicateWithKeys");

  int bit = getHigherMsb(tile_grid.x * tile_grid.y);

  // Sort complete list of (duplicated) Gaussian indices by keys
  size_t sorting_size;
  cub::DeviceRadixSort::SortPairs(
      nullptr, sorting_size, key_unsorted, key_sorted, value_unsorted, value_sorted, num_rendered, 0, 32 + bit);
  list_sorting_space.resize_(sorting_size);
  cub::DeviceRadixSort::SortPairs(list_sorting_space.data_ptr(), sorting_size, key_unsorted, key_sorted, value_unsorted,
      value_sorted, num_rendered, 0, 32 + bit);
  if (debug) cudaDeviceSynchronize();
  CHECK_CUDA_ERROR("SortPairs");
  cudaMemset(ranges, 0, tile_grid.x * tile_grid.y * sizeof(int2));
  if (debug) cudaDeviceSynchronize();
  CHECK_CUDA_ERROR("cudaMemset");
  // Identify start and end of per-tile workloads in sorted list
  if (num_rendered > 0) {
    identifyTileRanges<TK, int64_t> KERNEL_ARG((num_rendered + 255) / 256, 256)(num_rendered, key_sorted, ranges);
    if (debug) cudaDeviceSynchronize();
    CHECK_CUDA_ERROR("identifyTileRanges");
  }
  return num_rendered;
}

vector<Tensor> GS_prepare(
    int W, int H, Tensor& means2D, Tensor& depths, Tensor& radii, Tensor& tiles_touch, bool debug) {
  CHECK_INPUT(means2D);
  CHECK_INPUT(depths);
  CHECK_INPUT(radii);
  CHECK_INPUT(tiles_touch);
  int P = means2D.size(0);
  dim3 tile_grid((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, 1);

  Tensor ranges         = torch::zeros({tile_grid.x * tile_grid.y, 2}, means2D.options().dtype(torch::kInt64));
  Tensor point_offsets  = torch::zeros(P, means2D.options().dtype(torch::kInt64));
  Tensor sorted_key     = torch::zeros(0, means2D.options().dtype(torch::kInt64));
  Tensor sorted_value   = torch::zeros(0, means2D.options().dtype(torch::kInt32));
  Tensor unsorted_key   = torch::zeros(0, means2D.options().dtype(torch::kInt64));
  Tensor unsorted_value = torch::zeros(0, means2D.options().dtype(torch::kInt32));
  Tensor temp_scan      = torch::zeros(0, means2D.options().dtype(torch::kInt8));
  Tensor temp_sort      = torch::zeros(0, means2D.options().dtype(torch::kInt8));

  // using scalar_t = float;
  AT_DISPATCH_FLOATING_TYPES(means2D.scalar_type(), "prepare", [&] {
    prepare<scalar_t, int64_t>(P, W, H, debug, (T2<scalar_t>*) means2D.data_ptr<scalar_t>(),
        depths.data_ptr<scalar_t>(), radii.data_ptr<int32_t>(), tiles_touch.data_ptr<int32_t>(),
        point_offsets.data_ptr<int64_t>(), temp_scan, (T2<int64_t>*) ranges.data_ptr<int64_t>(), unsorted_value,
        unsorted_key, sorted_value, sorted_key, temp_sort);
  });
  return {ranges, sorted_value};
}

vector<Tensor> GS_prepare_v2(
    int W, int H, Tensor& means2D, Tensor& depths, Tensor& radii, Tensor& tiles_touch, bool debug) {
  CHECK_INPUT(means2D);
  CHECK_INPUT(depths);
  CHECK_INPUT(radii);
  CHECK_INPUT(tiles_touch);
  int P = means2D.size(0);
  dim3 tile_grid((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, 1);

  Tensor point_offsets = torch::cumsum(tiles_touch, 0, torch::kInt64);
  int64_t num_rendered = point_offsets.index({-1}).item<int64_t>();
  //  cudaMemcpy(&num_rendered, point_offsets + P - 1, sizeof(int32_t), cudaMemcpyDeviceToHost);
  if (debug) {
    cudaDeviceSynchronize();
    CHECK_CUDA_ERROR("point_offsets");
  }

  Tensor key, value;
  AT_DISPATCH_FLOATING_TYPES(means2D.scalar_type(), "sort point list", [&] {
    // using scalar_t     = float;
    // scalar_t min_depth = depths.min().item<scalar_t>();
    // scalar_t max_depth = depths.max().item<scalar_t>();
    // Tensor depth_ = depths * (2 / (max_depth - min_depth)) - 1 - min_depth * (2 / (max_depth - min_depth));
    Tensor depth_ = depths; /*in [-1., 1.]*/
    key           = torch::zeros(num_rendered, means2D.options().dtype(torch::kFloat64));
    value         = torch::zeros(num_rendered, means2D.options().dtype(torch::kInt32));
    duplicateWithKeys<scalar_t, double, int32_t> KERNEL_ARG((P + 255) / 256, 256)(P,
        (T2<scalar_t>*) means2D.data_ptr<scalar_t>(), depth_.data_ptr<scalar_t>(), point_offsets.data_ptr<int64_t>(),
        key.data_ptr<double>(), value.data_ptr<int32_t>(), radii.data_ptr<int32_t>(), tile_grid);
    if (debug) {
      cudaDeviceSynchronize();
      CHECK_CUDA_ERROR("duplicateWithKeys");
    }
  });

  auto sorted         = torch::sort(key);
  Tensor sorted_key   = std::get<0>(sorted);
  Tensor sorted_value = value.index({std::get<1>(sorted)});

  Tensor ranges = torch::zeros({tile_grid.x * tile_grid.y, 2}, means2D.options().dtype(torch::kInt64));

  if (num_rendered > 0) {
    identifyTileRanges<double, int64_t> KERNEL_ARG((num_rendered + 255) / 256, 256)(
        num_rendered, sorted_key.data_ptr<double>(), (T2<int64_t>*) ranges.data_ptr<int64_t>());
    if (debug) {
      cudaDeviceSynchronize();
      CHECK_CUDA_ERROR("identifyTileRanges");
    }
  }
  return {ranges, sorted_value};
}

vector<Tensor> GS_prepare_v3(int W, int H, Tensor& means2D, Tensor& conic_opacity, Tensor& depths, Tensor& radii,
    Tensor& tiles_touch, torch::optional<Tensor> rects) {
  CHECK_INPUT(means2D);
  CHECK_INPUT(depths);
  CHECK_INPUT(radii);
  CHECK_INPUT(tiles_touch);
  int P = means2D.size(0);
  dim3 tile_grid((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, 1);

  Tensor ranges         = torch::zeros({tile_grid.x * tile_grid.y, 2}, means2D.options().dtype(torch::kInt64));
  Tensor point_offsets  = torch::zeros(P, means2D.options().dtype(torch::kInt64));
  Tensor sorted_key     = torch::zeros(0, means2D.options().dtype(torch::kInt64));
  Tensor sorted_value   = torch::zeros(0, means2D.options().dtype(torch::kInt32));
  Tensor unsorted_key   = torch::zeros(0, means2D.options().dtype(torch::kInt64));
  Tensor unsorted_value = torch::zeros(0, means2D.options().dtype(torch::kInt32));
  Tensor temp_scan      = torch::zeros(0, means2D.options().dtype(torch::kInt8));
  Tensor temp_sort      = torch::zeros(0, means2D.options().dtype(torch::kInt8));

  using scalar_t = float;

  int32_t* tiles_touched_p = tiles_touch.data_ptr<int32_t>();
  int64_t* point_offsets_p = point_offsets.data_ptr<int64_t>();
  size_t scan_size;
  cudaDeviceSynchronize();
  cub::DeviceScan::InclusiveSum(nullptr, scan_size, tiles_touched_p, point_offsets_p, P);  // get scan size
  temp_scan.resize_(scan_size);
  cub::DeviceScan::InclusiveSum(temp_scan.data_ptr(), scan_size, tiles_touched_p, point_offsets_p, P);
  cudaDeviceSynchronize();
  CHECK_CUDA_ERROR("InclusiveSum");
  int64_t num_rendered;
  cudaMemcpy(&num_rendered, point_offsets_p + P - 1, sizeof(int64_t), cudaMemcpyDeviceToHost);
  CHECK_CUDA_ERROR("cudaMemcpy");

  unsorted_key.resize_(num_rendered);
  unsorted_value.resize_(num_rendered);
  sorted_key.resize_(num_rendered);
  sorted_value.resize_(num_rendered);
  int64_t* key_unsorted   = unsorted_key.data_ptr<int64_t>();
  int32_t* value_unsorted = unsorted_value.data_ptr<int32_t>();
  int64_t* key_sorted     = sorted_key.data_ptr<int64_t>();
  int32_t* value_sorted   = sorted_value.data_ptr<int32_t>();

  duplicateWithKeys KERNEL_ARG((P + 255) / 256, 256)(P, (T2<scalar_t>*) means2D.data_ptr<scalar_t>(),
      (T4<scalar_t>*) conic_opacity.data_ptr<scalar_t>(), depths.data_ptr<scalar_t>(), point_offsets_p,
      (uint64_t*) key_unsorted, (uint32_t*) value_unsorted, radii.data_ptr<int>(), tile_grid,
      rects.has_value() ? (T2<scalar_t>*) rects.value().data_ptr<scalar_t>() : nullptr);
  cudaDeviceSynchronize();
  CHECK_CUDA_ERROR("duplicateWithKeys");

  int bit = getHigherMsb(tile_grid.x * tile_grid.y);

  // Sort complete list of (duplicated) Gaussian indices by keys
  size_t sorting_size;
  cub::DeviceRadixSort::SortPairs(
      nullptr, sorting_size, key_unsorted, key_sorted, value_unsorted, value_sorted, num_rendered, 0, 32 + bit);
  temp_sort.resize_(sorting_size);
  cub::DeviceRadixSort::SortPairs(temp_sort.data_ptr(), sorting_size, key_unsorted, key_sorted, value_unsorted,
      value_sorted, num_rendered, 0, 32 + bit);
  cudaDeviceSynchronize();
  CHECK_CUDA_ERROR("SortPairs");
  int64_t* ranges_p = ranges.data_ptr<int64_t>();
  cudaMemset(ranges_p, 0, tile_grid.x * tile_grid.y * sizeof(int2));
  CHECK_CUDA_ERROR("cudaMemset");
  // Identify start and end of per-tile workloads in sorted list
  if (num_rendered > 0) {
    identifyTileRanges<int64_t, int64_t> KERNEL_ARG((num_rendered + 255) / 256, 256)(
        num_rendered, key_sorted, (T2<int64_t>*) ranges_p);
    cudaDeviceSynchronize();
    CHECK_CUDA_ERROR("identifyTileRanges");
  }

  return {ranges, sorted_value};
}

REGIST_PYTORCH_EXTENSION(gs_gaussian_prepare, {
  m.def("GS_prepare", &GS_prepare, "GS_prepare (CUDA)");
  m.def("GS_prepare_v2", &GS_prepare_v2, "GS_prepare_v2 (CUDA)");
  m.def("GS_prepare_v3", &GS_prepare_v3, "GS_prepare_v3 (CUDA)");
})

#define INSTANCE_FUNC(T, TK)                                                                                       \
  template int64_t prepare<T, TK>(int P, int width, int height, bool debug, T2<T>* means2D, T* depths, int* radii, \
      int32_t* tiles_touched, int64_t* point_offsets, Tensor& scanning_space, T2<int64_t>* ranges,                 \
      Tensor& point_list_unsorted, Tensor& point_list_keys_unsorted, Tensor& point_list, Tensor& point_list_keys,  \
      Tensor& list_sorting_space);

INSTANCE_FUNC(float, int64_t);
INSTANCE_FUNC(double, int64_t);

}  // namespace GaussianRasterizer
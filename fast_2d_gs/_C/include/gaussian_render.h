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
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include <functional>
#include <vector>

#include "device_launch_parameters.h"
// #include "ops_3d.h"
namespace GaussianRasterizer {
#define NUM_CHANNELS 3  // Default 3, RGB
#define BLOCK_X 16
#define BLOCK_Y 16
#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)
#define NUM_WARPS (BLOCK_SIZE / 32)

template <typename T2 = float2>
__forceinline__ __device__ void getRect(const T2 p, int max_radius, int2& rect_min, int2& rect_max, dim3 grid) {
  rect_min = {min((int) grid.x, max((int) 0, (int) ((p.x - max_radius) / BLOCK_X))),
      min((int) grid.y, max((int) 0, (int) ((p.y - max_radius) / BLOCK_Y)))};
  rect_max = {min((int) grid.x, max((int) 0, (int) ((p.x + max_radius + BLOCK_X - 1) / BLOCK_X))),
      min((int) grid.y, max((int) 0, (int) ((p.y + max_radius + BLOCK_Y - 1) / BLOCK_Y)))};
}

template <typename T2 = float2>
__forceinline__ __device__ void getRect(const T2 p, const T2 rect_extent, int2& rect_min, int2& rect_max, dim3 grid) {
  rect_min = {min((int) grid.x, max((int) 0, (int) floor((p.x - rect_extent.x) / BLOCK_X))),
      min((int) grid.y, max((int) 0, (int) floor((p.y - rect_extent.y) / BLOCK_Y)))};
  rect_max = {min((int) grid.x, max((int) 0, (int) ceil((p.x + rect_extent.x) / BLOCK_X))),
      min((int) grid.y, max((int) 0, (int) ceil((p.y + rect_extent.y) / BLOCK_Y)))};
}

__forceinline__ __device__ float dnormvdz(float3 v, float3 dv) {
  float sum2     = v.x * v.x + v.y * v.y + v.z * v.z;
  float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);
  float dnormvdz = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
  return dnormvdz;
}

__forceinline__ __device__ float3 dnormvdv(float3 v, float3 dv) {
  float sum2     = v.x * v.x + v.y * v.y + v.z * v.z;
  float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

  float3 dnormvdv;
  dnormvdv.x = ((+sum2 - v.x * v.x) * dv.x - v.y * v.x * dv.y - v.z * v.x * dv.z) * invsum32;
  dnormvdv.y = (-v.x * v.y * dv.x + (sum2 - v.y * v.y) * dv.y - v.z * v.y * dv.z) * invsum32;
  dnormvdv.z = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
  return dnormvdv;
}

__forceinline__ __device__ float4 dnormvdv(float4 v, float4 dv) {
  float sum2     = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
  float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

  float4 vdv    = {v.x * dv.x, v.y * dv.y, v.z * dv.z, v.w * dv.w};
  float vdv_sum = vdv.x + vdv.y + vdv.z + vdv.w;
  float4 dnormvdv;
  dnormvdv.x = ((sum2 - v.x * v.x) * dv.x - v.x * (vdv_sum - vdv.x)) * invsum32;
  dnormvdv.y = ((sum2 - v.y * v.y) * dv.y - v.y * (vdv_sum - vdv.y)) * invsum32;
  dnormvdv.z = ((sum2 - v.z * v.z) * dv.z - v.z * (vdv_sum - vdv.z)) * invsum32;
  dnormvdv.w = ((sum2 - v.w * v.w) * dv.w - v.w * (vdv_sum - vdv.w)) * invsum32;
  return dnormvdv;
}

__forceinline__ __device__ float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

};  // namespace GaussianRasterizer

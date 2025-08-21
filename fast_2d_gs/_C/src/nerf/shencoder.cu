#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstdio>
#include <stdexcept>

#include "ops_3d_types.h"
#include "spherical_harmonic.h"
#include "util.cuh"

using OPS_3D::vec3;

template <typename scalar_t>
__global__ void sh_forward_kernel(
    uint32_t B, uint32_t D, uint32_t C, const scalar_t *__restrict__ inputs, scalar_t *__restrict__ outputs) {
  const uint32_t b = threadIdx.x + blockIdx.x * blockDim.x;
  if (b >= B) return;

  const uint32_t C2 = (C + 1) * (C + 1);

  // locate
  inputs += b * D;
  outputs += b * C2;

  scalar_t x = inputs[0], y = inputs[1], z = inputs[2];

  scalar_t xy = x * y, xz = x * z, yz = y * z, x2 = x * x, y2 = y * y, z2 = z * z, xyz = xy * z;
  scalar_t x4 = x2 * x2, y4 = y2 * y2, z4 = z2 * z2;
  scalar_t x6 = x4 * x2, y6 = y4 * y2, z6 = z4 * z2;
  // 1/(2*sqrt(pi))
  outputs[0] = 0.28209479177387814f;
  if (C <= 0) return;

  // -sqrt(3)*y/(2*sqrt(pi))
  outputs[1] = -0.48860251190291987f * y;
  // sqrt(3)*z/(2*sqrt(pi))
  outputs[2] = 0.48860251190291987f * z;
  // -sqrt(3)*x/(2*sqrt(pi))
  outputs[3] = -0.48860251190291987f * x;
  if (C <= 1) return;

  // sqrt(15)*xy/(2*sqrt(pi))
  outputs[4] = 1.0925484305920792f * xy;
  // -sqrt(15)*yz/(2*sqrt(pi))
  outputs[5] = -1.0925484305920792f * yz;
  // sqrt(5)*(3*z2 - 1)/(4*sqrt(pi))
  outputs[6] = 0.94617469575755997f * z2 - 0.31539156525251999f;
  // -sqrt(15)*xz/(2*sqrt(pi))
  outputs[7] = -1.0925484305920792f * xz;
  // sqrt(15)*(x2 - y2)/(4*sqrt(pi))
  outputs[8] = 0.54627421529603959f * x2 - 0.54627421529603959f * y2;
  if (C <= 2) return;

  // sqrt(70)*y*(-3*x2 + y2)/(8*sqrt(pi))
  outputs[9] = 0.59004358992664352f * y * (-3.0f * x2 + y2);
  // sqrt(105)*xy*z/(2*sqrt(pi))
  outputs[10] = 2.8906114426405538f * xy * z;
  // sqrt(42)*y*(1 - 5*z2)/(8*sqrt(pi))
  outputs[11] = 0.45704579946446572f * y * (1.0f - 5.0f * z2);
  // sqrt(7)*z*(5*z2 - 3)/(4*sqrt(pi))
  outputs[12] = 0.3731763325901154f * z * (5.0f * z2 - 3.0f);
  // sqrt(42)*x*(1 - 5*z2)/(8*sqrt(pi))
  outputs[13] = 0.45704579946446572f * x * (1.0f - 5.0f * z2);
  // sqrt(105)*z*(x2 - y2)/(4*sqrt(pi))
  outputs[14] = 1.4453057213202769f * z * (x2 - y2);
  // sqrt(70)*x*(-x2 + 3*y2)/(8*sqrt(pi))
  outputs[15] = 0.59004358992664352f * x * (-x2 + 3.0f * y2);
  if (C <= 3) return;
  // 3*sqrt(35)*xy*(x2 - y2)/(4*sqrt(pi))
  outputs[16] = 2.5033429417967046f * xy * (x2 - y2);
  // 3*sqrt(70)*yz*(-3*x2 + y2)/(8*sqrt(pi))
  outputs[17] = 1.7701307697799304f * yz * (-3.0f * x2 + y2);
  // 3*sqrt(5)*xy*(7*z2 - 1)/(4*sqrt(pi))
  outputs[18] = 0.94617469575756008f * xy * (7.0f * z2 - 1.0f);
  // 3*sqrt(10)*yz*(3 - 7*z2)/(8*sqrt(pi))
  outputs[19] = 0.66904654355728921f * yz * (3.0f - 7.0f * z2);
  // 3*(-30*z2 + 35*z4 + 3)/(16*sqrt(pi))
  outputs[20] = -3.1735664074561294f * z2 + 3.7024941420321507f * z4 + 0.31735664074561293f;
  // 3*sqrt(10)*xz*(3 - 7*z2)/(8*sqrt(pi))
  outputs[21] = 0.66904654355728921f * xz * (3.0f - 7.0f * z2);
  // 3*sqrt(5)*(x2 - y2)*(7*z2 - 1)/(8*sqrt(pi))
  outputs[22] = 0.47308734787878004f * (x2 - y2) * (7.0f * z2 - 1.0f);
  // 3*sqrt(70)*xz*(-x2 + 3*y2)/(8*sqrt(pi))
  outputs[23] = 1.7701307697799304f * xz * (-x2 + 3.0f * y2);
  // 3*sqrt(35)*(-6*x2*y2 + x4 + y4)/(16*sqrt(pi))
  outputs[24] = -3.7550144126950569f * x2 * y2 + 0.62583573544917614f * x4 + 0.62583573544917614f * y4;
  if (C <= 4) return;

  // 3*sqrt(154)*y*(10*x2*y2 - 5*x4 - y4)/(32*sqrt(pi))
  outputs[25] = 0.65638205684017015f * y * (10.0f * x2 * y2 - 5.0f * x4 - y4);
  // 3*sqrt(385)*xy*z*(x2 - y2)/(4*sqrt(pi))
  outputs[26] = 8.3026492595241645f * xy * z * (x2 - y2);
  // -sqrt(770)*y*(3*x2 - y2)*(9*z2 - 1)/(32*sqrt(pi))
  outputs[27] = -0.48923829943525038f * y * (3.0f * x2 - y2) * (9.0f * z2 - 1.0f);
  // sqrt(1155)*xy*z*(3*z2 - 1)/(4*sqrt(pi))
  outputs[28] = 4.7935367849733241f * xy * z * (3.0f * z2 - 1.0f);
  // sqrt(165)*y*(14*z2 - 21*z4 - 1)/(16*sqrt(pi))
  outputs[29] = 0.45294665119569694f * y * (14.0f * z2 - 21.0f * z4 - 1.0f);
  // sqrt(11)*z*(-70*z2 + 63*z4 + 15)/(16*sqrt(pi))
  outputs[30] = 0.1169503224534236f * z * (-70.0f * z2 + 63.0f * z4 + 15.0f);
  // sqrt(165)*x*(14*z2 - 21*z4 - 1)/(16*sqrt(pi))
  outputs[31] = 0.45294665119569694f * x * (14.0f * z2 - 21.0f * z4 - 1.0f);
  // sqrt(1155)*z*(x2 - y2)*(3*z2 - 1)/(8*sqrt(pi))
  outputs[32] = 2.3967683924866621f * z * (x2 - y2) * (3.0f * z2 - 1.0f);
  // -sqrt(770)*x*(x2 - 3*y2)*(9*z2 - 1)/(32*sqrt(pi))
  outputs[33] = -0.48923829943525038f * x * (x2 - 3.0f * y2) * (9.0f * z2 - 1.0f);
  // 3*sqrt(385)*z*(-6*x2*y2 + x4 + y4)/(16*sqrt(pi))
  outputs[34] = 2.0756623148810411f * z * (-6.0f * x2 * y2 + x4 + y4);
  // 3*sqrt(154)*x*(10*x2*y2 - x4 - 5*y4)/(32*sqrt(pi))
  outputs[35] = 0.65638205684017015f * x * (10.0f * x2 * y2 - x4 - 5.0f * y4);
  if (C <= 5) return;

  // sqrt(6006)*xy*(-10*x2*y2 + 3*x4 + 3*y4)/(32*sqrt(pi))
  outputs[36] = 1.3663682103838286f * xy * (-10.0f * x2 * y2 + 3.0f * x4 + 3.0f * y4);
  // 3*sqrt(2002)*yz*(10*x2*y2 - 5*x4 - y4)/(32*sqrt(pi))
  outputs[37] = 2.3666191622317521f * yz * (10.0f * x2 * y2 - 5.0f * x4 - y4);
  // 3*sqrt(91)*xy*(x2 - y2)*(11*z2 - 1)/(8*sqrt(pi))
  outputs[38] = 2.0182596029148963f * xy * (x2 - y2) * (11.0f * z2 - 1.0f);
  // -sqrt(2730)*yz*(3*x2 - y2)*(11*z2 - 3)/(32*sqrt(pi))
  outputs[39] = -0.92120525951492349f * yz * (3.0f * x2 - y2) * (11.0f * z2 - 3.0f);
  // sqrt(2730)*xy*(-18*z2 + 33*z4 + 1)/(32*sqrt(pi))
  outputs[40] = 0.92120525951492349f * xy * (-18.0f * z2 + 33.0f * z4 + 1.0f);
  // sqrt(273)*yz*(30*z2 - 33*z4 - 5)/(16*sqrt(pi))
  outputs[41] = 0.58262136251873131f * yz * (30.0f * z2 - 33.0f * z4 - 5.0f);
  // sqrt(13)*(105*z2 - 315*z4 + 231*z6 - 5)/(32*sqrt(pi))
  outputs[42] = 6.6747662381009842f * z2 - 20.024298714302954f * z4 + 14.684485723822165f * z6 - 0.31784601133814211f;
  // sqrt(273)*xz*(30*z2 - 33*z4 - 5)/(16*sqrt(pi))
  outputs[43] = 0.58262136251873131f * xz * (30.0f * z2 - 33.0f * z4 - 5.0f);
  // sqrt(2730)*(x2 - y2)*(11*z2*(3*z2 - 1) - 7*z2 + 1)/(64*sqrt(pi))
  outputs[44] = 0.46060262975746175f * (x2 - y2) * (11.0f * z2 * (3.0f * z2 - 1.0f) - 7.0f * z2 + 1.0f);
  // -sqrt(2730)*xz*(x2 - 3*y2)*(11*z2 - 3)/(32*sqrt(pi))
  outputs[45] = -0.92120525951492349f * xz * (x2 - 3.0f * y2) * (11.0f * z2 - 3.0f);
  // 3*sqrt(91)*(11*z2 - 1)*(-6*x2*y2 + x4 + y4)/(32*sqrt(pi))
  outputs[46] = 0.50456490072872406f * (11.0f * z2 - 1.0f) * (-6.0f * x2 * y2 + x4 + y4);
  // 3*sqrt(2002)*xz*(10*x2*y2 - x4 - 5*y4)/(32*sqrt(pi))
  outputs[47] = 2.3666191622317521f * xz * (10.0f * x2 * y2 - x4 - 5.0f * y4);
  // sqrt(6006)*(15*x2*y4 - 15*x4*y2 + x6 - y6)/(64*sqrt(pi))
  outputs[48] = 10.247761577878714f * x2 * y4 - 10.247761577878714f * x4 * y2 + 0.6831841051919143f * x6 -
                0.6831841051919143f * y6;
  if (C <= 6) return;
  // 3*sqrt(715)*y*(-21*x2*y4 + 35*x4*y2 - 7*x6 + y6)/(64*sqrt(pi))
  outputs[49] = 0.70716273252459627f * y * (-21.0f * x2 * y4 + 35.0f * x4 * y2 - 7.0f * x6 + y6);
  // 3*sqrt(10010)*xy*z*(-10*x2*y2 + 3*x4 + 3*y4)/(32*sqrt(pi))
  outputs[50] = 5.2919213236038001f * xy * z * (-10.0f * x2 * y2 + 3.0f * x4 + 3.0f * y4);
  // -3*sqrt(385)*y*(13*z2 - 1)*(-10*x2*y2 + 5*x4 + y4)/(64*sqrt(pi))
  outputs[51] = -0.51891557872026028f * y * (13.0f * z2 - 1.0f) * (-10.0f * x2 * y2 + 5.0f * x4 + y4);
  // 3*sqrt(385)*xy*z*(x2 - y2)*(13*z2 - 3)/(8*sqrt(pi))
  outputs[52] = 4.1513246297620823f * xy * z * (x2 - y2) * (13.0f * z2 - 3.0f);
  // -3*sqrt(35)*y*(3*x2 - y2)*(13*z2*(11*z2 - 3) - 27*z2 + 3)/(64*sqrt(pi))
  outputs[53] = -0.15645893386229404f * y * (3.0f * x2 - y2) * (13.0f * z2 * (11.0f * z2 - 3.0f) - 27.0f * z2 + 3.0f);
  // 3*sqrt(70)*xy*z*(-110*z2 + 143*z4 + 15)/(32*sqrt(pi))
  outputs[54] = 0.44253269244498261f * xy * z * (-110.0f * z2 + 143.0f * z4 + 15.0f);
  // sqrt(105)*y*(-135*z2 + 495*z4 - 429*z6 + 5)/(64*sqrt(pi))
  outputs[55] = 0.090331607582517306f * y * (-135.0f * z2 + 495.0f * z4 - 429.0f * z6 + 5.0f);
  // sqrt(15)*z*(315*z2 - 693*z4 + 429*z6 - 35)/(32*sqrt(pi))
  outputs[56] = 0.068284276912004949f * z * (315.0f * z2 - 693.0f * z4 + 429.0f * z6 - 35.0f);
  // sqrt(105)*x*(-135*z2 + 495*z4 - 429*z6 + 5)/(64*sqrt(pi))
  outputs[57] = 0.090331607582517306f * x * (-135.0f * z2 + 495.0f * z4 - 429.0f * z6 + 5.0f);
  // sqrt(70)*z*(x2 - y2)*(143*z2*(3*z2 - 1) - 187*z2 + 45)/(64*sqrt(pi))
  outputs[58] = 0.07375544874083044f * z * (x2 - y2) * (143.0f * z2 * (3.0f * z2 - 1.0f) - 187.0f * z2 + 45.0f);
  // -3*sqrt(35)*x*(x2 - 3*y2)*(13*z2*(11*z2 - 3) - 27*z2 + 3)/(64*sqrt(pi))
  outputs[59] = -0.15645893386229404f * x * (x2 - 3.0f * y2) * (13.0f * z2 * (11.0f * z2 - 3.0f) - 27.0f * z2 + 3.0f);
  // 3*sqrt(385)*z*(13*z2 - 3)*(-6*x2*y2 + x4 + y4)/(32*sqrt(pi))
  outputs[60] = 1.0378311574405206f * z * (13.0f * z2 - 3.0f) * (-6.0f * x2 * y2 + x4 + y4);
  // -3*sqrt(385)*x*(13*z2 - 1)*(-10*x2*y2 + x4 + 5*y4)/(64*sqrt(pi))
  outputs[61] = -0.51891557872026028f * x * (13.0f * z2 - 1.0f) * (-10.0f * x2 * y2 + x4 + 5.0f * y4);
  // 3*sqrt(10010)*z*(15*x2*y4 - 15*x4*y2 + x6 - y6)/(64*sqrt(pi))
  outputs[62] = 2.6459606618019f * z * (15.0f * x2 * y4 - 15.0f * x4 * y2 + x6 - y6);
  // 3*sqrt(715)*x*(-35*x2*y4 + 21*x4*y2 - x6 + 7*y6)/(64*sqrt(pi))
  outputs[63] = 0.70716273252459627f * x * (-35.0f * x2 * y4 + 21.0f * x4 * y2 - x6 + 7.0f * y6);
}

template <typename scalar_t>
__global__ void kernel_sh_backward(const scalar_t *__restrict__ grad, const scalar_t *__restrict__ inputs, uint32_t B,
    uint32_t D, uint32_t C, scalar_t *grad_inputs) {
  const uint32_t t = threadIdx.x + blockIdx.x * blockDim.x;
  const uint32_t b = t / D;
  if (b >= B) return;

  // const uint32_t d  = t - b * D;
  const uint32_t C2 = (C + 1) * (C + 1);

  // locate
  grad += b * C2;
  inputs += b * D;
  grad_inputs += b * D;

  scalar_t x = inputs[0], y = inputs[1], z = inputs[2];
  scalar_t dx = 0, dy = 0, dz = 0;

  scalar_t xy = x * y, xz = x * z, yz = y * z, x2 = x * x, y2 = y * y, z2 = z * z, xyz = xy * z;
  scalar_t x4 = x2 * x2, y4 = y2 * y2, z4 = z2 * z2;
  scalar_t x6 = x4 * x2, y6 = y4 * y2, z6 = z4 * z2;

  // scalar_t *dx = dy_dx + b * D * C2;
  // scalar_t *dy = dx + C2;
  // scalar_t *dz = dy + C2;
  //(d[xyz])(\[\d+\])\s+=([ 0123456789.f*xyz()+\n-]*);\s+(//.*)
  auto write_sh_dx = [&]() {
    dx += grad[0] * (0.0f);
    if (C <= 0) return;

    // 0
    dx += grad[1] * (0.0f);
    // 0
    dx += grad[2] * (0.0f);
    // -sqrt(3)/(2*sqrt(pi))
    dx += grad[3] * (-0.48860251190291992f);
    if (C <= 1) return;

    // sqrt(15)*y/(2*sqrt(pi))
    dx += grad[4] * (1.0925484305920792f * y);
    // 0
    dx += grad[5] * (0.0f);
    // 0
    dx += grad[6] * (0.0f);
    // -sqrt(15)*z/(2*sqrt(pi))
    dx += grad[7] * (-1.0925484305920792f * z);
    // sqrt(15)*x/(2*sqrt(pi))
    dx += grad[8] * (1.0925484305920792f * x);
    if (C <= 2) return;

    // -3*sqrt(70)*xy/(4*sqrt(pi))
    dx += grad[9] * (-3.5402615395598609f * xy);
    // sqrt(105)*yz/(2*sqrt(pi))
    dx += grad[10] * (2.8906114426405538f * yz);
    // 0
    dx += grad[11] * (0.0f);
    // 0
    dx += grad[12] * (0.0f);
    // sqrt(42)*(1 - 5*z2)/(8*sqrt(pi))
    dx += grad[13] * (0.45704579946446572f - 2.2852289973223288f * z2);
    // sqrt(105)*xz/(2*sqrt(pi))
    dx += grad[14] * (2.8906114426405538f * xz);
    // 3*sqrt(70)*(-x2 + y2)/(8*sqrt(pi))
    dx += grad[15] * (-1.7701307697799304f * x2 + 1.7701307697799304f * y2);
    if (C <= 3) return;

    // 3*sqrt(35)*y*(3*x2 - y2)/(4*sqrt(pi))
    dx += grad[16] * (2.5033429417967046f * y * (3.0f * x2 - y2));
    // -9*sqrt(70)*xy*z/(4*sqrt(pi))
    dx += grad[17] * (-10.620784618679583f * xy * z);
    // 3*sqrt(5)*y*(7*z2 - 1)/(4*sqrt(pi))
    dx += grad[18] * (0.94617469575756008f * y * (7.0f * z2 - 1.0f));
    // 0
    dx += grad[19] * (0.0f);
    // 0
    dx += grad[20] * (0.0f);
    // 3*sqrt(10)*z*(3 - 7*z2)/(8*sqrt(pi))
    dx += grad[21] * (0.66904654355728921f * z * (3.0f - 7.0f * z2));
    // 3*sqrt(5)*x*(7*z2 - 1)/(4*sqrt(pi))
    dx += grad[22] * (0.94617469575756008f * x * (7.0f * z2 - 1.0f));
    // 9*sqrt(70)*z*(-x2 + y2)/(8*sqrt(pi))
    dx += grad[23] * (5.3103923093397913f * z * (-x2 + y2));
    // 3*sqrt(35)*x*(x2 - 3*y2)/(4*sqrt(pi))
    dx += grad[24] * (2.5033429417967046f * x * (x2 - 3.0f * y2));
    if (C <= 4) return;

    // 15*sqrt(154)*xy*(-x2 + y2)/(8*sqrt(pi))
    dx += grad[25] * (13.127641136803401f * xy * (-x2 + y2));
    // 3*sqrt(385)*yz*(3*x2 - y2)/(4*sqrt(pi))
    dx += grad[26] * (8.3026492595241645f * yz * (3.0f * x2 - y2));
    // 3*sqrt(770)*xy*(1 - 9*z2)/(16*sqrt(pi))
    dx += grad[27] * (2.9354297966115022f * xy * (1.0f - 9.0f * z2));
    // sqrt(1155)*yz*(3*z2 - 1)/(4*sqrt(pi))
    dx += grad[28] * (4.7935367849733241f * yz * (3.0f * z2 - 1.0f));
    // 0
    dx += grad[29] * (0.0f);
    // 0
    dx += grad[30] * (0.0f);
    // sqrt(165)*(14*z2 - 21*z4 - 1)/(16*sqrt(pi))
    dx += grad[31] * (6.3412531167397574f * z2 - 9.5118796751096362f * z4 - 0.45294665119569694f);
    // sqrt(1155)*xz*(3*z2 - 1)/(4*sqrt(pi))
    dx += grad[32] * (4.7935367849733241f * xz * (3.0f * z2 - 1.0f));
    // 3*sqrt(770)*(-9*x2*z2 + x2 + 9*y2*z2 - y2)/(32*sqrt(pi))
    dx += grad[33] * (-13.209434084751759f * x2 * z2 + 1.4677148983057511f * x2 + 13.209434084751759f * y2 * z2 -
                         1.4677148983057511f * y2);
    // 3*sqrt(385)*xz*(x2 - 3*y2)/(4*sqrt(pi))
    dx += grad[34] * (8.3026492595241645f * xz * (x2 - 3.0f * y2));
    // 15*sqrt(154)*(6*x2*y2 - x4 - y4)/(32*sqrt(pi))
    dx += grad[35] * (19.6914617052051f * x2 * y2 - 3.2819102842008503f * x4 - 3.2819102842008503f * y4);
    if (C <= 5) return;

    // 3*sqrt(6006)*y*(-10*x2*y2 + 5*x4 + y4)/(32*sqrt(pi))
    dx += grad[36] * (4.0991046311514854f * y * (-10.0f * x2 * y2 + 5.0f * x4 + y4));
    // 15*sqrt(2002)*xy*z*(-x2 + y2)/(8*sqrt(pi))
    dx += grad[37] * (47.332383244635047f * xy * z * (-x2 + y2));
    // 3*sqrt(91)*y*(3*x2 - y2)*(11*z2 - 1)/(8*sqrt(pi))
    dx += grad[38] * (2.0182596029148963f * y * (3.0f * x2 - y2) * (11.0f * z2 - 1.0f));
    // 3*sqrt(2730)*xy*z*(3 - 11*z2)/(16*sqrt(pi))
    dx += grad[39] * (5.5272315570895412f * xy * z * (3.0f - 11.0f * z2));
    // sqrt(2730)*y*(-18*z2 + 33*z4 + 1)/(32*sqrt(pi))
    dx += grad[40] * (0.92120525951492349f * y * (-18.0f * z2 + 33.0f * z4 + 1.0f));
    // 0
    dx += grad[41] * (0.0f);
    // 0
    dx += grad[42] * (0.0f);
    // sqrt(273)*z*(30*z2 - 33*z4 - 5)/(16*sqrt(pi))
    dx += grad[43] * (0.58262136251873131f * z * (30.0f * z2 - 33.0f * z4 - 5.0f));
    // sqrt(2730)*x*(-18*z2 + 33*z4 + 1)/(32*sqrt(pi))
    dx += grad[44] * (0.92120525951492349f * x * (-18.0f * z2 + 33.0f * z4 + 1.0f));
    // -3*sqrt(2730)*z*(x2 - y2)*(11*z2 - 3)/(32*sqrt(pi))
    dx += grad[45] * (-2.7636157785447706f * z * (x2 - y2) * (11.0f * z2 - 3.0f));
    // 3*sqrt(91)*x*(x2 - 3*y2)*(11*z2 - 1)/(8*sqrt(pi))
    dx += grad[46] * (2.0182596029148963f * x * (x2 - 3.0f * y2) * (11.0f * z2 - 1.0f));
    // 15*sqrt(2002)*z*(6*x2*y2 - x4 - y4)/(32*sqrt(pi))
    dx += grad[47] * (11.833095811158762f * z * (6.0f * x2 * y2 - x4 - y4));
    // 3*sqrt(6006)*x*(-10*x2*y2 + x4 + 5*y4)/(32*sqrt(pi))
    dx += grad[48] * (4.0991046311514854f * x * (-10.0f * x2 * y2 + x4 + 5.0f * y4));
    if (C <= 6) return;

    // 21*sqrt(715)*xy*(10*x2*y2 - 3*x4 - 3*y4)/(32*sqrt(pi))
    dx += grad[49] * (9.9002782553443485f * xy * (10.0f * x2 * y2 - 3.0f * x4 - 3.0f * y4));
    // 9*sqrt(10010)*yz*(-10*x2*y2 + 5*x4 + y4)/(32*sqrt(pi))
    dx += grad[50] * (15.875763970811402f * yz * (-10.0f * x2 * y2 + 5.0f * x4 + y4));
    // -15*sqrt(385)*xy*(x2 - y2)*(13*z2 - 1)/(16*sqrt(pi))
    dx += grad[51] * (-10.378311574405206f * xy * (x2 - y2) * (13.0f * z2 - 1.0f));
    // 3*sqrt(385)*yz*(3*x2 - y2)*(13*z2 - 3)/(8*sqrt(pi))
    dx += grad[52] * (4.1513246297620823f * yz * (3.0f * x2 - y2) * (13.0f * z2 - 3.0f));
    // 9*sqrt(35)*xy*(66*z2 - 143*z4 - 3)/(32*sqrt(pi))
    dx += grad[53] * (0.93875360317376422f * xy * (66.0f * z2 - 143.0f * z4 - 3.0f));
    // 3*sqrt(70)*yz*(-110*z2 + 143*z4 + 15)/(32*sqrt(pi))
    dx += grad[54] * (0.44253269244498261f * yz * (-110.0f * z2 + 143.0f * z4 + 15.0f));
    // 0
    dx += grad[55] * (0.0f);
    // 0
    dx += grad[56] * (0.0f);
    // sqrt(105)*(-135*z2 + 495*z4 - 429*z6 + 5)/(64*sqrt(pi))
    dx += grad[57] *
          (-12.194767023639836f * z2 + 44.714145753346067f * z4 - 38.752259652899923f * z6 + 0.45165803791258652f);
    // 3*sqrt(70)*xz*(-110*z2 + 143*z4 + 15)/(32*sqrt(pi))
    dx += grad[58] * (0.44253269244498261f * xz * (-110.0f * z2 + 143.0f * z4 + 15.0f));
    // 9*sqrt(35)*(66*x2*z2 - 143*x2*z4 - 3*x2 - 66*y2*z2 + 143*y2*z4 + 3*y2)/(64*sqrt(pi))
    dx += grad[59] * (30.97886890473422f * x2 * z2 - 67.120882626924143f * x2 * z4 - 1.4081304047606462f * x2 -
                         30.97886890473422f * y2 * z2 + 67.120882626924143f * y2 * z4 + 1.4081304047606462f * y2);
    // 3*sqrt(385)*xz*(x2 - 3*y2)*(13*z2 - 3)/(8*sqrt(pi))
    dx += grad[60] * (4.1513246297620823f * xz * (x2 - 3.0f * y2) * (13.0f * z2 - 3.0f));
    // -3*sqrt(385)*(13*z2 - 1)*(-10*x2*y2 + 4*x2*(x2 - 5*y2) + x4 + 5*y4)/(64*sqrt(pi))
    dx += grad[61] * (-0.51891557872026028f * (13.0f * z2 - 1.0f) *
                         (-10.0f * x2 * y2 + 4.0f * x2 * (x2 - 5.0f * y2) + x4 + 5.0f * y4));
    // 9*sqrt(10010)*xz*(-10*x2*y2 + x4 + 5*y4)/(32*sqrt(pi))
    dx += grad[62] * (15.875763970811402f * xz * (-10.0f * x2 * y2 + x4 + 5.0f * y4));
    // 21*sqrt(715)*(-15*x2*y4 + 15*x4*y2 - x6 + y6)/(64*sqrt(pi))
    dx += grad[63] * (-74.252086915082614f * x2 * y4 + 74.252086915082614f * x4 * y2 - 4.9501391276721742f * x6 +
                         4.9501391276721742f * y6);
  };

  auto write_sh_dy = [&]() {
    // 0
    dy += grad[0] * (0.0f);
    if (C <= 0) return;

    // -sqrt(3)/(2*sqrt(pi))
    dy += grad[1] * (-0.48860251190291992f);
    // 0
    dy += grad[2] * (0.0f);
    // 0
    dy += grad[3] * (0.0f);
    if (C <= 1) return;

    // sqrt(15)*x/(2*sqrt(pi))
    dy += grad[4] * (1.0925484305920792f * x);
    // -sqrt(15)*z/(2*sqrt(pi))
    dy += grad[5] * (-1.0925484305920792f * z);
    // 0
    dy += grad[6] * (0.0f);
    // 0
    dy += grad[7] * (0.0f);
    // -sqrt(15)*y/(2*sqrt(pi))
    dy += grad[8] * (-1.0925484305920792f * y);
    if (C <= 2) return;

    // 3*sqrt(70)*(-x2 + y2)/(8*sqrt(pi))
    dy += grad[9] * (-1.7701307697799304f * x2 + 1.7701307697799304f * y2);
    // sqrt(105)*xz/(2*sqrt(pi))
    dy += grad[10] * (2.8906114426405538f * xz);
    // sqrt(42)*(1 - 5*z2)/(8*sqrt(pi))
    dy += grad[11] * (0.45704579946446572f - 2.2852289973223288f * z2);
    // 0
    dy += grad[12] * (0.0f);
    // 0
    dy += grad[13] * (0.0f);
    // -sqrt(105)*yz/(2*sqrt(pi))
    dy += grad[14] * (-2.8906114426405538f * yz);
    // 3*sqrt(70)*xy/(4*sqrt(pi))
    dy += grad[15] * (3.5402615395598609f * xy);
    if (C <= 3) return;

    // 3*sqrt(35)*x*(x2 - 3*y2)/(4*sqrt(pi))
    dy += grad[16] * (2.5033429417967046f * x * (x2 - 3.0f * y2));
    // 9*sqrt(70)*z*(-x2 + y2)/(8*sqrt(pi))
    dy += grad[17] * (5.3103923093397913f * z * (-x2 + y2));
    // 3*sqrt(5)*x*(7*z2 - 1)/(4*sqrt(pi))
    dy += grad[18] * (0.94617469575756008f * x * (7.0f * z2 - 1.0f));
    // 3*sqrt(10)*z*(3 - 7*z2)/(8*sqrt(pi))
    dy += grad[19] * (0.66904654355728921f * z * (3.0f - 7.0f * z2));
    // 0
    dy += grad[20] * (0.0f);
    // 0
    dy += grad[21] * (0.0f);
    // 3*sqrt(5)*y*(1 - 7*z2)/(4*sqrt(pi))
    dy += grad[22] * (0.94617469575756008f * y * (1.0f - 7.0f * z2));
    // 9*sqrt(70)*xy*z/(4*sqrt(pi))
    dy += grad[23] * (10.620784618679583f * xy * z);
    // 3*sqrt(35)*y*(-3*x2 + y2)/(4*sqrt(pi))
    dy += grad[24] * (2.5033429417967046f * y * (-3.0f * x2 + y2));
    if (C <= 4) return;

    // 15*sqrt(154)*(6*x2*y2 - x4 - y4)/(32*sqrt(pi))
    dy += grad[25] * (19.6914617052051f * x2 * y2 - 3.2819102842008503f * x4 - 3.2819102842008503f * y4);
    // 3*sqrt(385)*xz*(x2 - 3*y2)/(4*sqrt(pi))
    dy += grad[26] * (8.3026492595241645f * xz * (x2 - 3.0f * y2));
    // -3*sqrt(770)*(x2 - y2)*(9*z2 - 1)/(32*sqrt(pi))
    dy += grad[27] * (-1.4677148983057511f * (x2 - y2) * (9.0f * z2 - 1.0f));
    // sqrt(1155)*xz*(3*z2 - 1)/(4*sqrt(pi))
    dy += grad[28] * (4.7935367849733241f * xz * (3.0f * z2 - 1.0f));
    // sqrt(165)*(14*z2 - 21*z4 - 1)/(16*sqrt(pi))
    dy += grad[29] * (6.3412531167397574f * z2 - 9.5118796751096362f * z4 - 0.45294665119569694f);
    // 0
    dy += grad[30] * (0.0f);
    // 0
    dy += grad[31] * (0.0f);
    // sqrt(1155)*yz*(1 - 3*z2)/(4*sqrt(pi))
    dy += grad[32] * (4.7935367849733241f * yz * (1.0f - 3.0f * z2));
    // 3*sqrt(770)*xy*(9*z2 - 1)/(16*sqrt(pi))
    dy += grad[33] * (2.9354297966115022f * xy * (9.0f * z2 - 1.0f));
    // 3*sqrt(385)*yz*(-3*x2 + y2)/(4*sqrt(pi))
    dy += grad[34] * (8.3026492595241645f * yz * (-3.0f * x2 + y2));
    // 15*sqrt(154)*xy*(x2 - y2)/(8*sqrt(pi))
    dy += grad[35] * (13.127641136803401f * xy * (x2 - y2));
    if (C <= 5) return;

    // 3*sqrt(6006)*x*(-10*x2*y2 + x4 + 5*y4)/(32*sqrt(pi))
    dy += grad[36] * (4.0991046311514854f * x * (-10.0f * x2 * y2 + x4 + 5.0f * y4));
    // 15*sqrt(2002)*z*(6*x2*y2 - x4 - y4)/(32*sqrt(pi))
    dy += grad[37] * (11.833095811158762f * z * (6.0f * x2 * y2 - x4 - y4));
    // 3*sqrt(91)*x*(x2 - 3*y2)*(11*z2 - 1)/(8*sqrt(pi))
    dy += grad[38] * (2.0182596029148963f * x * (x2 - 3.0f * y2) * (11.0f * z2 - 1.0f));
    // -3*sqrt(2730)*z*(x2 - y2)*(11*z2 - 3)/(32*sqrt(pi))
    dy += grad[39] * (-2.7636157785447706f * z * (x2 - y2) * (11.0f * z2 - 3.0f));
    // sqrt(2730)*x*(-18*z2 + 33*z4 + 1)/(32*sqrt(pi))
    dy += grad[40] * (0.92120525951492349f * x * (-18.0f * z2 + 33.0f * z4 + 1.0f));
    // sqrt(273)*z*(30*z2 - 33*z4 - 5)/(16*sqrt(pi))
    dy += grad[41] * (0.58262136251873131f * z * (30.0f * z2 - 33.0f * z4 - 5.0f));
    // 0
    dy += grad[42] * (0.0f);
    // 0
    dy += grad[43] * (0.0f);
    // sqrt(2730)*y*(18*z2 - 33*z4 - 1)/(32*sqrt(pi))
    dy += grad[44] * (0.92120525951492349f * y * (18.0f * z2 - 33.0f * z4 - 1.0f));
    // 3*sqrt(2730)*xy*z*(11*z2 - 3)/(16*sqrt(pi))
    dy += grad[45] * (5.5272315570895412f * xy * z * (11.0f * z2 - 3.0f));
    // -3*sqrt(91)*y*(3*x2 - y2)*(11*z2 - 1)/(8*sqrt(pi))
    dy += grad[46] * (-2.0182596029148963f * y * (3.0f * x2 - y2) * (11.0f * z2 - 1.0f));
    // 15*sqrt(2002)*xy*z*(x2 - y2)/(8*sqrt(pi))
    dy += grad[47] * (47.332383244635047f * xy * z * (x2 - y2));
    // 3*sqrt(6006)*y*(10*x2*y2 - 5*x4 - y4)/(32*sqrt(pi))
    dy += grad[48] * (4.0991046311514854f * y * (10.0f * x2 * y2 - 5.0f * x4 - y4));
    if (C <= 6) return;

    // 21*sqrt(715)*(-15*x2*y4 + 15*x4*y2 - x6 + y6)/(64*sqrt(pi))
    dy += grad[49] * (-74.252086915082614f * x2 * y4 + 74.252086915082614f * x4 * y2 - 4.9501391276721742f * x6 +
                         4.9501391276721742f * y6);
    // 9*sqrt(10010)*xz*(-10*x2*y2 + x4 + 5*y4)/(32*sqrt(pi))
    dy += grad[50] * (15.875763970811402f * xz * (-10.0f * x2 * y2 + x4 + 5.0f * y4));
    // 3*sqrt(385)*(13*z2 - 1)*(10*x2*y2 - 5*x4 + 4*y2*(5*x2 - y2) - y4)/(64*sqrt(pi))
    dy += grad[51] * (0.51891557872026028f * (13.0f * z2 - 1.0f) *
                         (10.0f * x2 * y2 - 5.0f * x4 + 4.0f * y2 * (5.0f * x2 - y2) - y4));
    // 3*sqrt(385)*xz*(x2 - 3*y2)*(13*z2 - 3)/(8*sqrt(pi))
    dy += grad[52] * (4.1513246297620823f * xz * (x2 - 3.0f * y2) * (13.0f * z2 - 3.0f));
    // -9*sqrt(35)*(x2 - y2)*(13*z2*(11*z2 - 3) - 27*z2 + 3)/(64*sqrt(pi))
    dy += grad[53] * (-0.46937680158688211f * (x2 - y2) * (13.0f * z2 * (11.0f * z2 - 3.0f) - 27.0f * z2 + 3.0f));
    // 3*sqrt(70)*xz*(-110*z2 + 143*z4 + 15)/(32*sqrt(pi))
    dy += grad[54] * (0.44253269244498261f * xz * (-110.0f * z2 + 143.0f * z4 + 15.0f));
    // sqrt(105)*(-135*z2 + 495*z4 - 429*z6 + 5)/(64*sqrt(pi))
    dy += grad[55] *
          (-12.194767023639836f * z2 + 44.714145753346067f * z4 - 38.752259652899923f * z6 + 0.45165803791258652f);
    // 0
    dy += grad[56] * (0.0f);
    // 0
    dy += grad[57] * (0.0f);
    // 3*sqrt(70)*yz*(110*z2 - 143*z4 - 15)/(32*sqrt(pi))
    dy += grad[58] * (0.44253269244498261f * yz * (110.0f * z2 - 143.0f * z4 - 15.0f));
    // 9*sqrt(35)*xy*(-66*z2 + 143*z4 + 3)/(32*sqrt(pi))
    dy += grad[59] * (0.93875360317376422f * xy * (-66.0f * z2 + 143.0f * z4 + 3.0f));
    // -3*sqrt(385)*yz*(3*x2 - y2)*(13*z2 - 3)/(8*sqrt(pi))
    dy += grad[60] * (-4.1513246297620823f * yz * (3.0f * x2 - y2) * (13.0f * z2 - 3.0f));
    // 15*sqrt(385)*xy*(x2 - y2)*(13*z2 - 1)/(16*sqrt(pi))
    dy += grad[61] * (10.378311574405206f * xy * (x2 - y2) * (13.0f * z2 - 1.0f));
    // 9*sqrt(10010)*yz*(10*x2*y2 - 5*x4 - y4)/(32*sqrt(pi))
    dy += grad[62] * (15.875763970811402f * yz * (10.0f * x2 * y2 - 5.0f * x4 - y4));
    // 21*sqrt(715)*xy*(-10*x2*y2 + 3*x4 + 3*y4)/(32*sqrt(pi))
    dy += grad[63] * (9.9002782553443485f * xy * (-10.0f * x2 * y2 + 3.0f * x4 + 3.0f * y4));
  };

  auto write_sh_dz = [&]() {
    // 0
    dz += grad[0] * (0.0f);
    if (C <= 0) return;

    // 0
    dz += grad[1] * (0.0f);
    // sqrt(3)/(2*sqrt(pi))
    dz += grad[2] * (0.48860251190291992f);
    // 0
    dz += grad[3] * (0.0f);
    if (C <= 1) return;

    // 0
    dz += grad[4] * (0.0f);
    // -sqrt(15)*y/(2*sqrt(pi))
    dz += grad[5] * (-1.0925484305920792f * y);
    // 3*sqrt(5)*z/(2*sqrt(pi))
    dz += grad[6] * (1.8923493915151204f * z);
    // -sqrt(15)*x/(2*sqrt(pi))
    dz += grad[7] * (-1.0925484305920792f * x);
    // 0
    dz += grad[8] * (0.0f);
    if (C <= 2) return;

    // 0
    dz += grad[9] * (0.0f);
    // sqrt(105)*xy/(2*sqrt(pi))
    dz += grad[10] * (2.8906114426405538f * xy);
    // -5*sqrt(42)*yz/(4*sqrt(pi))
    dz += grad[11] * (-4.5704579946446566f * yz);
    // 3*sqrt(7)*(5*z2 - 1)/(4*sqrt(pi))
    dz += grad[12] * (5.597644988851731f * z2 - 1.1195289977703462f);
    // -5*sqrt(42)*xz/(4*sqrt(pi))
    dz += grad[13] * (-4.5704579946446566f * xz);
    // sqrt(105)*(x2 - y2)/(4*sqrt(pi))
    dz += grad[14] * (1.4453057213202769f * x2 - 1.4453057213202769f * y2);
    // 0
    dz += grad[15] * (0.0f);
    if (C <= 3) return;

    // 0
    dz += grad[16] * (0.0f);
    // 3*sqrt(70)*y*(-3*x2 + y2)/(8*sqrt(pi))
    dz += grad[17] * (1.7701307697799304f * y * (-3.0f * x2 + y2));
    // 21*sqrt(5)*xy*z/(2*sqrt(pi))
    dz += grad[18] * (13.246445740605839f * xy * z);
    // 9*sqrt(10)*y*(1 - 7*z2)/(8*sqrt(pi))
    dz += grad[19] * (2.0071396306718676f * y * (1.0f - 7.0f * z2));
    // (105*z**3 - 45*z)/(4*sqrt(pi))
    dz += grad[20] * (14.809976568128603f * pow(z, 3) - 6.3471328149122579f * z);
    // 9*sqrt(10)*x*(1 - 7*z2)/(8*sqrt(pi))
    dz += grad[21] * (2.0071396306718676f * x * (1.0f - 7.0f * z2));
    // 21*sqrt(5)*z*(x2 - y2)/(4*sqrt(pi))
    dz += grad[22] * (6.6232228703029197f * z * (x2 - y2));
    // 3*sqrt(70)*x*(-x2 + 3*y2)/(8*sqrt(pi))
    dz += grad[23] * (1.7701307697799304f * x * (-x2 + 3.0f * y2));
    // 0
    dz += grad[24] * (0.0f);
    if (C <= 4) return;

    // 0
    dz += grad[25] * (0.0f);
    // 3*sqrt(385)*xy*(x2 - y2)/(4*sqrt(pi))
    dz += grad[26] * (8.3026492595241645f * xy * (x2 - y2));
    // 9*sqrt(770)*yz*(-3*x2 + y2)/(16*sqrt(pi))
    dz += grad[27] * (8.8062893898345074f * yz * (-3.0f * x2 + y2));
    // sqrt(1155)*xy*(9*z2 - 1)/(4*sqrt(pi))
    dz += grad[28] * (4.7935367849733241f * xy * (9.0f * z2 - 1.0f));
    // 7*sqrt(165)*yz*(1 - 3*z2)/(4*sqrt(pi))
    dz += grad[29] * (12.682506233479513f * yz * (1.0f - 3.0f * z2));
    // 15*sqrt(11)*(-14*z2 + 21*z4 + 1)/(16*sqrt(pi))
    dz += grad[30] * (-24.559567715218954f * z2 + 36.839351572828434f * z4 + 1.754254836801354f);
    // 7*sqrt(165)*xz*(1 - 3*z2)/(4*sqrt(pi))
    dz += grad[31] * (12.682506233479513f * xz * (1.0f - 3.0f * z2));
    // sqrt(1155)*(x2 - y2)*(9*z2 - 1)/(8*sqrt(pi))
    dz += grad[32] * (2.3967683924866621f * (x2 - y2) * (9.0f * z2 - 1.0f));
    // 9*sqrt(770)*xz*(-x2 + 3*y2)/(16*sqrt(pi))
    dz += grad[33] * (8.8062893898345074f * xz * (-x2 + 3.0f * y2));
    // 3*sqrt(385)*(-6*x2*y2 + x4 + y4)/(16*sqrt(pi))
    dz += grad[34] * (-12.453973889286246f * x2 * y2 + 2.0756623148810411f * x4 + 2.0756623148810411f * y4);
    // 0
    dz += grad[35] * (0.0f);
    if (C <= 5) return;

    // 0
    dz += grad[36] * (0.0f);
    // 3*sqrt(2002)*y*(10*x2*y2 - 5*x4 - y4)/(32*sqrt(pi))
    dz += grad[37] * (2.3666191622317521f * y * (10.0f * x2 * y2 - 5.0f * x4 - y4));
    // 33*sqrt(91)*xy*z*(x2 - y2)/(4*sqrt(pi))
    dz += grad[38] * (44.401711264127719f * xy * z * (x2 - y2));
    // -3*sqrt(2730)*y*(3*x2 - y2)*(11*z2 - 1)/(32*sqrt(pi))
    dz += grad[39] * (-2.7636157785447706f * y * (3.0f * x2 - y2) * (11.0f * z2 - 1.0f));
    // 3*sqrt(2730)*xy*z*(11*z2 - 3)/(8*sqrt(pi))
    dz += grad[40] * (11.054463114179082f * xy * z * (11.0f * z2 - 3.0f));
    // 5*sqrt(273)*y*(18*z2 - 33*z4 - 1)/(16*sqrt(pi))
    dz += grad[41] * (2.9131068125936568f * y * (18.0f * z2 - 33.0f * z4 - 1.0f));
    // 21*sqrt(13)*z*(-30*z2 + 33*z4 + 5)/(16*sqrt(pi))
    dz += grad[42] * (2.6699064952403937f * z * (-30.0f * z2 + 33.0f * z4 + 5.0f));
    // 5*sqrt(273)*x*(18*z2 - 33*z4 - 1)/(16*sqrt(pi))
    dz += grad[43] * (2.9131068125936568f * x * (18.0f * z2 - 33.0f * z4 - 1.0f));
    // 3*sqrt(2730)*z*(x2 - y2)*(11*z2 - 3)/(16*sqrt(pi))
    dz += grad[44] * (5.5272315570895412f * z * (x2 - y2) * (11.0f * z2 - 3.0f));
    // -3*sqrt(2730)*x*(x2 - 3*y2)*(11*z2 - 1)/(32*sqrt(pi))
    dz += grad[45] * (-2.7636157785447706f * x * (x2 - 3.0f * y2) * (11.0f * z2 - 1.0f));
    // 33*sqrt(91)*z*(-6*x2*y2 + x4 + y4)/(16*sqrt(pi))
    dz += grad[46] * (11.10042781603193f * z * (-6.0f * x2 * y2 + x4 + y4));
    // 3*sqrt(2002)*x*(10*x2*y2 - x4 - 5*y4)/(32*sqrt(pi))
    dz += grad[47] * (2.3666191622317521f * x * (10.0f * x2 * y2 - x4 - 5.0f * y4));
    // 0
    dz += grad[48] * (0.0f);
    if (C <= 6) return;

    // 0
    dz += grad[49] * (0.0f);
    // 3*sqrt(10010)*xy*(-10*x2*y2 + 3*x4 + 3*y4)/(32*sqrt(pi))
    dz += grad[50] * (5.2919213236038001f * xy * (-10.0f * x2 * y2 + 3.0f * x4 + 3.0f * y4));
    // 39*sqrt(385)*yz*(10*x2*y2 - 5*x4 - y4)/(32*sqrt(pi))
    dz += grad[51] * (13.491805046726766f * yz * (10.0f * x2 * y2 - 5.0f * x4 - y4));
    // 9*sqrt(385)*xy*(x2 - y2)*(13*z2 - 1)/(8*sqrt(pi))
    dz += grad[52] * (12.453973889286248f * xy * (x2 - y2) * (13.0f * z2 - 1.0f));
    // -33*sqrt(35)*yz*(3*x2 - y2)*(13*z2 - 3)/(16*sqrt(pi))
    dz += grad[53] * (-6.8841930899409371f * yz * (3.0f * x2 - y2) * (13.0f * z2 - 3.0f));
    // 15*sqrt(70)*xy*(-66*z2 + 143*z4 + 3)/(32*sqrt(pi))
    dz += grad[54] * (2.2126634622249131f * xy * (-66.0f * z2 + 143.0f * z4 + 3.0f));
    // 9*sqrt(105)*yz*(110*z2 - 143*z4 - 15)/(32*sqrt(pi))
    dz += grad[55] * (1.6259689364853116f * yz * (110.0f * z2 - 143.0f * z4 - 15.0f));
    // 7*sqrt(15)*(135*z2 - 495*z4 + 429*z6 - 5)/(32*sqrt(pi))
    dz += grad[56] *
          (64.528641681844675f * z2 - 236.60501950009714f * z4 + 205.05768356675085f * z6 - 2.3899496919201733f);
    // 9*sqrt(105)*xz*(110*z2 - 143*z4 - 15)/(32*sqrt(pi))
    dz += grad[57] * (1.6259689364853116f * xz * (110.0f * z2 - 143.0f * z4 - 15.0f));
    // sqrt(70)*(x2 - y2)*(143*z2*(3*z2 - 1) + 132*z2*(13*z2 - 5) - 187*z2 + 45)/(64*sqrt(pi))
    dz += grad[58] * (0.07375544874083044f * (x2 - y2) *
                         (143.0f * z2 * (3.0f * z2 - 1.0f) + 132.0f * z2 * (13.0f * z2 - 5.0f) - 187.0f * z2 + 45.0f));
    // -33*sqrt(35)*xz*(x2 - 3*y2)*(13*z2 - 3)/(16*sqrt(pi))
    dz += grad[59] * (-6.8841930899409371f * xz * (x2 - 3.0f * y2) * (13.0f * z2 - 3.0f));
    // 9*sqrt(385)*(13*z2 - 1)*(-6*x2*y2 + x4 + y4)/(32*sqrt(pi))
    dz += grad[60] * (3.1134934723215619f * (13.0f * z2 - 1.0f) * (-6.0f * x2 * y2 + x4 + y4));
    // 39*sqrt(385)*xz*(10*x2*y2 - x4 - 5*y4)/(32*sqrt(pi))
    dz += grad[61] * (13.491805046726766f * xz * (10.0f * x2 * y2 - x4 - 5.0f * y4));
    // 3*sqrt(10010)*(15*x2*y4 - 15*x4*y2 + x6 - y6)/(64*sqrt(pi))
    dz += grad[62] *
          (39.6894099270285f * x2 * y4 - 39.6894099270285f * x4 * y2 + 2.6459606618019f * x6 - 2.6459606618019f * y6);
    // 0
    dz += grad[63] * (0.0f);
  };
  write_sh_dx();
  write_sh_dy();
  write_sh_dz();
  grad_inputs[0] = dx;
  grad_inputs[1] = dy;
  grad_inputs[2] = dz;
}

void sh_encode_forward(at::Tensor inputs, at::Tensor outputs, const uint32_t B, const uint32_t D, const uint32_t C) {
  CHECK_CUDA(inputs);
  CHECK_CUDA(outputs);

  CHECK_CONTIGUOUS(inputs);
  CHECK_CONTIGUOUS(outputs);

  CHECK_IS_FLOATING(inputs);
  CHECK_IS_FLOATING(outputs);

  static constexpr uint32_t N_THREADS = 256;
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(inputs.scalar_type(), "sh_encode_forward_cuda", ([&] {
    sh_forward_kernel<scalar_t> KERNEL_ARG(div_round_up(B, N_THREADS), N_THREADS)(
        B, D, C, inputs.data_ptr<scalar_t>(), outputs.data_ptr<scalar_t>());
  }));
}

void sh_encode_backward(
    at::Tensor grad, at::Tensor inputs, const uint32_t B, const uint32_t D, const uint32_t C, at::Tensor grad_inputs) {
  CHECK_CUDA(grad);
  CHECK_CUDA(inputs);
  CHECK_CUDA(grad_inputs);

  CHECK_CONTIGUOUS(grad);
  CHECK_CONTIGUOUS(inputs);
  CHECK_CONTIGUOUS(grad_inputs);

  CHECK_IS_FLOATING(grad);
  CHECK_IS_FLOATING(inputs);
  CHECK_IS_FLOATING(grad_inputs);

  static constexpr uint32_t N_THREADS = 256;
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad.scalar_type(), "sh_encode_backward_cuda", ([&] {
    kernel_sh_backward<scalar_t> KERNEL_ARG(div_round_up(B * D, N_THREADS), N_THREADS)(
        grad.data_ptr<scalar_t>(), inputs.data_ptr<scalar_t>(), B, D, C, grad_inputs.data_ptr<scalar_t>());
  }));
}

REGIST_PYTORCH_EXTENSION(nerf_sh_encode, {
  m.def("sh_encode_forward", &sh_encode_forward, "SH encode forward (CUDA)");
  m.def("sh_encode_backward", &sh_encode_backward, "SH encode backward (CUDA)");
})
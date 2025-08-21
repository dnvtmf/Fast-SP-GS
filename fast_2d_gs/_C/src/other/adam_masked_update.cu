#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include "util.cuh"
namespace cg = cooperative_groups;

// step on a grid of size (N, M)
// N is always number of gaussians
template <typename T = float>
__global__ void adamUpdateCUDA(T* __restrict__ param, const T* __restrict__ param_grad, T* __restrict__ exp_avg,
    T* __restrict__ exp_avg_sq, const bool* tiles_touched, const T lr, const T b1, const T b2, const T eps,
    const uint32_t N, const uint32_t M) {
  auto p_idx           = cg::this_grid().thread_rank();
  const uint32_t g_idx = p_idx / M;
  if (g_idx >= N) return;
  if (tiles_touched[g_idx]) {
    T Register_param_grad = param_grad[p_idx];
    T Register_exp_avg    = exp_avg[p_idx];
    T Register_exp_avg_sq = exp_avg_sq[p_idx];
    Register_exp_avg      = b1 * Register_exp_avg + (T(1.0) - b1) * Register_param_grad;
    Register_exp_avg_sq   = b2 * Register_exp_avg_sq + (T(1.0) - b2) * Register_param_grad * Register_param_grad;
    T step                = -lr * Register_exp_avg / (sqrt(Register_exp_avg_sq) + eps);

    param[p_idx] += step;
    exp_avg[p_idx]    = Register_exp_avg;
    exp_avg_sq[p_idx] = Register_exp_avg_sq;
  }
}

void AdamMasksedUpdated(Tensor param, Tensor param_grad, Tensor exp_avg, Tensor exp_avg_sq, Tensor tiles_touched,
    double lr, double b1, double b2, double eps, size_t N, size_t M) {
  using scalar_t = float;
  adamUpdateCUDA<scalar_t> KERNEL_ARG(div_round_up<size_t>(N * M, 256), 256)(param.contiguous().data_ptr<scalar_t>(),
      param_grad.contiguous().data_ptr<scalar_t>(), exp_avg.contiguous().data_ptr<scalar_t>(),
      exp_avg_sq.contiguous().data_ptr<scalar_t>(), tiles_touched.contiguous().data_ptr<bool>(), lr, b1, b2, eps, N, M);
}

REGIST_PYTORCH_EXTENSION(
    adam_masked_update, { m.def("AdamMasksedUpdated", &AdamMasksedUpdated, "AdamMasksedUpdated (CUDA)"); });
#include "util.cuh"

inline constexpr __device__ float PI() { return 3.141592653589793f; }

// inputs: [B, D]
// outputs: [B, C], C = D + D * degree * 2
template <typename T>
__global__ void kernel_freq(const T* __restrict__ inputs, uint32_t B, uint32_t degree, uint32_t D_in, uint32_t D_out,
    const T scale, const bool include_input, T* __restrict__ outputs) {
  const uint32_t t = threadIdx.x + blockIdx.x * blockDim.x;
  if (t >= B * D_in) return;
  const uint32_t b = t / D_in, c = t % D_in;  // t % C;
  outputs += b * D_out + c;

  T x = inputs[t];
  // get index
  if (include_input) {
    outputs[0] = x;
    outputs += D_in;
  }
  x = x * scale;
  for (int d = 0; d < degree; ++d) {
    // const uint32_t col          = c / D - 1;
    // const uint32_t d            = c % D;
    // const uint32_t freq         = col / 2;
    // const float phase_shift     = (col % 2) * (PI() / 2);
    outputs[(2 * d + 0) * D_in] = sin(scalbnf(x, d));  // sin(x * 2 ** d)
    outputs[(2 * d + 1) * D_in] = cos(scalbnf(x, d));
  }
}

template <typename T>
__global__ void kernel_freq_backward(uint32_t B, uint32_t degree, uint32_t Di, uint32_t Do, const T scale,
    const bool include_input, const T* __restrict__ grad, const T* __restrict__ outputs, T* grad_inputs) {
  // parallel on per-element
  const uint32_t t = threadIdx.x + blockIdx.x * blockDim.x;
  if (t >= B * Di) return;

  const uint32_t b = t / Di, c = t % Di;

  // locate
  grad += b * Do + c;
  outputs += b * Do + c;

  // register
  T grad_x = 0;
  if (include_input) {
    grad_x = grad[0];
    grad += Di;
    outputs += Di;
  }

  for (uint32_t f = 0; f < degree; f++) {
    grad_x += scalbnf(T(1.0), f) * scale * (grad[0] * outputs[Di] - grad[Di] * outputs[0]);
    grad += 2 * Di;
    outputs += 2 * Di;
  }

  // write
  grad_inputs[t] = grad_x;
}

Tensor freq_encode_forward(Tensor& inputs, const uint32_t degree, bool include_inputs, double scale) {
  CHECK_INPUT(inputs);
  CHECK_IS_FLOATING(inputs);

  uint32_t input_dim  = inputs.size(-1);
  uint32_t output_dim = input_dim * (degree * 2 + include_inputs);
  uint32_t B          = inputs.numel() / input_dim;

  auto shape              = inputs.sizes().vec();
  shape[shape.size() - 1] = output_dim;
  Tensor outputs          = torch::zeros(shape, inputs.options());

  static constexpr uint32_t N_THREADS = 128;
  AT_DISPATCH_FLOATING_TYPES(inputs.scalar_type(), "kernel_freq", [&] {
    kernel_freq<scalar_t> KERNEL_ARG(div_round_up(B * input_dim, N_THREADS), N_THREADS)(inputs.data_ptr<scalar_t>(), B,
        degree, input_dim, output_dim, scale, include_inputs, outputs.data_ptr<scalar_t>());
  });
  return outputs;
}

Tensor freq_encode_backward(Tensor grad, Tensor outputs, uint32_t degree, bool include_inputs, double scale) {
  CHECK_INPUT(grad);
  CHECK_SHAPE(grad, outputs.sizes());

  CHECK_IS_FLOATING(grad);
  CHECK_IS_FLOATING(outputs);
  uint32_t output_dim = outputs.size(-1);
  uint32_t B          = outputs.numel() / output_dim;
  uint32_t input_dim  = output_dim / (degree * 2 + include_inputs);

  auto shape              = outputs.sizes().vec();
  shape[shape.size() - 1] = input_dim;
  Tensor grad_inputs      = torch::zeros(shape, outputs.options());

  static constexpr uint32_t N_THREADS = 128;
  AT_DISPATCH_FLOATING_TYPES(outputs.scalar_type(), "kernel_freq_backward", [&] {
    kernel_freq_backward<scalar_t> KERNEL_ARG(div_round_up(B * input_dim, N_THREADS), N_THREADS)(B, degree, input_dim,
        output_dim, scale, include_inputs, grad.data_ptr<scalar_t>(), outputs.data_ptr<scalar_t>(),
        grad_inputs.data_ptr<scalar_t>());
  });
  return grad_inputs;
}

REGIST_PYTORCH_EXTENSION(nerf_freq_encode, {
  m.def("freq_encode_forward", &freq_encode_forward, "freq encode forward (CUDA)");
  m.def("freq_encode_backward", &freq_encode_backward, "freq encode backward (CUDA)");
})

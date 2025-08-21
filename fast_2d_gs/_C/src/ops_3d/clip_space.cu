#include "util.cuh"

namespace OPS_3D {

template <typename T>
__host__ __device__ __forceinline__ void _perspective(T fovy, T aspect_r, T m22, T m23, T* out) {
  T c     = 1. / tan(fovy * (T) 0.5);
  out[0]  = c * aspect_r;
  out[5]  = c;
  out[10] = m22;
  out[11] = m23;
  out[14] = T(-1);
}

template <typename T>
void __global__ perspective_kernel(int N, const T* __restrict__ fovy, T aspect_r, T m22, T m23, T* __restrict__ out) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < N) _perspective(fovy[idx], aspect_r, m22, m23, out + idx * 16);
}

Tensor perspective(Tensor fovy, double aspect, double near, double far) {
  fovy       = fovy.contiguous();
  auto shape = fovy.sizes().vec();
  shape.push_back(4);
  shape.push_back(4);
  Tensor matrix = torch::zeros(shape, fovy.options());
  int N         = fovy.numel();

  if (fovy.is_cuda()) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(fovy.scalar_type(), "perspective (cuda)", [&] {
      double m22 = (near + far) / (near - far);
      double m23 = (2 * near * far) / (near - far);
      perspective_kernel<scalar_t> KERNEL_ARG(div_round_up(N, 256), 256)(
          N, fovy.data_ptr<scalar_t>(), 1. / aspect, m22, m23, matrix.data<scalar_t>());
      CHECK_CUDA_ERROR("perspective_kernel");
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES(fovy.scalar_type(), "perspective (cpu)", [&] {
      auto in_ptr  = fovy.data_ptr<scalar_t>();
      auto out_ptr = matrix.data_ptr<scalar_t>();
      scalar_t m22 = (near + far) / (near - far);
      scalar_t m23 = (2 * near * far) / (near - far);

      scalar_t raspect = 1. / aspect;
      for (int i = 0; i < N; ++i) {
        _perspective<scalar_t>(in_ptr[i], raspect, m22, m23, out_ptr + i * 16);
      }
    });
  }

  return matrix;
}

template <typename T>
void __host__ __device__ __forceinline__ _ortho(const T* __restrict__ box, T* __restrict__ out) {
  T l = box[0], r = box[1], b = box[2], t = box[3], n = box[4], f = box[5];
  out[0]  = 2 / (r - l);         // M[0, 0]
  out[5]  = 2 / (t - b);         // [1, 1]
  out[3]  = -(r + l) / (r - l);  // [0, 3]
  out[7]  = -(t + b) / (t - b);  // [1, 3]
  out[10] = -2 / (f - n);        // [2, 2]
  out[11] = -(f + n) / (f - n);  // [2, 3]
  out[15] = 1;                   // [3, 3]
}

template <typename T>
void __global__ ortho_kernel(int N, const T* __restrict__ box, T* __restrict__ out) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < N) _ortho(box + idx * 6, out + idx * 16);
}

Tensor ortho(Tensor box) {
  box        = box.contiguous();
  auto shape = box.sizes().vec();
  BCNN_ASSERT(shape.back() == 6, "ERROR shape of input");
  shape.pop_back();
  shape.push_back(4);
  shape.push_back(4);
  Tensor matrix = torch::zeros(shape, box.options());
  int N         = box.numel() / 6;

  if (box.is_cuda()) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(box.scalar_type(), "ortho (cuda)", [&] {
      ortho_kernel<scalar_t> KERNEL_ARG(div_round_up(N, 256), 256)(
          N, box.data_ptr<scalar_t>(), matrix.data<scalar_t>());
      CHECK_CUDA_ERROR("ortho_kernel");
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES(box.scalar_type(), "ortho (cpu)", [&] {
      auto in_ptr  = box.data_ptr<scalar_t>();
      auto out_ptr = matrix.data_ptr<scalar_t>();
      for (int i = 0; i < N; ++i) _ortho(in_ptr + i * 6, out_ptr + i * 16);
    });
  }
  return matrix;
}

REGIST_PYTORCH_EXTENSION(ops_3d_coordinate, {
  m.def("perspective", &perspective, "perspective (CUDA, CPU)");
  m.def("ortho", &ortho, "ortho (CUDA, CPU)");
})
}  // namespace OPS_3D
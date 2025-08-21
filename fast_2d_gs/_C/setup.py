import glob
import os
import time
from itertools import chain

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

SRC_DIR = os.path.abspath(os.path.dirname(__file__))


def get_cpp_or_cuda_sources(src_dir):
    files = glob.glob(f'{src_dir}/*.cu') + glob.glob(f'{src_dir}/*.cpp')
    print(f'\033[31mFind {len(files)} cu/cpp files in directory: {src_dir}\033[0m')
    return files


setup(
    name='Fast SP-GS',
    version='0.1.0',
    description='build time {}'.format(time.strftime("%y-%m-%d %H:%M:%S", time.localtime(time.time()))),
    ext_modules=[
        CUDAExtension(
            name='_C',
            sources=list(
                chain(
                    get_cpp_or_cuda_sources('src'),
                    get_cpp_or_cuda_sources('src/gaussian_splatting'),
                    get_cpp_or_cuda_sources('src/nerf'),
                    get_cpp_or_cuda_sources('src/ops_3d'),
                    get_cpp_or_cuda_sources('src/other'),
                )
            ),
            extra_compile_args={
                'cxx': ["-O3", "-Wno-deprecated-declarations"],
                'nvcc': [
                    '-O3',
                    '-rdc=true',
                    "-Wno-deprecated-declarations",
                    # '--ptxas-options=-v',
                ]
            },
            define_macros=[("__CUDA_NO_HALF_OPERATORS__", None)],
            include_dirs=[
                os.path.join(SRC_DIR, "include"),
            ],
            # libraries=[],
            # library_dirs=[]
        )
    ],
    cmdclass={'build_ext': BuildExtension},
)

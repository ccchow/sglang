"""
Setup script for building MeZO CUDA kernels
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

# Check CUDA availability
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required to build MeZO kernels")

# Define the extension
ext_modules = [
    CUDAExtension(
        name='mezo_cuda_ops',
        sources=['mezo_ops.cu'],
        extra_compile_args={
            'cxx': ['-O3'],
            'nvcc': [
                '-O3',
                '-gencode', 'arch=compute_70,code=sm_70',  # V100
                '-gencode', 'arch=compute_75,code=sm_75',  # T4
                '-gencode', 'arch=compute_80,code=sm_80',  # A100
                '-gencode', 'arch=compute_86,code=sm_86',  # RTX 3090
                '-gencode', 'arch=compute_89,code=sm_89',  # RTX 4090
                '--use_fast_math',
            ]
        }
    )
]

setup(
    name='mezo_cuda_ops',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension}
)
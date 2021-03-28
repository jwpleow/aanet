from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from torch.utils.cpp_extension import ROCM_HOME, CUDA_HOME
import torch
import path
import os

define_macros = []
extra_compile_args = {"cxx": []}

is_rocm_pytorch = (
        True if ((torch.version.hip is not None) and (ROCM_HOME is not None)) else False
    )
print(f"Cuda home: {CUDA_HOME}")
if (torch.cuda.is_available() and ((CUDA_HOME is not None) or is_rocm_pytorch)) or os.getenv(
    "FORCE_CUDA", "0"
) == "1":
 
    define_macros += [("WITH_CUDA", None)]
    extra_compile_args["nvcc"] = [
        "-O3",
        "-DCUDA_HAS_FP16=1",
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
    ]


setup(
    name='detectron2',
    ext_modules=[
        CUDAExtension('detectron2._C', [
            'csrc/vision.cpp',
            'csrc/deformable/deform_conv_cuda.cu',
            'csrc/deformable/deform_conv_cuda_kernel.cu',
        ], include_dirs = ['csrc/'], define_macros=define_macros, extra_compile_args=extra_compile_args),
    ],
    cmdclass={'build_ext': BuildExtension})

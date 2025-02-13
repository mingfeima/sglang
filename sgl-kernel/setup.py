import os
from pathlib import Path

import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

root = Path(__file__).parent.resolve()

force_cuda = os.environ.get("SGL_KERNEL_FORCE_CUDA", "0") == "1"

build_cuda_sources = torch.cuda.is_available() or force_cuda


def get_version():
    with open(root / "pyproject.toml") as f:
        for line in f:
            if line.startswith("version"):
                return line.split("=")[1].strip().strip('"')


def update_wheel_platform_tag():
    wheel_dir = Path("dist")
    if wheel_dir.exists() and wheel_dir.is_dir():
        old_wheel = next(wheel_dir.glob("*.whl"))
        new_wheel = wheel_dir / old_wheel.name.replace(
            "linux_x86_64", "manylinux2014_x86_64"
        )
        old_wheel.rename(new_wheel)


cutlass = root / "3rdparty" / "cutlass"
include_dirs = []
cuda_include_dirs = [
    cutlass.resolve() / "include",
    cutlass.resolve() / "tools" / "util" / "include",
    root / "src" / "sgl-kernel" / "csrc",
]
nvcc_flags = [
    "-O3",
    "-Xcompiler",
    "-fPIC",
    "-gencode=arch=compute_75,code=sm_75",
    "-gencode=arch=compute_80,code=sm_80",
    "-gencode=arch=compute_89,code=sm_89",
    "-gencode=arch=compute_90,code=sm_90",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
]
cxx_flags = ["-O3"]
extra_compile_args = {"cxx": cxx_flags}
libraries = ["c10", "torch", "torch_python"]
sources = [
    "src/sgl-kernel/csrc/cpu/interface.cpp",
    "src/sgl-kernel/csrc/cpu/shm.cpp",
]
cuda_sources = (
    [
        "src/sgl-kernel/csrc/trt_reduce_internal.cu",
        "src/sgl-kernel/csrc/trt_reduce_kernel.cu",
        "src/sgl-kernel/csrc/moe_align_kernel.cu",
        "src/sgl-kernel/csrc/int8_gemm_kernel.cu",
        "src/sgl-kernel/csrc/sampling_scaling_penalties.cu",
        "src/sgl-kernel/csrc/sgl_kernel_ops.cu",
    ],
)
if build_cuda_sources:
    sources.update(cuda_sources)
    include_dirs.extend(cuda_include_dirs)
    extra_compile_args.update({"nvcc": nvcc_flags})
    libraries.append("cuda")
    Extension = CUDAExtension
else:
    Extension = CppExtension

extra_link_args = ["-Wl,-rpath,$ORIGIN/../../torch/lib", "-L/usr/lib/x86_64-linux-gnu"]
ext_modules = [
    Extension(
        name="sgl_kernel.ops._kernels",
        sources=sources,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        libraries=libraries,
        extra_link_args=extra_link_args,
    ),
]

setup(
    name="sgl-kernel",
    version=get_version(),
    packages=["sgl_kernel"],
    package_dir={"": "src"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    install_requires=["torch"],
)

update_wheel_platform_tag()

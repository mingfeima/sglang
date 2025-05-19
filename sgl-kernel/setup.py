# Copyright 2025 SGLang Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import shutil
import sys
from pathlib import Path

import torch
from packaging import version
from setuptools import find_packages, setup
from setuptools.command.build_py import build_py
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

root = Path(__file__).parent.resolve()

force_cuda = os.environ.get("SGL_KERNEL_FORCE_CUDA", "0") == "1"

build_cuda_sources = torch.cuda.is_available() or force_cuda


if "bdist_wheel" in sys.argv and "--plat-name" not in sys.argv:
    sys.argv.extend(["--plat-name", "manylinux2014_x86_64"])


def _get_cuda_version():
    if torch.version.cuda:
        return tuple(map(int, torch.version.cuda.split(".")))
    return (0, 0)


def _get_device_sm():
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        return major * 10 + minor
    return 0


def _get_version():
    with open(root / "pyproject.toml") as f:
        for line in f:
            if line.startswith("version"):
                return line.split("=")[1].strip().strip('"')


operator_namespace = "sgl_kernel"
cutlass_default = root / "3rdparty" / "cutlass"
cutlass = Path(os.environ.get("CUSTOM_CUTLASS_SRC_DIR", default=cutlass_default))
flashinfer = root / "3rdparty" / "flashinfer"
deepgemm = root / "3rdparty" / "deepgemm"
include_dirs = []
cuda_include_dirs = [
    root / "include",
    root / "csrc",
    cutlass.resolve() / "include",
    cutlass.resolve() / "tools" / "util" / "include",
    flashinfer.resolve() / "include",
    flashinfer.resolve() / "include" / "gemm",
    flashinfer.resolve() / "csrc",
    "cublas",
]


class CustomBuildPy(build_py):
    def run(self):
        self.copy_deepgemm_to_build_lib()
        self.make_jit_include_symlinks()
        build_py.run(self)

    def make_jit_include_symlinks(self):
        # Make symbolic links of third-party include directories
        build_include_dir = os.path.join(self.build_lib, "deep_gemm/include")
        os.makedirs(build_include_dir, exist_ok=True)

        third_party_include_dirs = [
            cutlass.resolve() / "include" / "cute",
            cutlass.resolve() / "include" / "cutlass",
        ]

        for d in third_party_include_dirs:
            dirname = str(d).split("/")[-1]
            src_dir = d
            dst_dir = f"{build_include_dir}/{dirname}"
            assert os.path.exists(src_dir)
            if os.path.exists(dst_dir):
                assert os.path.islink(dst_dir)
                os.unlink(dst_dir)
            os.symlink(src_dir, dst_dir, target_is_directory=True)

    def copy_deepgemm_to_build_lib(self):
        """
        This function copies DeepGemm to python's site-packages
        """
        dst_dir = os.path.join(self.build_lib, "deep_gemm")
        os.makedirs(dst_dir, exist_ok=True)

        # Copy deepgemm/deep_gemm to the build directory
        src_dir = os.path.join(str(deepgemm.resolve()), "deep_gemm")

        # Remove existing directory if it exists
        if os.path.exists(dst_dir):
            shutil.rmtree(dst_dir)

        # Copy the directory
        shutil.copytree(src_dir, dst_dir)


nvcc_flags = [
    "-DNDEBUG",
    f"-DOPERATOR_NAMESPACE={operator_namespace}",
    "-O3",
    "-Xcompiler",
    "-fPIC",
    "-gencode=arch=compute_75,code=sm_75",
    "-gencode=arch=compute_80,code=sm_80",
    "-gencode=arch=compute_89,code=sm_89",
    "-gencode=arch=compute_90,code=sm_90",
    "-std=c++17",
    "-DFLASHINFER_ENABLE_F16",
    "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
    "-DCUTLASS_VERSIONS_GENERATED",
    "-DCUTE_USE_PACKED_TUPLE=1",
    "-DCUTLASS_TEST_LEVEL=0",
    "-DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1",
    "-DCUTLASS_DEBUG_TRACE_LEVEL=0",
    "--ptxas-options=-v",
    "--expt-relaxed-constexpr",
    "-Xcompiler=-Wconversion",
    "-Xcompiler=-fno-strict-aliasing",
]
nvcc_flags_fp8 = [
    "-DFLASHINFER_ENABLE_FP8",
    "-DFLASHINFER_ENABLE_FP8_E4M3",
    "-DFLASHINFER_ENABLE_FP8_E5M2",
]

sources = [
    "csrc/cpu/activation.cpp",
    "csrc/cpu/bmm.cpp",
    "csrc/cpu/decode.cpp",
    "csrc/cpu/extend.cpp",
    "csrc/cpu/gemm.cpp",
    "csrc/cpu/gemm_fp8.cpp",
    "csrc/cpu/gemm_int8.cpp",
    "csrc/cpu/gemm_int4_w4a16.cpp",
    "csrc/cpu/moe.cpp",
    "csrc/cpu/moe_fp8.cpp",
    "csrc/cpu/moe_int8.cpp",
    "csrc/cpu/moe_int4_w4a16.cpp",
    "csrc/cpu/norm.cpp",
    "csrc/cpu/qkv_proj.cpp",
    "csrc/cpu/rope.cpp",
    "csrc/cpu/topk.cpp",
    "csrc/cpu/interface.cpp",
    "csrc/cpu/shm.cpp",
    "csrc/cpu/torch_extension_cpu.cpp",
    "csrc/cpu/models/deepseek.cpp",
]
cuda_sources = (
    [
        "csrc/allreduce/trt_reduce_internal.cu",
        "csrc/allreduce/trt_reduce_kernel.cu",
        "csrc/attention/lightning_attention_decode_kernel.cu",
        "csrc/elementwise/activation.cu",
        "csrc/elementwise/fused_add_rms_norm_kernel.cu",
        "csrc/elementwise/rope.cu",
        "csrc/gemm/bmm_fp8.cu",
        "csrc/gemm/cublas_grouped_gemm.cu",
        "csrc/gemm/awq_kernel.cu",
        "csrc/gemm/fp8_gemm_kernel.cu",
        "csrc/gemm/fp8_blockwise_gemm_kernel.cu",
        "csrc/gemm/int8_gemm_kernel.cu",
        "csrc/gemm/per_token_group_quant_fp8.cu",
        "csrc/gemm/per_token_quant_fp8.cu",
        "csrc/gemm/per_tensor_quant_fp8.cu",
        "csrc/moe/moe_align_kernel.cu",
        "csrc/speculative/eagle_utils.cu",
        "csrc/speculative/speculative_sampling.cu",
        "csrc/torch_extension.cc",
        "3rdparty/flashinfer/csrc/norm.cu",
        "3rdparty/flashinfer/csrc/renorm.cu",
        "3rdparty/flashinfer/csrc/sampling.cu",
    ],
)
enable_bf16 = os.getenv("SGL_KERNEL_ENABLE_BF16", "0") == "1"
enable_fp8 = os.getenv("SGL_KERNEL_ENABLE_FP8", "0") == "1"
enable_sm90a = os.getenv("SGL_KERNEL_ENABLE_SM90A", "0") == "1"
enable_sm100a = os.getenv("SGL_KERNEL_ENABLE_SM100A", "0") == "1"
cuda_version = _get_cuda_version()
sm_version = _get_device_sm()
cpu_fp8_ftz = os.getenv("SGLANG_CPU_FP8_CVT_FTZ", "1") == "1"
cpu_amx_int8 = version.parse(torch.__version__) >= version.parse("2.7")
if cpu_amx_int8:
    print("Enable AMX-INT8 build on CPU kernels.")

if torch.cuda.is_available():
    if cuda_version >= (12, 0) and sm_version >= 90:
        nvcc_flags.append("-gencode=arch=compute_90a,code=sm_90a")
    if cuda_version >= (12, 8) and sm_version >= 100:
        nvcc_flags.append("-gencode=arch=compute_100,code=sm_100")
        nvcc_flags.append("-gencode=arch=compute_100a,code=sm_100a")
    if sm_version >= 90:
        nvcc_flags.extend(nvcc_flags_fp8)
    if sm_version >= 80:
        nvcc_flags.append("-DFLASHINFER_ENABLE_BF16")
else:
    # compilation environment without GPU
    if enable_sm90a:
        nvcc_flags.append("-gencode=arch=compute_90a,code=sm_90a")
    if enable_sm100a:
        nvcc_flags.append("-gencode=arch=compute_100a,code=sm_100a")
    if enable_fp8:
        nvcc_flags.extend(nvcc_flags_fp8)
    if enable_bf16:
        nvcc_flags.append("-DFLASHINFER_ENABLE_BF16")

for flag in [
    "-D__CUDA_NO_HALF_OPERATORS__",
    "-D__CUDA_NO_HALF_CONVERSIONS__",
    "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-D__CUDA_NO_HALF2_OPERATORS__",
]:
    try:
        torch.utils.cpp_extension.COMMON_NVCC_FLAGS.remove(flag)
    except ValueError:
        pass

extra_compile_args = {
    "cxx": [
        "-O3",
        "-Wno-unknown-pragmas",
        "-march=native",
        "-fopenmp",
    ]
}
if cpu_fp8_ftz:
    extra_compile_args["cxx"].append("-DSGLANG_CPU_FP8_CVT_FTZ")
if cpu_amx_int8:
    extra_compile_args["cxx"].append("-DSGLANG_CPU_AMX_INT8")

libraries = ["c10", "torch", "torch_python"]
cuda_libraries = ["cuda", "cublas"]
cmdclass = {
    "build_ext": BuildExtension.with_options(use_ninja=True),
}
if build_cuda_sources:
    sources.update(cuda_sources)
    include_dirs.extend(cuda_include_dirs)
    extra_compile_args.update({"nvcc": nvcc_flags})
    libraries.extend(cuda_libraries)
    cmdclass.update(
        {
            "build_py": CustomBuildPy,
        }
    )
    Extension = CUDAExtension
else:
    Extension = CppExtension

extra_link_args = ["-Wl,-rpath,$ORIGIN/../../torch/lib", "-L/usr/lib/x86_64-linux-gnu"]

# https://github.com/pytorch/pytorch/issues/152243
py_limited_api = version.parse(torch.__version__) < version.parse("2.7")

ext_modules = [
    Extension(
        name="sgl_kernel.common_ops",
        sources=sources,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        libraries=libraries,
        extra_link_args=extra_link_args,
        py_limited_api=py_limited_api,
    ),
]

setup(
    name="sgl-kernel",
    version=_get_version(),
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    options={"bdist_wheel": {"py_limited_api": "cp39"}},
)

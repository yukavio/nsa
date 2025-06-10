import os
from pathlib import Path
from setuptools import setup, find_packages
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension



device_capability = torch.cuda.get_device_capability()
device_capability = f"{device_capability[0]}{device_capability[1]}"

cwd = Path(os.path.dirname(os.path.abspath(__file__)))

nvcc_flags = [
    "-std=c++17",
    "-DENABLE_BF16",
]

if device_capability:
    nvcc_flags.extend([
        f"--generate-code=arch=compute_{device_capability},code=sm_{device_capability}",
    ])

# ext_modules = [
#     CUDAExtension(
#         "nsa_backend",
#         ["csrc/ops.cu", "csrc/group_gemm.cu"],
#         include_dirs = [
#             f"{cwd}/third_party/cutlass/include/"
#         ],
#         extra_compile_args={
#             "cxx": [
#                 "-fopenmp", "-fPIC", "-Wno-strict-aliasing"
#             ],
#             "nvcc": nvcc_flags,
#         }
#     )
# ]

setup(
    name="nsa",
    version="0.0.1",
    author="kavioyu",
    author_email="kavioyu@gmail.com",
    description="GEMM Grouped",
    url="https://github.com/yukavio/nsa",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: Unix",
    ],
    packages=find_packages(),
    #ext_modules=ext_modules,
    #cmdclass={"build_ext": BuildExtension},
    install_requires=[
        "flash-attn",
        "flash-linear-attention@git+https://github.com/fla-org/flash-linear-attention"
        ],
)

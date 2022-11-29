# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
""" Build the adaptive_gridsampler_cuda.so """

import os
import sys
from setuptools import setup

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

cxx_args = ['-std=c++14']

so_name = 'adaptive_gridsampler_cuda.so'
if os.path.exists(so_name):
    os.remove(so_name)

name = 'adaptive_gridsampler_cuda'
nvcc_args = [
    '-gencode', 'arch=compute_60,code=sm_60',
    '-gencode', 'arch=compute_61,code=sm_61',
    '-gencode', 'arch=compute_70,code=sm_70',
    '-gencode', 'arch=compute_70,code=compute_70'
]
sys.argv.append('build_ext')
sys.argv.append('-i')
setup(
    name=name,
    ext_modules=[
        CUDAExtension(name,
                      ['adaptive_gridsampler_cuda.cpp', 'adaptive_gridsampler_kernel.cu'],
                      extra_compile_args={'cxx': cxx_args, 'nvcc': nvcc_args})
    ],
    cmdclass={'build_ext': BuildExtension}
)

files = os.listdir(".")
old_name = None
for f in files:
    if  f.endswith('.so'):
        old_name = f
if old_name:
    os.rename(old_name, so_name)

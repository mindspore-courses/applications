#!/bin/bash
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

project_path=$1
build_path=$2
if [[ ! -d "$project_path" ]]; then
    echo "[ERROR] No projcet path is provided"
    exit 1
fi

if [[ ! -d "$build_path" ]]; then
    echo "[ERROR] No build path is provided"
    exit 1
fi

# copy ai_core operators implements
tbe_impl_files_num=$(ls $project_path/tbe/custom_impl/*.py 2> /dev/null | wc -l)
if [[ "$tbe_impl_files_num" -gt 0 ]];then
    cp -f ${project_path}/tbe/custom_impl/*.py ${build_path}/makepkg/packages/op_impl/custom/ai_core/tbe/custom_impl
    cp -f ${project_path}/tbe/custom_impl/*.py ${build_path}/makepkg/packages/op_impl/custom/vector_core/tbe/custom_impl
fi

# copy aicpu kernel so operators
if [[ -d "${project_path}/cpukernel/aicpu_kernel_lib" ]]; then
    cp -f ${project_path}/cpukernel/aicpu_kernel_lib/* ${build_path}/makepkg/packages/op_impl/custom/cpu/aicpu_kernel/custom_impl
    rm -rf ${project_path}/cpukernel/aicpu_kernel_lib
fi

# merge aicpu.ini and aicore.ini to generate npu_supported_ops.json
mkdir -p ${build_path}/framework/op_info_cfg
mkdir -p ${build_path}/framework/op_info_cfg/aicpu_kernel
mkdir -p ${build_path}/framework/op_info_cfg/ai_core

if [[ -d "${project_path}/tbe/op_info_cfg/ai_core" ]]; then
    bash ${project_path}/cmake/util/gen_ops_filter.sh ${project_path}/tbe/op_info_cfg/ai_core ${build_path}/framework/op_info_cfg/ai_core
fi

if [[ -d "${project_path}/cpukernel/op_info_cfg/aicpu_kernel" ]]; then
    bash ${project_path}/cmake/util/gen_ops_filter.sh ${project_path}/cpukernel/op_info_cfg/aicpu_kernel ${build_path}/framework/op_info_cfg/aicpu_kernel
fi

aicpu_filter_file=${build_path}/framework/op_info_cfg/aicpu_kernel/npu_supported_ops.json
aicore_filter_file=${build_path}/framework/op_info_cfg/ai_core/npu_supported_ops.json
if [[ -f "${aicpu_filter_file}" ]] && [[ ! -f "${aicore_filter_file}" ]]; then
    cp $aicpu_filter_file ${build_path}/makepkg/packages/framework/custom/tensorflow
fi
if [[ -f "${aicore_filter_file}" ]] && [[ ! -f "${aicpu_filter_file}" ]]; then
    cp $aicore_filter_file ${build_path}/makepkg/packages/framework/custom/tensorflow
fi

if [[ -f "${aicore_filter_file}" ]] && [[ -f "${aicpu_filter_file}" ]]; then
    chmod u+w ${aicpu_filter_file}
    python3.7.5 ${project_path}/cmake/util/insert_op_info.py ${aicore_filter_file} ${aicpu_filter_file}
    chmod u-w ${aicpu_filter_file}
    cp $aicpu_filter_file ${build_path}/makepkg/packages/framework/custom/tensorflow
fi


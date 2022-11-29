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
echo $@
if [[ ! -d "$project_path" ]]; then
    echo "[ERROR] No projcet path is provided"
    exit 1
fi

if [[ ! -d "$build_path" ]]; then
    echo "[ERROR] No build path is provided"
    exit 1
fi

if [[ ! -d "$ASCEND_OPP_PATH" ]]; then
    echo "[ERROR] No opp install path is provided"
    exit 1
fi
custom_exist_info_json=$ASCEND_OPP_PATH/op_impl/custom/cpu/config/cust_aicpu_kernel.json
custom_new_info_json=$build_path/makepkg/packages/op_impl/custom/cpu/config/cust_aicpu_kernel.json
temp_info_json=$build_path/makepkg/packages/op_impl/custom/cpu/config/temp_cust_aicpu_kernel.json

if [[ -f "$custom_exist_info_json" ]] && [[ -f "$custom_new_info_json" ]]; then
    cp -f $custom_exist_info_json $temp_info_json
    chmod +w $temp_info_json
    python3.7.5 ${project_path}/cmake/util/insert_op_info.py ${custom_new_info_json} ${temp_info_json}
    cp -f $temp_info_json $custom_new_info_json
    rm -f $temp_info_json
fi

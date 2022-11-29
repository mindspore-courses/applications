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
# ==============================================================================
"""
Local adapter.
"""

import os


def get_device_id():
    """
    Check the status of the corresponding device_id, the default value is 0.

    Returns:
        Int, device id.
    """
    device_id = os.getenv('DEVICE_ID', '0')
    return int(device_id)


def get_device_num():
    """
    Check the corresponding device_num situation, the default value is 1.

    Returns:
        Int, number of device.
    """
    device_num = os.getenv('RANK_SIZE', '1')
    return int(device_num)


def get_rank_id():
    """
    Check the status of the corresponding get_rank_id, the default value is 1.

    Returns:
        Int, rank id.
    """
    global_rank_id = os.getenv('RANK_ID', '0')
    return int(global_rank_id)


def get_job_id():
    """
    Returns a job that is determined to be local.

    Returns:
        Str, local job.
    """
    return "Local Job"

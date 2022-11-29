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
Save the information and data generated in the training process in real time.
"""

import os
from datetime import datetime

from mindvision import log

logger_name = 'mindspore-benchmark'


class LOGGER:
    """
    Define logging format and handler related information.
    """

    def __init__(self, log_dir):
        super().__init__()

        # If the specified folder does not exist, create it
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        # Log file name
        log_name = datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S')

        # Path to the log file
        log_fn = os.path.join(log_dir, log_name)
        if not os.path.exists(log_fn):
            os.mkdir(log_fn)

        # Controls how logs are output. When the value is 0, the log is output to a file.
        os.environ['GLOG_logtostderr'] = '0'

        # Specify the path of log output.
        os.environ['GLOG_log_dir'] = log_fn

        # GLOG_ V control the level of log, the default value is 2.
        # Log information greater than or equal to this level will be output.
        os.environ['GLOG_v'] = '1'

        # When the log is output to a file, it will also be printed to the screen.
        # Control the log level printed to the screen, default value is 2.
        os.environ['GLOG_stderrthreshold'] = '1'

    def info(self, msg):
        """Set log related information."""
        log.info(msg)

    def save_args(self, args):
        """Save the specified parameters to the log."""
        self.info('Args:')
        if isinstance(args, (list, tuple)):
            for value in args:
                message = '--> {}'.format(value)
                self.info(message)
        else:
            if isinstance(args, dict):
                args_dict = args
            else:
                args_dict = vars(args)
            for key in args_dict.keys():
                message = '--> {}: {}'.format(key, args_dict[key])
                self.info(message)
        self.info('')


def get_logger(path):
    """
    Get encapsulated logger object.

    Args:
        path (str): The specified path where the log file is located.

    Returns:
        Logger, log object.
    """
    logger = LOGGER(path)
    return logger

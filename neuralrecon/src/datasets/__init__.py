"""Init"""

import importlib


# find the dataset definition by name, for example ScanNetDataset (scannet.py)
def find_dataset_def(dataset_name):
    """Find dataset definition"""
    module_name = 'datasets.{}'.format(dataset_name)
    module = importlib.import_module(module_name)
    attr = None
    if dataset_name == 'scannet':
        attr = getattr(module, "ScanNetDataset")
    elif dataset_name == 'demo':
        attr = getattr(module, "DemoDataset")
    else:
        pass
    return attr

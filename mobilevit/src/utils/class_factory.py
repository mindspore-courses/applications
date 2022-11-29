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
""" Class Register Module for base architecture engine. """

import inspect


class ModuleType:
    """Module type for registry."""

    # dataset
    DATASET = 'dataset'
    DATASET_LOADER = 'dataset_loader'
    DATASET_SAMPLER = 'dataset_sampler'
    PIPELINE = 'pipeline'

    # models
    BACKBONE = 'backbone'
    CLASSIFIER = 'classifier'
    RECOGNIZER = 'recognizer'
    DETECTOR = 'detector'
    HEAD = 'head'
    NECK = 'neck'
    EMBEDDING = 'embedding'
    ANCHOR_GENERATOR = 'Anchor generator'
    DETECTION_ENGINE = 'detection_engine'
    GENERATOR = 'generator'
    DISCRIMINATOR = 'discriminator'

    # train
    LOSS = 'loss'
    LR_SCHEDULE = 'lr_schedule'
    OPTIMIZER = 'optimizer'

    # bbox
    BBOX_ASSIGNERS = 'bbox_assigner'
    BBOX_SAMPLERS = 'bbox_sampler'
    BBOX_CODERS = 'bbox_coder'

    GENERAL = 'general'
    WRAPPER = 'wrapper'


class ClassFactory:
    """Module class factory for builder."""

    registry = {}

    def __init__(self):
        pass

    @classmethod
    def register(cls, module_type=ModuleType.GENERAL, alias=None):
        """
        Register class into registry.

        Args:
            module_type (ModuleType): Module type name, default: ModuleType.GENERAL.
            alias (str) : class alias, default: None.

        Returns:
            Wrapper.

        Usage:
            @ClassFactory.register(ModuleType.BACKBONE)
        """

        def wrapper(register_class):
            """
            Register class with wrapper function.

            Args:
                register_class: Class which need to be register.

            Returns:
                Wrapper of register_class.
            """
            class_name = alias if alias is not None else register_class.__name__

            if module_type not in cls.registry:
                cls.registry[module_type] = {class_name: register_class}
            else:
                cls.registry[module_type][register_class.__name__] = register_class
            return register_class

        return wrapper

    @classmethod
    def register_cls(cls, register_class, module_type=ModuleType.GENERAL, alias=None):
        """
        Register class with type name into registry.

        Args:
            register_class: Class which need to be register.
            module_type(ModuleType): Module type name, default: ModuleType.GENERAL.
            alias(String): class name.

        Usage:
            ClassFactory.register_cls(dataset, ModuleType.DATASET)
        """
        class_name = alias if alias is not None else register_class.__name__

        if module_type not in cls.registry:
            cls.registry[module_type] = {class_name: register_class}
        else:
            cls.registry[module_type][register_class.__name__] = register_class

        return register_class

    @classmethod
    def is_exist(cls, module_type, class_name=None):
        """
        Determine whether class name is in the current type group.

        Args:
            module_type(ModuleType): Module type.
            class_name(string): Class name.

        Returns:
            Bool.
        """
        if not class_name:
            return module_type in cls.registry

        registered = module_type in cls.registry and class_name in cls.registry.get(module_type)
        return registered

    @classmethod
    def get_cls(cls, module_type, class_name=None):
        """
        Get class.

        Args:
            module_type(ModuleType): Module type.
            class_name(String): class name.

        Returns:
            register_class.
        """
        # verify
        if not cls.is_exist(module_type, class_name):
            raise ValueError(f"Can't find type `{module_type}` and name `{class_name}` in class registry.")
        if not class_name:
            raise ValueError(f"Can't find class which type is `{class_name}`.")

        register_class = cls.registry.get(module_type).get(class_name)
        return register_class

    @classmethod
    def get_instance_from_cfg(cls, cfg, module_type=ModuleType.GENERAL, default_args=None):
        """
        Get instance from configure.

        Args:
            cfg(dict): Config dict which should at least contain the key "type".
            module_type(ModuleType): module type.
            default_args(dict, optional) : Default initialization arguments.

        Returns:
            obj: The constructed object.
        """
        if not isinstance(cfg, dict):
            raise TypeError(f"`cfg` must be a Config, but got `{type(cfg)}`.")
        if 'type' not in cfg:
            raise KeyError(f"`cfg` or `default_args` must contain key type, but got:\n`{cfg}`.")
        if not (isinstance(default_args, dict) or not default_args):
            raise TypeError(f"`default_args` must be a dict or None but got:\n`{type(default_args)}`")

        args = cfg.copy()
        if default_args:
            for k, v in default_args.items():
                args.setdefault(k, v)

        obj_type = args.pop('type')
        if isinstance(obj_type, str):
            obj_cls = cls.get_cls(module_type, obj_type)
        elif inspect.isclass(obj_type):
            obj_cls = obj_type
        else:
            raise ValueError(f"Can't find class type {type} which name `{obj_type}` in class registry.")

        try:
            return obj_cls(**args)
        except Exception as e:
            raise type(e)(f"{obj_cls.__name__}: {e}")

    @classmethod
    def get_instance(cls, module_type=ModuleType.GENERAL, obj_type=None, args=None):
        """
        Get instance by ModuleType with object type.

        Args:
            module_type(ModuleType): Module type. Default: ModuleType.GENERAL.
            obj_type(String): Class type.
            args(dict): Object arguments.

        Returns:
            object: The constructed object.
        """
        if obj_type is None:
            raise ValueError("Class name cannot be None.")

        if isinstance(obj_type, str):
            obj_cls = cls.get_cls(module_type, obj_type)
        elif inspect.isclass(obj_type):
            obj_cls = obj_type
        else:
            raise ValueError(f"Can't find class type {type} which name `{obj_type}` in class registry.")

        try:
            return obj_cls(**args)
        except Exception as e:
            raise type(e)(f"{obj_cls.__name__}: {e}")

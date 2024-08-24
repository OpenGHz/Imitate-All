from dataclasses import dataclass
from typing import List
from importlib import import_module


@dataclass
class CommonConfig(object):
    module_path = None
    class_name = None
    instance_name = None
    default_act = None

    def __post_init__(self):
        assert self.default_act is not None, "default_act must be set"
        assert self.instance_name is not None, "instance_name must be set"
        assert self.module_path is not None, "class_path must be set"
        assert self.class_name is not None, "class_name must be set"


class Configer(object):
    @staticmethod
    def config2dict(config):
        return {key: value for key, value in config.__dict__.items()}

    @staticmethod
    def config2tuple(config):
        return tuple(config.__dict__.values())


class Creator(object):
    @staticmethod
    def instancer(configs: List[CommonConfig]) -> list:
        instances = []
        for config in configs:
            module = import_module(config.module_path)
            instances.append(getattr(module, config.class_name)(config))
        return instances

    @classmethod
    def give_eyes(cls, robot, configs):
        setattr(robot, "eyes", cls.instancer(configs))

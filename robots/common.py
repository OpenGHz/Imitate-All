"""A robot is a physical instance that has its proprioception state and can interact with the environment by subjective initiative actions. The robot's state is the information that can be obtained from the robot's body sensors while the actions are the control commands that can be sent to the robot's actuators. Vision, touch and the other external sensation (obtained by tactile sensors, cameras, lidar, radar, ultrasonic sensors, etc.) are not included in the robot's state, but in the environment. However, in addition to being related to the external environment, external observation also depends on the robot's state and the position and posture of the corresponding sensors. So the robot instance should have the full information and configurations of its external sensors to let the environment obtaining correct observations."""

from typing import Protocol, Dict
from dataclasses import dataclass, field
from le_studio.common.robot_devices.cameras.utils import Camera

# class Robot(Protocol):
#     """Assume the __init__ method of the robot class is the same as the reset method.
#     So you can inherit this class to save writing the initialization function."""

#     def __init__(self, config) -> None:
#         self.reset(config)

#     def reset(self, config):
#         raise NotImplementedError

#     def step(self, action):
#         raise NotImplementedError

#     def exit(self):
#         raise NotImplementedError

#     @property
#     def state(self):
#         raise NotImplementedError


class Robot(Protocol):
    def init_teleop(self): ...
    def run_calibration(self): ...
    def teleop_step(self, record_data=False): ...
    def enter_active_mode(self): ...
    def enter_passive_mode(self): ...
    def get_low_dim_data(self): ...
    def capture_observation(self): ...
    def send_action(self, action): ...
    def reset(self): ...
    def exit(self): ...
    def get_state_mode(self): ...


@dataclass
class FakeRobotConfig(object):
    cameras: Dict[str, Camera] = field(default_factory=lambda: {})


class FakeRobot(object):
    def __init__(self, config: FakeRobotConfig) -> None:
        self.config = config
        self.state = [0] * 6
        self.cameras = self.config.cameras

    def init_teleop(self):
        print("init_teleop")

    def run_calibration(self):
        print("run_calibration`")

    def teleop_step(self, record_data=False):
        print("teleop_step")

    def capture_observation(self):
        print("capture_observation")
        return self.state

    def send_action(self, action):
        print(f"send_action:{action}")
        self.state = action

    def exit(self):
        print("exit")


def make_robot(config) -> Robot:
    if isinstance(config, str):
        pass
    return Robot(config)


def make_robot_from_hydra_config(cfg) -> Robot:
    from omegaconf import DictConfig
    import hydra

    cfg: DictConfig
    robot = hydra.utils.instantiate(cfg)
    return robot


def make_robot_from_yaml(robot_path, robot_overrides) -> Robot:
    from le_studio.common.utils.utils import init_hydra_config

    robot_cfg = init_hydra_config(robot_path, robot_overrides)
    robot = make_robot_from_hydra_config(robot_cfg)

    return robot

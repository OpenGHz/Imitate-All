from habitats.common.creator import Configer
from habitats.fake.common import FakeCommon
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

try:
    import airbot
except ImportError:
    airbot = None
    logger.warning("Failed to import airbot python package, you can only use the fake robot.")

"""
This file contains the basic class for AIRBOTPlay robot, which is implemented in airbot python package and a fake class for testing.
"""

@dataclass
class AIRBOTPlayConfig(object):
    # init * 7
    model_path = "/usr/share/airbot_models/airbot_play_with_gripper.urdf"
    gravity_mode = "down"
    can_bus = "can0"
    vel = 2.0
    eef_mode = "none"
    bigarm_type = "OD"
    forearm_type = "DM"
    # other
    joint_vel = 6.0
    dt = 25
    # common
    module_path = logger.name
    class_name = "AIRBOTPlayPos"
    instance_name = "airbot_player"
    default_act = [0.0] * 7


class AIRBOTPlay(object):
    def __init__(self, config: AIRBOTPlayConfig) -> None:
        self.config = config
        self.robot = airbot.create_agent(*Configer.config2tuple(config)[:7])
        self._arm_joints_num = 6
        self._joints_num = 7

    def _set_eef(self, target, ctrl_type):
        if len(target) == 1:
            target = target[0]
        if ctrl_type == "pos":
            self.robot.set_target_end(target)
        elif ctrl_type == "vel":
            self.robot.set_target_end_v(target)
        elif ctrl_type == "eff":
            self.robot.set_target_end_t(target)
        else:
            raise ValueError(f"Invalid type: {ctrl_type}")

    def get_current_joint_positions(self):
        joints = self.robot.get_current_joint_q()
        if self.config.eef_mode in ["gripper"]:
            joints += [self.robot.get_current_end()]
        return joints

    def get_current_joint_velocities(self):
        joints = self.robot.get_current_joint_v()
        if self.config.eef_mode in ["gripper"]:
            joints += [self.robot.get_current_end_v()]
        return joints

    def get_current_joint_efforts(self):
        joints = self.robot.get_current_joint_t()
        if self.config.eef_mode in ["gripper"]:
            joints += [self.robot.get_current_end_t()]
        return joints

    def set_joint_position_target(self, qpos, qvel=None, blocking=False):
        if qvel is None:
            qvel = self.config.joint_vel
        use_planning = blocking
        self.robot.set_target_joint_q(
            qpos[: self._arm_joints_num], use_planning, qvel[0], blocking
        )
        if len(qpos) - self._arm_joints_num > 0:
            self._set_eef(qpos[self._arm_joints_num :], "pos")

    def set_joint_velocity_target(self, qvel, blocking=False):
        self.robot.set_target_joint_v(qvel[: self._arm_joints_num])
        if len(qvel) - self._arm_joints_num > 0:
            self._set_eef(qvel[self._arm_joints_num :], "vel")

    def set_joint_effort_target(self, qeffort, blocking=False):
        self.robot.set_target_joint_t(qeffort[: self._arm_joints_num])
        if len(qeffort) - self._arm_joints_num > 0:
            self._set_eef(qeffort[self._arm_joints_num :], "eff")


class AIRBOTPlayPos(AIRBOTPlay):
    def __init__(self, config: AIRBOTPlayConfig) -> None:
        super().__init__(config)
        # TODO: 应该在创建policy的后处理中进行动作的拆分，在预处理中进行状态的拼接，不需强制要求定义state_dim和action_dim

    def reset(self):
        self.set_joint_position_target(self.config.default_act)

    def act(self, action):
        self.set_joint_position_target(action)

    @property
    def state(self):
        return self.get_current_joint_positions()


class AIRBOTPlayVel(AIRBOTPlay):
    def __init__(self, config: AIRBOTPlayConfig) -> None:
        super().__init__(config)

    def reset(self):
        self.set_joint_velocity_target(self.config.default_act)

    def act(self, action):
        self.set_joint_velocity_target(action)

    @property
    def state(self):
        return self.get_current_joint_velocities()


class AIRBOTPlayMIT(AIRBOTPlay):
    def __init__(self, config: AIRBOTPlayConfig) -> None:
        super().__init__(config)

    def reset(self):
        self.act(self.config.default_act)

    def act(self, action):
        self.set_joint_mit_target(action)

    @property
    def state(self):
        return self.get_current_joint_positions() + self.get_current_joint_velocities()


class AIRBOTPlayPosFake(FakeCommon):
    """A fake robot for AIRBOTPlayPos."""


def make_robot(robot_config, robot_type):
    """A factory function to create a robot instance based on the robot config and type."""
    if robot_type == "pos":
        return AIRBOTPlayPos(robot_config)
    elif robot_type == "vel":
        return AIRBOTPlayVel(robot_config)
    elif robot_type == "mit":
        return AIRBOTPlayMIT(robot_config)
    elif robot_type == "fake":
        return AIRBOTPlayPosFake(robot_config)
    else:
        raise ValueError(f"Invalid robot type: {robot_type}")

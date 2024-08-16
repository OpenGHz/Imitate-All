import airbot
from robots.common_robot import Configer


class AIRBOTPlayConfig(object):
    def __init__(self) -> None:
        # init * 7
        self.model_path = "/usr/share/airbot_models/airbot_play_with_gripper.urdf"
        self.gravity_mode = "down"
        self.can_bus = "can0"
        self.vel = 2.0
        self.eef_mode = "none"
        self.bigarm_type = "OD"
        self.forearm_type = "DM"
        # other
        self.joint_vel = 6.0


class AIRBOTPlay(object):
    def __init__(self, config: AIRBOTPlayConfig) -> None:
        self.config = config
        self.robot = airbot.create_agent(*Configer.config2tuple(config)[:7])
        self._arm_joints_num = 6
        self._joints_num = 7
        self.end_effector_open = 1
        self.end_effector_close = 0

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

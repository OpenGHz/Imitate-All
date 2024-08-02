import time
import numpy as np
import collections
import dm_env
from typing import List
from robots.custom_robot import AssembledMmkRobot as AssembledRobot


class AirbotMmkEnv:
    """
    Environment for real robot one-manual manipulation
    Action space:      [arm_qpos (6),             # absolute joint position
                        gripper_positions (1),]    # normalized gripper position (0: close, 1: open)

    Observation space: {"qpos": Concat[ arm_qpos (6),          # absolute joint position
                                        gripper_position (1),]  # normalized gripper position (0: close, 1: open)
                        "qvel": Concat[ arm_qvel (6),         # absolute joint velocity (rad)
                                        gripper_velocity (1),]  # normalized gripper velocity (pos: opening, neg: closing)
                        "images": {"0": (480x640x3),        # h, w, c, dtype='uint8'
                                   "1": (480x640x3),         # h, w, c, dtype='uint8'
                                   "2": (480x640x3),}  # h, w, c, dtype='uint8'
    """

    def __init__(
        self,
        record_images=True,
        robot_instances: List[AssembledRobot] = None,
        cameras=None,
    ):
        self.airbot_players = robot_instances
        self.robot_num = len(self.airbot_players)

        if record_images:
            from robot_tools.recorder import ImageRecorderRos as ImageRecorder
            import rospy

            if rospy.get_name() == "/unnamed":
                rospy.init_node("real_env", anonymous=True)
            name_converter = {
                "0": "camera_head",
                "1": "camera_left",
                "2": "camera_right",
            }
            for i in range(cameras):
                cameras[i] = name_converter[cameras[i]]
            self.image_recorder = ImageRecorder(cameras)
        self.reset_position = None
        self.eefs_open = np.array(
            [robot.end_effector_open for robot in self.airbot_players]
        )
        self.eefs_close = np.array(
            [robot.end_effector_close for robot in self.airbot_players]
        )
        self.all_joints_num = 7
        print("AIRBOT MMK Real Env Created.")

    def set_reset_position(self, reset_position):
        self.reset_position = reset_position

    def _get_images(self):
        return self.image_recorder.get_images()

    def get_reward(self):
        return 0

    def reset(self, fake=False, sleep_time=0):
        assert self.reset_position is not None
        for index, robot in enumerate(self.airbot_players):
            robot.set_joint_position_target(self.reset_position[index], [0], True)

        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=self.get_reward(),
            discount=None,
            observation=self._get_observation(),
        )

    def step(
        self,
        action,
        sleep_time=0,
        arm_vel=0,
    ):
        for index, robot in enumerate(self.airbot_players):
            jn = robot.all_joints_num
            robot.set_joint_position_target(
                action[jn * index : jn * (index + 1)],
                [arm_vel],
                False,
            )
        time.sleep(sleep_time)
        obs = self._get_observation()
        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID,
            reward=self.get_reward(),
            discount=None,
            observation=obs,
        )

    def _get_qpos(self):
        """17 dof: (6 arm joints + 1 gripper joint) * 2 + 2 head joints + 1 spine joint"""
        qpos = []
        for airbot in self.airbot_players:
            qpos.append(airbot.get_current_joint_positions())
        return np.hstack(qpos)

    def _get_observation(self):
        obs = collections.OrderedDict()
        obs["qpos"] = self._get_qpos()
        obs["images"] = self._get_images()
        return obs

def make_env(
    record_images=True,
    robot_instance=None,
    cameras=None,
):
    env = AirbotMmkEnv(record_images, robot_instance, cameras)
    return env

import time
import numpy as np
import collections
import dm_env
from typing import List
from robots.ros_robots.ros1_robot import AssembledROS1Robot as MMK2Robot
from robot_tools.recorder import ImageRecorderRos
import rospy


class AirbotMmkEnv:
    """
    Environment of ROS1 images topic.
    """

    def __init__(
        self,
        robot_instances: List[MMK2Robot] = None,
        camera_names=None,
    ):
        self.mmk2_robots = robot_instances
        self.robot_num = len(self.mmk2_robots)

        if rospy.get_name() == "/unnamed":
            rospy.init_node("real_env", anonymous=True)
        name_converter = {
            "0": "camera_head",
            "1": "camera_left",
            "2": "camera_right",
        }
        self.name_converter_rev = {v: k for k, v in name_converter.items()}
        for i in range(camera_names):
            camera_names[i] = name_converter[camera_names[i]]
        self.image_recorder = ImageRecorderRos(camera_names)
        self.all_joints_num = 7 * 2 + 2 + 1
        print("AIRBOT MMK Real Environment Created.")

    def _get_images(self):
        images_dic = self.image_recorder.get_images()
        images = {}
        for key, value in images_dic.items():
            images[self.name_converter_rev[key]] = value
        return images

    def get_reward(self):
        return 0

    def reset(self, sleep_time=0):
        for robot in self.mmk2_robots:
            robot.reset()

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
        for index, robot in enumerate(self.mmk2_robots):
            jn = robot.all_joints_num
            robot.set_target_states(
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

    def _get_state(self):
        """17 dof: (6 arm joints + 1 gripper joint) * 2 + 2 head joints + 1 spine joint"""
        qpos = []
        for robot in self.mmk2_robots:
            qpos.append(robot.get_current_states())
        return np.hstack(qpos)

    def _get_observation(self):
        obs = collections.OrderedDict()
        obs["qpos"] = self._get_state()
        obs["images"] = self._get_images()
        return obs

def make_env(
    robot_instance=None,
    cameras=None,
):
    env = AirbotMmkEnv(robot_instance, cameras)
    return env

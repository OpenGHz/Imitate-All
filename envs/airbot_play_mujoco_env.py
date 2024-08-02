import time
import numpy as np
import collections
import dm_env
from typing import List
from custom_robot import AssembledRobot
from dlabsim.envs.airbot_play_base import AirbotPlayCfg, AirbotPlayBase


class MujocoEnv:
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
        setup_robots=True,
        setup_base=False,
        record_images=True,
        robot_instances: List[AssembledRobot] = None,
        cameras=None,
    ):
        self.exec_node = AirbotPlayBase(AirbotPlayCfg())
        self.exec_node.cam_id = self.exec_node.config.obs_camera_id
        self.reset_position = None

    def set_reset_position(self, reset_position):
        self.reset_position = reset_position
        print("Resetting to the given position: ", self.reset_position)

    def get_qpos(self, normalize_gripper=False):
        """7 dof: 6 arm joints + 1 gripper joint"""
        qpos = []
        for airbot in self.airbot_players:
            qpos.append(airbot.get_current_joint_positions())
        return np.hstack(qpos)

    def get_qvel(self):
        qvel = []
        for airbot in self.airbot_players:
            qvel.append(airbot.get_current_joint_velocities())
        return np.hstack(qvel)

    def get_effort(self):
        effort = []
        for airbot in self.airbot_players:
            effort.append(airbot.get_current_joint_efforts())
        return np.hstack(effort)

    def get_images(self):
        return self.image_recorder.get_images()

    def get_base_vel(self):
        raise NotImplementedError
        vel, right_vel = 0.1, 0.1
        right_vel = -right_vel  # right wheel is inverted
        base_linear_vel = (vel + right_vel) * self.wheel_r / 2
        base_angular_vel = (right_vel - vel) * self.wheel_r / self.base_r

        return np.array([base_linear_vel, base_angular_vel])

    def get_tracer_vel(self):
        raise NotImplementedError
        linear_vel, angular_vel = 0.1, 0.1
        return np.array([linear_vel, angular_vel])

    def get_reward(self):
        return 0

    def reset(self, fake=False, sleep_time=0):
        raw_obs = self.exec_node.reset()
        # if self.reset_position is not None:
        #     # self.reset_position[-1] = 0.96
        #     # print("Resetting to the given position: ", self.reset_position)
        #     self.reset_position[-1] = 0.04  # undo the normalization
        #     self.exec_node.mj_data.ctrl[:7] = self.reset_position
        #     self.exec_node.mj_data.ctrl[7] = -self.exec_node.mj_data.ctrl[6]
        #     self.exec_node.mj_data.qpos[:7] = self.reset_position
        #     self.exec_node.mj_data.qpos[7] = -self.exec_node.mj_data.qpos[6]
        #     raw_obs, pri_obs, rew, ter, info = self.exec_node.step(self.reset_position)
        time.sleep(sleep_time)
        obs = collections.OrderedDict()
        obs["qpos"] = list(raw_obs["jq"])
        print("obs gripper", raw_obs["jq"][-1])
        # print("pre_obs", obs["qpos"])
        # obs["qpos"][-1] *= 25  # undo the normalization
        # print("post_obs", obs["qpos"])
        # obs["qvel"] = raw_obs["jv"]
        obs["images"] = {}
        obs["images"]["0"] = raw_obs["img"]
        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=self.get_reward(),
            discount=None,
            observation=obs,
        )

    def step(
        self,
        action,
        base_action=None,
        get_tracer_vel=False,
        get_obs=True,
        sleep_time=0,
        arm_vel=0,
    ):
        all_joints_num = (7, )
        eef_joints_num = (1, )
        # print("action", action)
        for index, jn in enumerate(all_joints_num):
            one_action = action[jn * index : jn * (index + 1)]
            # if one_action[-1] < 0:
            #     one_action[-1] = 0
            # elif one_action[-1] > 1:
            #     one_action[-1] = 1
            # one_action[-1] *= 0.04  # undo the normalization
            # if one_action[-1] < 0.02:  # TODO: remove this
            #     one_action[-1] = 0
            print("one_action gripper", one_action[-1])
            # one_action[-1] = 0
            raw_obs, pri_obs, rew, ter, info = self.exec_node.step(one_action)
            print("obs gripper", raw_obs["jq"][-1])
        time.sleep(sleep_time)

        if get_obs:
            obs = collections.OrderedDict()
            obs["qpos"] = list(raw_obs["jq"])
            # obs["qpos"][-1] *= 25
            # if obs["qpos"][-1] < 0:
            #     obs["qpos"][-1] = 0
            # elif obs["qpos"][-1] > 1:
            #     obs["qpos"][-1] = 1
            # obs["qvel"] = raw_obs["jv"]
            obs["images"] = {}
            obs["images"]["0"] = raw_obs["img"]
        else:
            obs = None
        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID,
            reward=self.get_reward(),
            discount=None,
            observation=obs,
        )


def make_env(
    setup_robots=True,
    setup_base=False,
    record_images=True,
    robot_instance=None,
    cameras=None,
):
    env = MujocoEnv(setup_robots, setup_base, record_images, robot_instance, cameras)
    return env
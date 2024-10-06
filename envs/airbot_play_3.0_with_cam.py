import time
import collections
import dm_env
from typing import List
from robots.airbots.airbot_play.airbot_play_3 import AIRBOTPlay
from robots.common import make_robot_from_yaml


class AIRBOTPlayWithCameraEnv(object):

    def __init__(
        self,
        config_paths: List[str],
    ):
        self.robots: List[AIRBOTPlay] = []
        for cfg in config_paths:
            robot = make_robot_from_yaml(cfg)
            self.robots.append(robot)

    def set_reset_position(self, reset_position):
        self.reset_position = reset_position

    def get_reward(self):
        return 0

    def _get_obs(self):
        q_pos = []
        images = []
        images_cnt = 0
        for robot in self.robots:
            raw_obs = robot.capture_observation()
            low_dim = raw_obs["low_dim"]
            q_pos.append(
                low_dim["observation/arm/joint_position"]
                + low_dim["observation/eef/joint_position"]
            )
            for name in robot.cameras:
                images.append(raw_obs[f"observation.images.{name}"])
                images_cnt += 1
            images.append(images)
        obs = collections.OrderedDict()
        obs["qpos"] = q_pos
        obs["images"] = {}
        for i in range(images_cnt):
            obs["images"][str(i)] = images[i]
        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=self.get_reward(),
            discount=None,
            observation=obs,
        )

    def reset(self, fake=False, sleep_time=0):
        for robot in self.robots:
            robot.reset()
        time.sleep(sleep_time)
        return self._get_obs()

    def step(
        self,
        action,
        get_obs=True,
        sleep_time=0,
        arm_vel=0,
    ):
        all_joints_num = (7,) * len(self.robots)
        # print("action", action)
        for index, jn in enumerate(all_joints_num):
            one_action = action[jn * index : jn * (index + 1)]
            self.robots[index].send_action(one_action)
        time.sleep(sleep_time)
        obs = self._get_obs() if get_obs else None
        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID,
            reward=self.get_reward(),
            discount=None,
            observation=obs,
        )

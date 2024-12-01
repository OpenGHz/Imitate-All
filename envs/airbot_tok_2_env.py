from robots.airbots.airbot_tok.airbot_tok_2 import AIRBOTTOK
from robots.common import make_robot_from_yaml
import time
import collections
import dm_env


class AIRBOTTOKEnv(object):
    def __init__(self, config_path: str):
        self.robot: AIRBOTTOK = make_robot_from_yaml(config_path)
        assert isinstance(self.robot, AIRBOTTOK)
        self._all_joints_num = 14

    def set_reset_position(self, reset_position):
        assert (
            len(reset_position) == self._all_joints_num
        ), f"Expected {self._all_joints_num} joints, got {len(reset_position)}"
        for i, arm in enumerate(self.robot.arms.values()):
            arm.config.default_action = reset_position[i * 7 : (i + 1) * 7]

    def reset(self, sleep_time=0):
        self.robot.reset()
        time.sleep(sleep_time)
        return self._get_obs()

    def _get_obs(self):
        obs = collections.OrderedDict()
        obs["qpos"] = []
        obs["images"] = {}
        raw_obs = self.robot.capture_observation()
        low_dim = raw_obs["low_dim"]
        for arm_name in self.robot.arms:
            obs["qpos"].extend(low_dim[f"observation/{arm_name}/joint_position"])
        for name in self.robot.cameras:
            assert name not in obs["images"], f"Duplicate camera name: {name}"
            obs["images"][name] = raw_obs[f"observation.images.{name}"]

        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=0,
            discount=None,
            observation=obs,
        )

    def step(
        self,
        action,
        sleep_time=0,
        get_obs=True,
    ):
        self.robot.send_action(action)
        time.sleep(sleep_time)
        obs = self._get_obs() if get_obs else None
        return obs

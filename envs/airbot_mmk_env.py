import time
import dm_env
import numpy as np
from robots.common import make_robot_from_yaml
from airbot_ie.robots.airbot_mmk import AIRBOTMMK, SystemMode
from collections import defaultdict


class AIRBOTMMK2Env:
    def __init__(self, config_path: str):
        self.robot: AIRBOTMMK = make_robot_from_yaml(config_path)
        assert isinstance(self.robot, AIRBOTMMK)
        assert self.robot.configure(), "Failed to configure robot"
        self.time_metrics = defaultdict(list)

    def set_reset_position(self, reset_position):
        self._reset_position = reset_position

    def reset(self, sleep_time=0):
        self.robot.switch_mode(SystemMode.RESETTING)
        self.robot.send_action(self._reset_position)
        time.sleep(sleep_time)
        self.robot.switch_mode(SystemMode.SAMPLING)
        self.time_metrics.clear()
        return self._get_obs()

    def _get_obs(self):
        obs = {}
        obs["qpos"] = []
        obs["images"] = {}
        raw_obs = self.robot.capture_observation()
        for comp in self.robot.config.components:
            obs["qpos"].extend(
                raw_obs[f"observation/{comp.value}/joint_state/position"]["data"]
            )
        for camera in self.robot.config.cameras:
            image = raw_obs[f"{camera.value}/color/image_raw"]["data"]
            obs["images"][camera.value] = image
        # print(obs["images"].keys())
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
        # TODO: require the arms to be first components
        joint_limits = (
            (-3.151, 2.09),  # (-3.09, 2.04)
            (-2.963, 0.181),  # (-2.92, 0.12)
            (-0.094, 3.161),  # (-0.04, 3.09)
            (-2.95, 2.95),  # (-3.012, 3.012)
            (-1.9, 1.9),  # (-1.859, 1.859)
            (-2.90, 2.90),  # (-3.017, 3.017)
            (0, 1),
        )

        jn = len(joint_limits)
        for i in range(2):
            for j in range(jn):
                index = j + jn * i
                action[index] = np.clip(action[index], *joint_limits[j])
        # print("sending action:", action)
        start = time.perf_counter()
        self.robot.send_action(action)
        # print(f"send action time: {time.perf_counter() - start:.4f}s")
        self.time_metrics["send_action"].append(time.perf_counter() - start)
        if sleep_time > 0:
            time.sleep(sleep_time)
        start = time.perf_counter()
        obs = self._get_obs() if get_obs else None
        # print(f"get obs time: {time.perf_counter() - start:.4f}s")
        self.time_metrics["get_obs"].append(time.perf_counter() - start)
        return obs


def make_env(config):
    env = AIRBOTMMK2Env(config["env_config_path"])
    env.set_reset_position(config["start_joint"])

    return env

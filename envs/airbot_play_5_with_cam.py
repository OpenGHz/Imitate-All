import collections
import time
from logging import getLogger
from pprint import pformat
from typing import Dict, List, Union

import dm_env
import numpy as np
from airbot_data_collection.basis import System, SystemMode


class AIRBOTPlayWithCameraEnv:

    def __init__(self, config: Dict[str, Union[dict, System]]):
        self.cameras: Dict[str, System] = {}
        self.arms: Dict[str, System] = {}
        for name, cam in config.pop("cameras").items():
            self.cameras[name] = cam
        for name, arm in config.items():
            self.arms[name] = arm
        print(f"robot groups: {self.arms.keys()}")
        print(f"camera names: {self.cameras.keys()}")
        self._reset_actions = {}
        self._all_ins = list(self.arms.items()) + list(self.cameras.items())
        for name, ins in self._all_ins:
            if not ins.configure():
                raise RuntimeError(f"Failed to configure {name}.")

    def set_reset_position(self, action: np.ndarray):
        for index, key in enumerate(self.arms.keys()):
            self._reset_actions[key] = action[index * 7 : (index + 1) * 7].tolist()
        self.get_logger().info(f"Set reset actions: {pformat(self._reset_actions)}")

    def _capture_observation(self) -> dict:
        """Capture the current observation from the robot."""
        obs = {}
        for name, ins in self._all_ins:
            for key, value in ins.capture_observation().items():
                obs[f"{name}/{key}"] = value
        return obs

    def _get_qpos(self, obs: dict) -> List[float]:
        """Get the joint positions of the robot."""
        qpos = []
        for group in self.arms:
            qpos.extend(obs[f"{group}/arm/joint_state"]["data"]["position"])
            qpos.extend(obs[f"{group}/eef/joint_state"]["data"]["position"])
        return qpos

    def _get_images(self, obs: dict) -> Dict[str, np.ndarray]:
        images = {}
        for name in self.cameras:
            images[name] = obs[f"{name}/color/image_raw"]["data"]
        return images

    def _get_obs(self):
        obs = collections.OrderedDict()
        cap_obs = self._capture_observation()
        # print(cap_obs.keys())
        obs["qpos"] = self._get_qpos(cap_obs)
        obs["images"] = self._get_images(cap_obs)

        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=0,
            discount=None,
            observation=obs,
        )

    def reset(self, sleep_time: float = 0):
        for name, robot in self.arms.items():
            robot.switch_mode(SystemMode.RESETTING)
            robot.send_action(self._reset_actions[name])
            robot.switch_mode(SystemMode.SAMPLING)
        time.sleep(sleep_time)
        return self._get_obs()

    def step(
        self,
        action: np.ndarray,
        sleep_time=0,
        get_obs=True,
    ):
        for index, (_key, robot) in enumerate(self.arms.items()):
            robot.send_action(action[index * 7 : (index + 1) * 7].tolist())
        time.sleep(sleep_time)
        obs = self._get_obs() if get_obs else None
        return obs

    def get_logger(self):
        return getLogger(self.__class__.__name__)

    def shutdown(self) -> bool:
        """Shutdown the robot."""
        for robot in self.arms.values():
            robot.shutdown()
        for camera in self.cameras.values():
            camera.shutdown()
        return True

    def __del__(self):
        self.shutdown()

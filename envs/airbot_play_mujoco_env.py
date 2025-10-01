import collections
import importlib
import time
import dm_env
from discoverse.robots_env.airbot_play_base import AirbotPlayCfg
from discoverse.task_base import AirbotPlayTaskBase


class MujocoEnv(object):
    """
    Mujoco environment for airbot_play
    path: path to the script containing the SimNode class and config (mainly for data collection)
    """

    def __init__(self, path: str):
        module = importlib.import_module(path.replace("/", ".").replace(".py", ""))
        node_cls = getattr(module, "SimNode")
        cfg: AirbotPlayCfg = getattr(module, "cfg")
        cfg.headless = False
        self.exec_node: AirbotPlayTaskBase = node_cls(cfg)
        # self.exec_node.cam_id = self.exec_node.config.obs_camera_id
        self.reset_position = None
        self._camera_names = self.exec_node.config.obs_rgb_cam_id
        print("MujocoEnv initialized")

    def set_reset_position(self, reset_position):
        self.reset_position = reset_position
        print("Resetting to the given position: ", self.reset_position)

    def get_reward(self):
        return 0

    def reset(self, fake=False, sleep_time=0):
        self.exec_node.domain_randomization()
        raw_obs = self.exec_node.reset()
        time.sleep(sleep_time)
        return self._process_obs(raw_obs, self._camera_names)

    @staticmethod
    def _process_obs(raw_obs, camera_names: list):
        obs = collections.OrderedDict()
        obs["qpos"] = list(raw_obs["jq"])
        obs["images"] = {}
        for id in camera_names:
            obs["images"][f"cam_{id}"] = raw_obs["img"][id][:, :, ::-1]
        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=0.0,
            discount=None,
            observation=obs,
        )

    def step(
        self,
        action,
        get_obs=True,
        sleep_time=0,
    ):
        raw_obs, pri_obs, rew, ter, info = self.exec_node.step(action)
        time.sleep(sleep_time)
        return self._process_obs(raw_obs, self._camera_names)


def make_env(path):
    env = MujocoEnv(path)
    return env

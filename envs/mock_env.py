import collections
import time
import dm_env


class MockEnv:
    """
    Mujoco environment for airbot_play
    path: path to the script containing the SimNode class and config (mainly for data collection)
    """

    def __init__(self, path: str):
        pass

    def set_reset_position(self, reset_position):
        self.reset_position = reset_position
        print("Resetting to the given position: ", self.reset_position)

    def get_reward(self):
        return 0

    def reset(self, fake=False, sleep_time=0):
        time.sleep(sleep_time)
        obs = collections.OrderedDict()
        obs["qpos"] = []
        # print("obs gripper", raw_obs["jq"][-1])
        # print("pre_obs", obs["qpos"])
        # obs["qpos"][-1] *= 25  # undo the normalization
        # print("post_obs", obs["qpos"])
        # obs["qvel"] = raw_obs["jv"]
        obs["images"] = {}
        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=self.get_reward(),
            discount=None,
            observation=obs,
        )

    def step(
        self,
        action,
        get_obs=True,
        sleep_time=0,
    ):
        time.sleep(sleep_time)

        if get_obs:
            obs = collections.OrderedDict()
            obs["qpos"] = []
            obs["images"] = {}
        else:
            obs = None
        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID,
            reward=self.get_reward(),
            discount=None,
            observation=obs,
        )


def make_env(path):
    env = MockEnv(path)
    return env

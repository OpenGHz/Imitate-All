from dataclasses import dataclass, field, replace
from habitats.common.robot_devices.cameras.utils import Camera
from typing import Dict, Optional, List, Union
from robots.airbots.airbot_play.airbot_play_4 import AIRBOTPlay, AIRBOTPlayConfig
from robots.airbots.airbot_play.airbot_replay_remote2 import (
    AIRBOTReplay,
    AIRBOTReplayConfig,
)
import time
import numpy as np
import logging
from collections import defaultdict


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AIRBOTPlayDemonstrationConfig(object):
    groups: Dict[str, Dict[str, Union[AIRBOTPlayConfig, AIRBOTReplayConfig]]] = field(
        default_factory=dict
    )
    cameras: Dict[str, Camera] = field(default_factory=dict)


class AIRBOTPlayDemonstration(object):
    def __init__(
        self, config: Optional[AIRBOTPlayDemonstrationConfig] = None, **kwargs
    ):
        if config is None:
            config = AIRBOTPlayDemonstrationConfig()
        # Overwrite config arguments using kwargs (used for yaml config)
        self.config = replace(config, **kwargs)
        self.cameras = self.config.cameras
        self.logs = {}
        self._exited = False
        self.leaders: Dict[str, Union[AIRBOTPlay, AIRBOTReplay]] = {}
        self.followers: Dict[str, List[AIRBOTPlay]] = {}

        self.is_replay = {}
        for g_name, g_value in self.config.groups.items():
            leader_cfg: dict = g_value["leader"]
            followers_cfg = g_value["followers"]
            if leader_cfg.get("joint_states_topic", None) is None:
                logger.info("Using local robot")
                leader = AIRBOTPlay(**leader_cfg)
                self.is_replay[g_name] = leader.robot.arm_type == "replay"
                self.leaders[g_name] = leader
            else:
                logger.info("Using remote robot")
                self.leaders[g_name] = AIRBOTReplay(**leader_cfg)
                self.is_replay[g_name] = True
            self.followers[g_name] = []
            for f_cfg in followers_cfg:
                self.followers[g_name].append(AIRBOTPlay(**f_cfg))
        for name in self.cameras:
            self.cameras[name].connect()
        self.reset()

    def reset(self):
        leaders = list(self.leaders.values())
        # move leaders to default action
        # the followers will be automatically moved
        for index, leader in enumerate(leaders):
            default_action = leader.config.default_action
            if default_action:
                # if leader.enter_active_mode():
                leader.send_action(default_action, True)
        # is_replay = list(self.is_replay.values())
        # for index, followers in enumerate(self.followers.values()):
        #     default_action = leaders[index].config.default_action
        #     # do not reset the follower when the leader is a replay robot
        #     if (default_action is not None) and not is_replay[index]:
        #         for follower in followers:
        #             # follower.enter_active_mode()
        #             follower.send_action(default_action, True)
        self._state_mode = "active"

    def enter_active_mode(self):
        # for leader in self.leaders.values():
        #     leader.enter_active_mode()
        self._state_mode = "active"

    def enter_passive_mode(self):
        for leader in self.leaders.values():
            leader.enter_passive_mode()
        self._state_mode = "passive"

    def get_low_dim_data(self) -> Dict[str, list]:
        data = defaultdict(list)
        data["/time"] = time.time()
        for leader in self.leaders.values():
            low_dim = leader.get_low_dim_data()
            low_dim.pop("/time")
            for key, value in low_dim.items():
                data[key.replace("observation", "action")].extend(value)
        for followers in self.followers.values():
            for follower in followers:
                low_dim = follower.get_low_dim_data()
                low_dim.pop("/time")
                for key, value in low_dim.items():
                    data[key].extend(value)
        return data

    def capture_observation(self) -> Dict[str, Union[dict, np.ndarray]]:
        """The returned observations do not have a batch dimension."""
        obs_act_dict = {}
        # Capture images from cameras
        images = {}
        depths = {}
        for name in self.cameras:
            before_camread_t = time.perf_counter()
            cam_data = self.cameras[name].async_read()
            if len(cam_data) == 2:
                images[name], depths[name] = cam_data
            else:
                images[name] = cam_data
            # images[name] = torch.from_numpy(images[name])
            obs_act_dict[f"/time/{name}"] = time.time()
            self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs[
                "delta_timestamp_s"
            ]
            self.logs[f"async_read_camera_{name}_dt_s"] = (
                time.perf_counter() - before_camread_t
            )

        obs_act_dict["low_dim"] = self.get_low_dim_data()
        for name in images:
            obs_act_dict[f"observation.images.{name}"] = images[name]
        for name in depths:
            obs_act_dict[f"observation.depths.{name}"] = depths[name]
        return obs_act_dict

    def exit(self):
        assert not self._exited, "Robot already exited"
        for name in self.cameras:
            self.cameras[name].disconnect()
        self._exited = True
        print("Robot exited")

    def get_state_mode(self):
        return self._state_mode

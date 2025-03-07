from dataclasses import dataclass, field, replace
from robots.airbots.airbot_play.airbot_play_4_demonstration import (
    AIRBOTPlayDemonstrationConfig,
    AIRBOTPlayDemonstration,
)
from robots.airbots.airbot_base.airbot_base import AIRBOTBase, AIRBOTBaseConfig
from robots.airbots.airbot_base.cmd_vel_remote import CmdVelRemote
from typing import Optional
from robot_utils import ping_ip
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AIRBOTTOKDemonstrationConfig(object):
    airbot_play_demonstration: AIRBOTPlayDemonstrationConfig = field(
        default_factory=AIRBOTPlayDemonstrationConfig
    )
    base: AIRBOTBaseConfig = field(default_factory=AIRBOTBaseConfig)
    base_action: bool = False


class AIRBOTTOKDemonstration(object):
    def __init__(
        self, config: Optional[AIRBOTTOKDemonstrationConfig] = None, **kwargs
    ) -> None:
        if config is None:
            config = AIRBOTTOKDemonstrationConfig()
        self.config = replace(config, **kwargs)
        logger.info("Connecting the base")
        self.airbot_base = None
        self.base_action = None
        if self.config.base is not None:
            if ping_ip(self.config.base.ip):
                self.airbot_base = AIRBOTBase(self.config.base)
                if self.config.base_action:
                    self.base_action = CmdVelRemote()
            else:
                logger.warning("Base IP is not reachable and will not be connected")
        self.airbot_play_demon = AIRBOTPlayDemonstration(
            self.config.airbot_play_demonstration
        )
        self.logs = {}
        self.cameras = self.airbot_play_demon.cameras
        print(self.config)

    def reset(self):
        self.airbot_play_demon.reset()

    def enter_active_mode(self):
        self.airbot_play_demon.enter_active_mode()

    def enter_passive_mode(self):
        self.airbot_play_demon.enter_passive_mode()

    def _get_base_data(self):
        data = {}
        if self.base_action is not None:
            data["observation/base/velocity"] = list(
                self.airbot_base.get_current_velocity2D()
            )
            data["action/base/velocity"] = list(
                self.base_action.get_action_velocity2D()
            )
        elif self.airbot_base is not None:
            data["action/base/velocity"] = self.airbot_base.get_current_velocity2D()
        return data

    def get_low_dim_data(self):
        data = self.airbot_play_demon.get_low_dim_data()
        data.update(self._get_base_data())
        return data

    def capture_observation(self):
        """The returned observations do not have a batch dimension."""
        obs_act_dict = self.airbot_play_demon.capture_observation()
        obs_act_dict["low_dim"].update(self._get_base_data())
        return obs_act_dict

    def exit(self):
        return self.airbot_play_demon.exit()

    def get_state_mode(self):
        return self.airbot_play_demon.get_state_mode()

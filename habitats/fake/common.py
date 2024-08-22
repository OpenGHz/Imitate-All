from typing import Optional
import logging
from habitats.common.creator import CommonConfig

logger = logging.getLogger(__name__)


class FakeCommon(object):
    def __init__(self, config: CommonConfig) -> None:
        self.reset(config)

    def reset(self, config: Optional[CommonConfig] = None):
        if config is not None:
            self.config = config
        self._state = self.config.default_act
        logger.debug(f"Reset {self.config.instance_name}: {self._state}")

    def step(self, action):
        self._state = action
        logger.debug(f"Step {self.config.instance_name}: {self._state}")

    @property
    def state(self):
        logger.debug(f"Get {self.config.instance_name} state: {self._state}")
        return self._state

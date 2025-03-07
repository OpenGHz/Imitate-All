try:
    import rclpy
    from sensor_msgs.msg import JointState
except Exception as e:
    print(e)
    print(
        "ROS2 not installed. This is expected if you are running this code on your local machine."
    )
    print("This code is meant to be run on the robot.")

from typing import Optional, List
from dataclasses import dataclass, replace
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AIRBOTReplayConfig(object):
    joint_states_topic: str = "/remote_states/arms/{pos}/joint_states"


class AIRBOTReplay(object):
    def __init__(self, config: Optional[AIRBOTReplayConfig] = None, **kwargs) -> None:
        if config is None:
            config = AIRBOTReplayConfig()
        self.config = replace(config, **kwargs)
        topic_name = self.config.joint_states_topic
        self._node = rclpy.create_node(
            f"airbot_replay_remote_{topic_name.split('/')[-2]}"
        )
        self._logger = self._node.get_logger()
        # Overwrite config arguments using kwargs (used for yaml config)
        self._joint_states: JointState = None
        self._joint_states_sub = self._node.create_subscription(
            JointState, topic_name, self._joint_states_callback
        )
        self._logger.info(f"Waiting for joint states topic: {topic_name}")
        while self._joint_states is None:
            rclpy.spin_once(self._node)
        self._logger.info(f"Received joint states topic")

    def reset(self):
        pass

    def send_action(self, action, wait=False):
        return action

    def get_low_dim_data(self):
        data = {}
        data["/time"] = time.time()
        joint_positions = self._get_current_joint_positions()
        data["observation/arm/joint_position"] = joint_positions[:6]
        data["observation/eef/joint_position"] = joint_positions[6:]
        data["observation/eef/pose"] = [0] * 7
        return data

    def exit(self):
        assert not self._exited, "Robot already exited"
        self._exited = True
        logger.info("Robot exited")

    def get_state_mode(self):
        return self._state_mode

    def enter_traj_mode(self):
        self._state_mode = "active"

    def enter_active_mode(self):
        self._state_mode = "active"

    def enter_passive_mode(self):
        self._state_mode = "passive"

    def enter_servo_mode(self):
        self._state_mode = "active"

    def _joint_states_callback(self, msg):
        self._joint_states = msg

    def _get_current_joint_positions(self) -> List[float]:
        if self._joint_states is None:
            return [0] * 7
        return self._joint_states.position

    def _get_current_joint_velocities(self):
        if self._joint_states is None:
            return [0] * 7
        return self._joint_states.velocity

    def _get_current_joint_efforts(self):
        if self._joint_states is None:
            return [0] * 7
        return self._joint_states.effort

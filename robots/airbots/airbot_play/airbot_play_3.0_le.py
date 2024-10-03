from dataclasses import dataclass, field, replace
import time
import torch
from le_studio.common.robot_devices.cameras.utils import Camera
from le_studio.common.robot_devices.utils import (
    RobotDeviceAlreadyConnectedError,
    RobotDeviceNotConnectedError,
)
import numpy as np


@dataclass
class AIRBOTPlayConfig(object):
    """
    Example of usage:
    ```python
    AIRBOTPlayConfig()
    ```
    """

    model_path: str | None = "/usr/share/airbot_models/airbot_play_with_gripper.urdf"
    gravity_mode: str = "down"
    can_bus: str = "can0"
    vel: float = 2.0
    eef_mode: str = "gripper"
    bigarm_type: str = "OD"
    forearm_type: str = "DM"

    joint_vel: float = 0.5
    robot_type: str | None = None
    joint_num: int = 7
    default_action: tuple[float] = (0, -0.766, 0.704, 1.537, -0.965, -1.576, 0)

    cameras: dict[str, Camera] = field(default_factory=lambda: {})


class AIRBOTPlay(object):
    def __init__(self, config: AIRBOTPlayConfig | None = None, **kwargs) -> None:
        if config is None:
            config = AIRBOTPlayConfig()
        # Overwrite config arguments using kwargs
        self.config = replace(config, **kwargs)
        self.calibration_dir = None

        self.robot_type = self.config.robot_type
        self.leader_arms = {}
        self.follower_arms = {}
        self.cameras = self.config.cameras
        self.is_connected = False
        self.logs = {}

    def connect(self):
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                "ManipulatorRobot is already connected. Do not run `robot.connect()` twice."
            )
        self.follower_arms["main"] = airbot_play_py.create_agent(
            *tuple(self.config.__dict__.values())[:7]
        )

        # Connect the cameras
        for name in self.cameras:
            self.cameras[name].connect()
        goal_pos = self.config.default_action
        self.follower_arms["main"].set_target_joint_q(goal_pos[:-1], True, 0.8, True)
        self.follower_arms["main"].set_target_end(goal_pos[-1], True)

        self.is_connected = True

    def capture_observation(self):
        """The returned observations do not have a batch dimension."""
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "ManipulatorRobot is not connected. You need to run `robot.connect()`."
            )

        # Read follower position
        follower_pos = {}
        for name in self.follower_arms:
            before_fread_t = time.perf_counter()
            # TODO: fix position reading
            follower_pos[name] = np.array(
                self.follower_arms[name].get_current_joint_q() + [
                self.follower_arms[name].get_current_end()
            ], dtype=np.float32
            )
            follower_pos[name] = torch.from_numpy(follower_pos[name])
            self.logs[f"read_follower_{name}_pos_dt_s"] = (
                time.perf_counter() - before_fread_t
            )

        # Create state by concatenating follower current position
        state = []
        for name in self.follower_arms:
            if name in follower_pos:
                state.append(follower_pos[name])
        state = torch.cat(state)

        # Capture images from cameras
        images = {}
        for name in self.cameras:
            before_camread_t = time.perf_counter()
            images[name] = self.cameras[name].async_read()
            images[name] = torch.from_numpy(images[name])
            self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs[
                "delta_timestamp_s"
            ]
            self.logs[f"async_read_camera_{name}_dt_s"] = (
                time.perf_counter() - before_camread_t
            )

        # Populate output dictionnaries and format to pytorch
        obs_dict = {}
        obs_dict["observation.state"] = state
        # for name in self.cameras:
        #     obs_dict[f"observation.images.{name}"] = images[name]
        obs_dict["observation.image"] = images[list(self.cameras.keys())[0]]
        return obs_dict

    def send_action(self, action: torch.Tensor) -> torch.Tensor:
        """Command the follower arms to move to a target joint configuration.

        The relative action magnitude may be clipped depending on the configuration parameter
        `max_relative_target`. In this case, the action sent differs from original action.
        Thus, this function always returns the action actually sent.

        Args:
            action: tensor containing the concatenated goal positions for the follower arms.
        """
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "ManipulatorRobot is not connected. You need to run `robot.connect()`."
            )

        from_idx = 0
        to_idx = 0
        action_sent = []
        for name in self.follower_arms:
            # Get goal position of each follower arm by splitting the action vector
            to_idx += self.config.joint_num
            goal_pos = action[from_idx:to_idx]
            from_idx = to_idx

            # Save tensor to concat and return
            action_sent.append(goal_pos)

            # Send goal position to each follower
            goal_pos = goal_pos.numpy()
            arm_pos = goal_pos[:-1]
            gripper_pos = goal_pos[-1]
            self.follower_arms[name].set_target_joint_q(arm_pos)
            self.follower_arms[name].set_target_end(gripper_pos)
            # print(f"arm_pos: {arm_pos}, gripper_pos: {gripper_pos}")
            # input("Press Enter to continue...")

        return torch.cat(action_sent)

    def disconnect(self):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "ManipulatorRobot is not connected. You need to run `robot.connect()` before disconnecting."
            )

        del self.follower_arms["main"]
        for name in self.cameras:
            self.cameras[name].disconnect()

        self.is_connected = False

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()

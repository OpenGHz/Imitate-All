from dataclasses import dataclass, field, replace
import time
import torch
from le_studio.common.robot_devices.cameras.utils import Camera
from le_studio.common.robot_devices.utils import (
    RobotDeviceAlreadyConnectedError,
    RobotDeviceNotConnectedError,
)
import argparse
from typing import Dict, Optional, List
import sys

sys.path.insert(0, "/home/ghz/Work/airbot_play/airbot_sdk/python3/host_sdk_service")

from airbot_client import Robot


@dataclass
class AIRBOTPlayConfig(object):
    """
    Example of usage:
    ```python
    AIRBOTPlayConfig()
    ```
    """

    # model_path: Optional[str] = "/usr/share/airbot_models/airbot_play_with_gripper.urdf"
    # gravity_mode: str = "down"
    # can_bus: str = "can0"
    # vel: float = 2.0
    # eef_mode: str = "gripper"
    # bigarm_type: str = "OD"
    # forearm_type: str = "DM"

    # joint_vel: float = 0.5
    # robot_type: Optional[str] = None
    # joint_num: int = 7
    # default_action: tuple[float] = (0, -0.766, 0.704, 1.537, -0.965, -1.576, 0)

    cameras: Dict[str, Camera] = field(default_factory=lambda: {})


class AIRBOTPlay(object):
    def __init__(self, config: Optional[AIRBOTPlayConfig] = None, **kwargs) -> None:
        if config is None:
            config = AIRBOTPlayConfig()
        # Overwrite config arguments using kwargs (used for yaml config)
        self.config = replace(config, **kwargs)
        self.cameras = self.config.cameras
        self.args = self.parse_args()
        self.init_robot()

    def init_robot(self):
        args = self.args
        leader_robot = []
        follower_robot = []
        for i in range(args.leader_number):
            leader_robot.append(Robot(arm_type=args.leader_arm_type[i], end_effector=args.leader_end_effector[i], can_interface=args.leader_can_interface[i], domain_id=args.leader_domain_id[i]))
        for i in range(args.follower_number):
            follower_robot.append(Robot(arm_type=args.follower_arm_type[i], end_effector=args.follower_end_effector[i], can_interface=args.follower_can_interface[i], domain_id=args.follower_domain_id[i]))
        self.leader_robot = leader_robot
        self.follower_robot = follower_robot
        self.reset_robot()

    def reset_robot(self):
        args = self.args
        leader_robot = self.leader_robot
        follower_robot = self.follower_robot
        for i in range(args.leader_number):
            if args.leader_arm_type[i] == "replay":
                continue
            if leader_robot[i].get_current_state() != "ONLINE_TRAJ":
                assert leader_robot[i].online_idle_mode(), "Leader robot %d online idle mode failed" % i
                assert leader_robot[i].online_traj_mode(), "Leader robot %d online traj mode failed" % i
            time.sleep(0.5)
        for i in range(args.follower_number):
            if follower_robot[i].get_current_state() != "SLAVE_MOVING":
                assert follower_robot[i].online_idle_mode(), "Follower robot %d online idle mode failed" % i
                assert follower_robot[i].slave_waiting_mode(args.leader_domain_id[i]), "Follower robot %d slave waiting mode failed" % i
                time.sleep(0.5)
                assert follower_robot[i].slave_reaching_mode(), "Follower robot %d slave reaching mode failed" % i
                while follower_robot[i].get_current_state() != "SLAVE_REACHED":
                    time.sleep(0.5)
                assert follower_robot[i].slave_moving_mode(), "Follower robot %d slave moving mode failed" % i
        for i in range(args.leader_number):
            if args.leader_arm_type[i] == "replay":
                continue
            assert leader_robot[i].set_target_joint_q(args.start_joint_position), "Leader robot %d set target joint q failed" % i
            assert leader_robot[i].online_idle_mode(), "Leader robot %d online idle mode failed" % i

    def _get_arm_eef_data(self):
        args = self.args
        leader_robot = self.leader_robot
        follower_robot = self.follower_robot
        data = {}
        data["/time"] = time.time()
        obs_arm_jq = []
        obs_eef_jq = []
        obs_eef_pose = []
        for i in range(args.leader_number):
            obs_arm_jq.extend(leader_robot[i].get_current_joint_q())
            obs_eef_jq.append(leader_robot[i].get_current_end())
            pose = leader_robot[i].get_current_pose()
            obs_eef_pose.extend(pose[0] + pose[1])  # xyz + quat(xyzw)
        data["observation/arm/joint_position"] = obs_arm_jq
        data["observation/eef/joint_position"] = obs_eef_jq
        data["observation/eef/pose"] = obs_eef_pose

        action_arm_jq = []
        action_eef_jq = []
        action_eef_pose = []
        for i in range(args.follower_number):
            action_arm_jq.extend(follower_robot[i].get_current_joint_q())
            action_eef_jq.append(follower_robot[i].get_current_end())
            pose = follower_robot[i].get_current_pose()
            action_eef_pose.extend(pose[0] + pose[1])  # xyz + quat(xyzw)
        data["action/arm/joint_position"] = action_arm_jq
        data["action/eef/joint_position"] = action_eef_jq
        data["action/eef/pose"] = action_eef_pose
        return data

    @staticmethod
    def parse_args()-> argparse.Namespace:
        parser = argparse.ArgumentParser(description="Aloha data collection")
        parser.add_argument("--leader_number", type=int, default=1, help="Number of the leader")
        parser.add_argument("--follower_number", type=int, default=1, help="Number of the follower")
        parser.add_argument("--leader_arm_type", type=List[str], default=["play_long"], help="Type of the leader's arm")
        parser.add_argument("--follower_arm_type", type=List[str], default=["play_long"], help="Type of the follower's arm")
        parser.add_argument("--leader_end_effector", type=List[str], default=["E2B"], help="End effector of the leader's arm")
        parser.add_argument("--follower_end_effector", type=List[str], default=["G2"], help="End effector of the follower's arm")
        parser.add_argument("--leader_can_interface", type=List[str], default=["can0"], help="Can interface of the leader's arm")
        parser.add_argument("--follower_can_interface", type=List[str], default=["can1"], help="Can interface of the follower's arm")
        parser.add_argument("--leader-domain-id", type=List[int], default=[50], help="Domain id of the leader")
        parser.add_argument("--follower-domain-id", type=List[int], default=[100], help="Domain id of the follower")
        parser.add_argument("--frequency", type=int, default=25, help="Frequency of the data collection")
        parser.add_argument("--start-episode", type=int, default=0, help="Start episode")
        parser.add_argument("--end-episode", type=int, default=100, help="End episode")
        parser.add_argument("--task-name", type=str, default="aloha", help="Task name")
        parser.add_argument("--start-joint-position", type=list, default=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], help="Start joint position")
        parser.add_argument("--start-end-effector-position", type=float, default=0.0, help="Start end effector position")
        assert len(parser.parse_args().leader_arm_type) == parser.parse_args().leader_number
        assert len(parser.parse_args().follower_arm_type) == parser.parse_args().follower_number
        assert len(parser.parse_args().leader_end_effector) == parser.parse_args().leader_number
        assert len(parser.parse_args().follower_end_effector) == parser.parse_args().follower_number
        assert len(parser.parse_args().leader_can_interface) == parser.parse_args().leader_number
        assert len(parser.parse_args().follower_can_interface) == parser.parse_args().follower_number
        assert len(parser.parse_args().leader_domain_id) == parser.parse_args().leader_number
        assert len(parser.parse_args().follower_domain_id) == parser.parse_args().follower_number
        return parser.parse_args()

    def capture_observation(self):
        """The returned observations do not have a batch dimension."""
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "ManipulatorRobot is not connected. You need to run `robot.connect()`."
            )

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

        low_dim_data = self._get_arm_eef_data()

        # Populate output dictionnaries and format to pytorch
        obs_act_dict = {}
        obs_act_dict["low_dim"] = low_dim_data
        for name in self.cameras:
            obs_act_dict[f"observation.images.{name}"] = images[name]
        # obs_dict["observation.image"] = images[list(self.cameras.keys())[0]]
        return obs_act_dict

    def exit(self):
        print("Robot exited")
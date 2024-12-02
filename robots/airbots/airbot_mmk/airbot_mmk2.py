from mmk2_sdk.mmk2_client import AIRBOTMMK2 as AIRBOTMMK2Client
from mmk2_types.types import MMK2Components, JointNames, ComponentTypes
from mmk2_types.grpc_msgs import (
    JointState,
    TrajectoryParams,
    MoveServoParams,
)
from typing import Optional, Dict, List
from dataclasses import dataclass, replace, field
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AIRBOTMMK2Config(object):
    name: str = "mmk2"
    domain_id: int = -1
    ip: str = "localhost"
    port: int = 50055
    default_action: List[float] = field(default_factory=lambda: [0] * 14)
    cameras: Dict[str, str] = field(default_factory=lambda: {})
    components: List[str] = field(
        default_factory=lambda: [
            MMK2Components.LEFT_ARM.value,
            MMK2Components.RIGHT_ARM.value,
            MMK2Components.LEFT_EEF.value,
            MMK2Components.RIGHT_EEF.value,
        ]
    )


class AIRBOTMMK2(object):
    def __init__(self, config: Optional[AIRBOTMMK2Config] = None, **kwargs) -> None:
        if config is None:
            config = AIRBOTMMK2Config()
        self.config = replace(config, **kwargs)
        self.robot = AIRBOTMMK2Client(
            self.config.name,
            self.config.domain_id,
            self.config.ip,
            self.config.port,
        )
        self.joint_names = {}
        self.cameras: Dict[MMK2Components, str] = {}
        self.components: Dict[MMK2Components, ComponentTypes] = {}
        all_joint_names = JointNames()
        self.joint_num = 0
        for k, v in self.config.cameras.items():
            self.cameras[MMK2Components(k)] = v
        for comp_str in self.config.components:
            comp = MMK2Components(comp_str)
            self.components[comp] = ComponentTypes.UNKNOWN
            names = all_joint_names.__dict__[comp_str]
            self.joint_names[comp] = names
            self.joint_num += len(names)
        self.logs = {}

    def reset(self, sleep_time=0):
        # goal = {
        #     MMK2Components.LEFT_ARM: JointState(
        #         position=self.config.default_action[:6]
        #     ),
        #     MMK2Components.LEFT_EEF: JointState(
        #         position=self.config.default_action[6:7]
        #     ),
        #     MMK2Components.RIGHT_ARM: JointState(
        #         position=self.config.default_action[7:13]
        #     ),
        #     MMK2Components.RIGHT_EEF: JointState(
        #         position=self.config.default_action[13:14]
        #     ),
        # }
        # self.robot.set_goal(goal, TrajectoryParams())
        goal = {
            MMK2Components.LEFT_ARM: JointState(
                position=self.config.default_action[:7]
            ),
            MMK2Components.RIGHT_ARM: JointState(
                position=self.config.default_action[7:14]
            ),
        }
        if sleep_time > 0:
            for _ in range(5):
                self.robot.set_goal(goal, MoveServoParams(header=self.robot.get_header()))
                time.sleep(5 / 5)

    def send_action(self, action, wait=False):
        goal = {
            MMK2Components.LEFT_ARM: JointState(position=action[:7]),
            MMK2Components.RIGHT_ARM: JointState(position=action[7:]),
        }
        self.robot.set_goal(
            goal,
            MoveServoParams(header=self.robot.get_header()),
        )

    def get_low_dim_data(self):
        data = {}
        all_joints = self.robot.get_robot_state().joint_state
        for comp in self.components:
            joint_states = self.robot.get_joint_values_by_names(
                all_joints, self.joint_names[comp]
            )
            data[f"observation/{comp.value}/joint_position"] = joint_states
        return data

    def capture_observation(self):
        """The returned observations do not have a batch dimension."""
        obs_act_dict = {}
        # Capture images from cameras
        images = {}
        before_camread_t = time.perf_counter()
        comp_images = self.robot.get_image(self.cameras)
        for comp, image in comp_images.items():
            # TODO: now only support for color image
            kind = self.cameras[comp]
            if kind == "rgb":
                images[comp.value] = image.color
            elif kind == "depth":
                images[comp.value] = image.depth
            elif kind == "rgb-d":
                images[comp.value + "_color"] = image.color
                images[comp.value + "_depth"] = image.depth
            obs_act_dict[f"/time/{comp.value}"] = (
                image.stamp.sec + image.stamp.nanosec / 1e9
            )

        name = "cameras"
        # self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs[
        #     "delta_timestamp_s"
        # ]
        self.logs[f"async_read_camera_{name}_dt_s"] = (
            time.perf_counter() - before_camread_t
        )

        low_dim_data = self.get_low_dim_data()

        # Populate output dictionnaries and format to pytorch
        obs_act_dict["low_dim"] = low_dim_data
        for name in images:
            obs_act_dict[f"observation.images.{name}"] = images[name]
        return obs_act_dict


def main(args=object):
    robot = AIRBOTMMK2()
    robot.reset()


if __name__ == "__main__":
    main()

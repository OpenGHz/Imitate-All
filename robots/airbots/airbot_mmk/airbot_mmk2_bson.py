from mmk2_sdk.mmk2_client import AIRBOTMMK2 as AIRBOTMMK2Client
from mmk2_types.types import (
    MMK2Components,
    JointNames,
    ComponentTypes,
    TopicNames,
    MMK2ComponentsGroup,
    ImageTypes,
)
from mmk2_types.grpc_msgs import (
    JointState,
    TrajectoryParams,
    MoveServoParams,
    Time,
)
from typing import Optional, Dict, List
from dataclasses import dataclass, replace, field
import time
import logging
import numpy as np


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
    demonstrate: bool = False


class AIRBOTMMK2(object):
    def __init__(self, config: Optional[AIRBOTMMK2Config] = None, **kwargs) -> None:
        if config is None:
            config = AIRBOTMMK2Config()
        self.config = replace(config, **kwargs)
        self.robot = AIRBOTMMK2Client(
            self.config.ip,
            self.config.port,
            self.config.name,
            self.config.domain_id,
        )
        self.joint_names = {}
        self.cameras: Dict[MMK2Components, str] = {}
        self.images_ts: Dict[str, int] = {}
        self.components: Dict[MMK2Components, ComponentTypes] = {}
        all_joint_names = JointNames()
        self.joint_num = 0
        for k, v in self.config.cameras.items():
            self.cameras[MMK2Components(k)] = ImageTypes(v)
        for comp_str in self.config.components:
            comp = MMK2Components(comp_str)
            # TODO: get the type info from SDK
            self.components[comp] = ComponentTypes.UNKNOWN
            names = all_joint_names.__dict__[comp_str]
            self.joint_names[comp] = names
            self.joint_num += len(names)
        self.robot.enable_resources(
            {
                comp: {
                    "rgb_camera.color_profile": "640,480,30",
                    "enable_depth": "false",
                }
                for comp in self.cameras
            }
        )
        if self.config.demonstrate:
            comp_action_topic = {
                comp: TopicNames.tracking.format(component=comp.value)
                for comp in MMK2ComponentsGroup.ARMS
            }
            comp_action_topic.update(
                {
                    comp: TopicNames.forward_position.format(component=comp.value)
                    for comp in MMK2ComponentsGroup.HEAD_SPINE
                }
            )
            self.robot.listen_to(list(comp_action_topic.values()))
            self._comp_action_topic = comp_action_topic
        self.logs = {}
        self.enter_active_mode = lambda: self._set_mode("active")
        self.enter_passive_mode = lambda: self._set_mode("passive")
        self.get_state_mode = lambda: self._state_mode
        self.exit = lambda: None

    def reset(self, sleep_time=0):
        if self.config.default_action is not None:
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
                    self.robot.set_goal(
                        goal, MoveServoParams(header=self.robot.get_header())
                    )
                    time.sleep(0.2)

    def send_action(self, action, wait=False):
        goal = {
            MMK2Components.LEFT_ARM: JointState(position=action[:7]),
            MMK2Components.RIGHT_ARM: JointState(position=action[7:]),
        }
        self.robot.set_goal(
            goal,
            MoveServoParams(header=self.robot.get_header()),
        )

    def _get_joint_state(
        self, ns: str, comp: str, stamp: Time, pos=None, vel=None, eff=None
    ) -> dict:
        data = {}
        handle = "joint_state"
        data[f"/{ns}/{comp}/{handle}"] = {
            "t": int((stamp.sec + stamp.nanosec / 1e9) * 1e3),
            "data": {
                "pos": pos,
                "vel": vel,
                "eff": eff,
            },
        }
        return data

    def _get_image(self, comp: str, stamp: Time, image: np.ndarray) -> dict:
        data = {}
        data[f"/images/{comp}"] = {
            "t": int((stamp.sec + stamp.nanosec / 1e9) * 1e3),
            "data": image,
        }
        if self.images_ts != {} and data[f"/images/{comp}"]["t"] <= self.images_ts[f"/images/{comp}"]:
            data[f"/images/{comp}"]["t"] = self.images_ts[f"/images/{comp}"]+1
        self.images_ts[f"/images/{comp}"] = data[f"/images/{comp}"]["t"]
        return data

    def get_low_dim_data(self):
        data = {}
        all_joints = self.robot.get_robot_state().joint_state
        for comp in self.components:
            joint_pos = self.robot.get_joint_values_by_names(
                all_joints, self.joint_names[comp], "position"
            )
            joint_vel = self.robot.get_joint_values_by_names(
                all_joints, self.joint_names[comp], "velocity"
            )
            joint_eff = self.robot.get_joint_values_by_names(
                all_joints, self.joint_names[comp], "effort"
            )
            stamp = all_joints.header.stamp
            data.update(
                self._get_joint_state("observation", comp.value, stamp, joint_pos, joint_vel, joint_eff)
            )
            if self.config.demonstrate:
                if comp in MMK2ComponentsGroup.ARMS:
                    arm_jn = JointNames().__dict__[comp.value]
                    comp_eef = comp.value.replace("arm", "eef")
                    eef_jn = JointNames().__dict__[comp_eef]
                    js = self.robot.get_listened(self._comp_action_topic[comp])
                    if js is not None:
                        jq = self.robot.get_joint_values_by_names(js, arm_jn + eef_jn)
                        data.update(
                            self._get_joint_state(
                                "action", comp.value, js.header.stamp, jq[:-1]
                            )
                        )
                        data.update(
                            self._get_joint_state(
                                "action", comp_eef, js.header.stamp, jq[-1:]
                            )
                        )
                elif comp in MMK2ComponentsGroup.HEAD_SPINE:
                    result = self.robot.get_listened(self._comp_action_topic[comp])
                    if result is not None:
                        jq = list(result.data)
                        data.update(
                            self._get_joint_state("action", comp.value, result.stamp, jq)
                        )
        return data

    def capture_observation(self):
        """The returned observations do not have a batch dimension."""
        # Capture images from cameras
        images = {}
        img_stamps = {}
        before_camread_t = time.perf_counter()
        comp_images = self.robot.get_image(self.cameras)
        for comp, image in comp_images.items():
            # TODO: now only support for color image
            kind = self.cameras[comp]
            if kind == ImageTypes.RGB:
                images[comp.value] = image.color
            elif kind == ImageTypes.DEPTH:
                images[comp.value] = image.depth
            elif kind == ImageTypes.RGBD:
                images[comp.value + "_color"] = image.color
                images[comp.value + "_depth"] = image.depth
            img_stamps[comp.value] = image.stamp
        name = "cameras"
        # self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs[
        #     "delta_timestamp_s"
        # ]
        self.logs[f"async_read_camera_{name}_dt_s"] = (
            time.perf_counter() - before_camread_t
        )
        data = self.get_low_dim_data()
        for name, img in images.items():
            data.update(self._get_image(name, img_stamps[name], img))
        return data

    def _set_mode(self, mode):
        self._state_mode = mode


def main():
    robot = AIRBOTMMK2()
    robot.reset()


if __name__ == "__main__":
    main()

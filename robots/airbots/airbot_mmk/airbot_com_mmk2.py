from airbot_py.airbot_mmk2 import AirbotMMK2
from mmk2_types.types import (
    RobotComponents,
    JointNames,
    ComponentTypes,
    TopicNames,
    RobotComponentsGroup,
    ImageTypes,
    ControllerTypes,
)
from mmk2_types.grpc_msgs import (
    Time,
    JointState,
    TrajectoryParams,
    MoveServoParams,
    TrackingParams,
    ForwardPositionParams,
    Pose3D,
    Twist3D,
)
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, replace, field
import time
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AIRBOTRobotConfig(object):
    name: str = "mmk2"
    domain_id: int = -1
    ip: str = "192.168.11.200"
    port: int = 50055
    default_action: Optional[List[float]] = None
    cameras: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    components: List[str] = field(
        default_factory=lambda: [
            RobotComponents.LEFT_ARM.value,
            RobotComponents.LEFT_ARM_EEF.value,
            RobotComponents.RIGHT_ARM.value,
            RobotComponents.RIGHT_ARM_EEF.value,
        ]
    )
    demonstrate: bool = False
    check_dim: bool = True
    ignore_base_action: bool = False


class AIRBOTMMK2(object):
    def __init__(self, config: Optional[AIRBOTRobotConfig] = None, **kwargs) -> None:
        if config is None:
            config = AIRBOTRobotConfig()
        self.config = replace(config, **kwargs)
        self.robot = AirbotMMK2(
            self.config.ip,
            self.config.port,
            self.config.name,
            self.config.domain_id,
        )
        self.joint_names = {}
        self.cameras_goal: Dict[RobotComponents, List[ImageTypes]] = {}
        self.cameras_cfg: Dict[RobotComponents, Dict[str, str]] = {}
        self.cameras = self.cameras_goal.keys()
        self.components: Dict[RobotComponents, ComponentTypes] = {}
        self.joint_num = 0
        for k, cfg in self.config.cameras.items():
            comp = RobotComponents(k)
            types = {ImageTypes(v) for v in cfg.pop("image_types")}
            self.cameras_goal[comp] = types
            if types != {ImageTypes.COLOR}:
                cfg["enable_depth"] = "true"
            if ImageTypes.ALIGNED_DEPTH_TO_COLOR in types:
                cfg["align_depth.enable"] = "true"
            self.cameras_cfg[comp] = cfg
        for comp_str in self.config.components:
            comp = RobotComponents(comp_str)
            # TODO: get the type info from SDK
            self.components[comp] = ComponentTypes.UNKNOWN
            names = JointNames[comp.name].value
            if comp == RobotComponents.BASE:
                # TODO: fix base control
                names.append("base")
            self.joint_names[comp] = names
            self.joint_num += len(names)
        logger.info(f"Components: {self.components}")
        logger.info(f"Joint numbers: {self.joint_num}")
        logger.info(f"enable resources cfg: {self.cameras_cfg}")
        self.robot.enable_resources(self.cameras_cfg)
        # use stream to get images
        # self.robot.enable_stream(self.robot.get_image, self.cameras)
        comp_action_topic = {}
        if self.config.demonstrate:
            for comp in self.components:
                if comp in RobotComponentsGroup.ARMS:
                    comp_action_topic[comp] = TopicNames.tracking.format(
                        component=comp.value
                    )
                elif comp in RobotComponentsGroup.HEAD_SPINE:
                    comp_action_topic[comp] = TopicNames.controller_command.format(
                        controller=f"/{comp.value}_{ControllerTypes.FORWARD_POSITION.value}_controller"
                    )
            self.robot.listen_to(list(comp_action_topic.values()))
            self._comp_action_topic = comp_action_topic
        self.logs = {}
        self.enter_active_mode = lambda: self._set_mode("active")
        self.enter_passive_mode = lambda: self._set_mode("passive")
        self.get_state_mode = lambda: self._state_mode
        self.exit = lambda: None
        # logger.info("Warm up the robot")
        # for _ in range(5):
        #     self._capture_images()
        # logger.info("AIRBOTMMK2 is ready")
        self.reset()

    def _move_by_traj(self, goal: dict):
        if self.config.demonstrate:
            # TODO: since the arms and eefs are controlled by the teleop bag
            for comp in RobotComponentsGroup.ARMS_EEFS:
                goal.pop(comp, None)
        if goal:
            # start = time.time()
            # logger.info(f"Move by trajectory")
            self.robot.set_goal(goal, TrajectoryParams())
            # logger.info(f"Move by trajectory time: {time.time() - start}")
            self.robot.set_goal(goal, ForwardPositionParams())
            # logger.info(f"Move by trajectory: {goal}")
        return goal

    def reset(self, sleep_time=0):
        if self.config.default_action is not None:
            goal = self._action_to_goal(self.config.default_action)
            logger.info(f"Reset to default action: {self.config.default_action}")
            # logger.info(f"Reset to default goal: {goal}")
            # TODO: hard code for spine&head control
            self._move_by_traj(goal)
        else:
            logger.warning("No default action is set.")
        time.sleep(sleep_time)
        self.enter_servo_mode()

    def send_action(self, action, wait=False):
        goal = self._action_to_goal(action)
        # logger.info(f"Send action: {action}")
        # logger.info(f"Send goal: {goal}")

        # param = MoveServoParams(header=self.robot.get_header())
        if self.config.ignore_base_action:
            goal.pop(RobotComponents.BASE, None)
        if self.traj_mode:
            self._move_by_traj(goal)
        else:
            param = ForwardPositionParams()
            # param = TrackingParams()
            # param = MoveServoParams(header=self.robot.get_header())
            # param = {
            #     RobotComponents.LEFT_ARM: MoveServoParams(header=self.robot.get_header()),
            #     RobotComponents.RIGHT_ARM: MoveServoParams(header=self.robot.get_header()),
            #     RobotComponents.LEFT_ARM_EEF: TrajectoryParams(),
            #     RobotComponents.RIGHT_ARM_EEF: TrajectoryParams(),
            #     RobotComponents.HEAD: ForwardPositionParams(),
            #     RobotComponents.SPINE: ForwardPositionParams(),
            # }
            self.robot.set_goal(goal, param)

    def get_low_dim_data(self):
        data = {}
        robot_state = self.robot.get_robot_state()
        all_joints = robot_state.joint_state
        # logger.info(f"joint_stamp: {all_joints.header.stamp}")
        for comp in self.components:
            joint_states = self.robot.get_joint_values_by_names(
                all_joints, self.joint_names[comp]
            )
            data[f"observation/{comp.value}/joint_position"] = joint_states
            if comp == RobotComponents.BASE:
                base_pose = robot_state.base_state.pose
                base_vel = robot_state.base_state.velocity
                data_pose = [
                    base_pose.x,
                    base_pose.y,
                    base_pose.theta,
                ]
                data[f"action/{comp.value}/pose"] = data_pose
                data_vel = [
                    base_vel.x,
                    base_vel.y,
                    base_vel.omega,
                ]
                data[f"action/{comp.value}/velocity"] = data_vel
                data[f"action/{comp.value}/joint_position"] = data_vel + data_pose
            if self.config.demonstrate:
                if comp in RobotComponentsGroup.ARMS:
                    arm_jn = JointNames().__dict__[comp.value]
                    comp_eef = comp.value + "_eef"
                    eef_jn = JointNames().__dict__[comp_eef]
                    js = self.robot.get_listened(self._comp_action_topic[comp])
                    jq = self.robot.get_joint_values_by_names(js, arm_jn + eef_jn)
                    data[f"action/{comp.value}/joint_position"] = jq[:-1]
                    # the eef joint is in arms
                    data[f"action/{comp_eef}/joint_position"] = jq[-1:]
                elif comp in RobotComponentsGroup.HEAD_SPINE:
                    jq = list(
                        self.robot.get_listened(self._comp_action_topic[comp]).data
                    )
                    data[f"action/{comp.value}/joint_position"] = jq
        return data

    def _capture_images(self) -> Tuple[Dict[str, bytes], Dict[str, Time]]:
        images = {}
        img_stamps: Dict[RobotComponents, Time] = {}
        before_camread_t = time.perf_counter()
        comp_images = self.robot.get_image(self.cameras_goal)
        for comp, image in comp_images.items():
            # TODO: now only support for color image
            images[comp.value] = image.data[ImageTypes.COLOR]
            img_stamps[comp.value] = image.stamp

        name = "cameras"
        # self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs[
        #     "delta_timestamp_s"
        # ]
        self.logs[f"async_read_camera_{name}_dt_s"] = (
            time.perf_counter() - before_camread_t
        )
        return images, img_stamps

    def capture_observation(self):
        """The returned observations do not have a batch dimension."""
        # Capture images from cameras
        images, img_stamps = self._capture_images()
        low_dim_data = self.get_low_dim_data()

        obs_act_dict = {}
        for comp, stamp in img_stamps.items():
            obs_act_dict[f"/time/{comp}"] = stamp.sec + stamp.nanosec * 1e-9
        # Populate output dictionnaries and format to pytorch
        obs_act_dict["low_dim"] = low_dim_data
        for name in images:
            obs_act_dict[f"observation.images.{name}"] = images[name]
        return obs_act_dict

    def low_dim_to_action(self, low_dim: dict, step: int) -> list:
        action = []
        # logger.info(low_dim.keys())
        for comp in self.components:
            # action.extend(low_dim[f"action/{comp.value}/joint_position"][step])
            # old version
            if comp in RobotComponentsGroup.ARMS_EEFS:
                pos_comp = comp.value.split("_")
                key = f"{pos_comp[1]}/{pos_comp[0]}"
            else:
                key = comp.value
            action.extend(low_dim[f"action/{key}/joint_position"][step])
        return action

    def _set_mode(self, mode):
        self._state_mode = mode

    def _action_check(self, action):
        assert (
            len(action) == self.joint_num
        ), f"Invalid action {action} with length: {len(action)}"

    def _action_to_goal(self, action) -> Dict[RobotComponents, JointState]:
        if self.config.check_dim:
            self._action_check(action)
        goal = {}
        j_cnt = 0
        for comp in self.components:
            end = j_cnt + len(self.joint_names[comp])
            goal[comp] = JointState(position=action[j_cnt:end])
            j_cnt = end
        return goal

    def enter_traj_mode(self):
        self.traj_mode = True

    def enter_servo_mode(self):
        self.traj_mode = False


def main():
    robot = AIRBOTMMK2()
    robot.reset()


if __name__ == "__main__":
    main()

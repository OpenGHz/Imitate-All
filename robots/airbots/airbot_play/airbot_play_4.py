from dataclasses import dataclass, replace, field
import time
from typing import Optional, List, Dict
from airbot_py.airbot_play import AirbotPlay as Robot
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AIRBOTPlayConfig(object):
    ip: str = "localhost"
    port: int = 50051
    default_action: List[float] = field(default_factory=lambda: [])


class AIRBOTPlay(object):
    def __init__(self, config: Optional[AIRBOTPlayConfig] = None, **kwargs) -> None:
        if config is None:
            config = AIRBOTPlayConfig()
        self.config = replace(config, **kwargs)
        self.logs = {}
        self.__init()
        self._state_mode = "active"
        self._exited = False

    def __init(self):
        args = self.config
        # Connect the robot
        logger.info(f"Connecting to robot at {args.ip}:{args.port}")
        self.robot = Robot(
            ip=args.ip,
            port=args.port,
        )
        logger.info(
            f"Connected to robot: {self.robot.arm_type} with {self.robot.eef_type}"
        )
        logger.info(f"ROS_DOMAIN_ID: {self.robot.ros_domain_id}")
        time.sleep(0.3)
        self.reset()

    def reset(self, robot_mode=None):
        logger.info("Resetting robot")
        args = self.config
        robot = self.robot
        # # set to traj mode
        # if (
        #     self.robot.arm_type != "replay"
        #     and robot.get_current_state() != "ONLINE_TRAJ"
        # ):
        #     assert robot.online_mode(), "online mode failed"
        # time.sleep(0.3)
        # go to start position
        if self.robot.arm_type != "replay":
            action = args.default_action
            assert len(action) in [6, 7], f"Invalid default action: {action}"
            joint_num = 6
            assert robot.set_target_joint_q(
                action[:joint_num], blocking=True
            ), "set target joint q failed"
            if len(action) == 7:
                assert robot.set_target_end(
                    action[joint_num], blocking=False
                ), "set target end failed"
            # enter target mode
            # if robot_mode in {"ONLINE_TRAJ", None}:
            #     self.enter_traj_mode()
            # elif robot_mode == "ONLINE_IDLE":
            #     self.enter_active_mode()
            # elif robot_mode == "ONLINE_SERVO":
            #     self.enter_servo_mode()
            # elif robot_mode:
            #     raise ValueError(
            #         f"Invalid default robot mode: {robot_mode}"
            #     )

        self._state_mode = "active"

    def enter_traj_mode(self):
        # self.enter_active_mode()
        # if self.robot.arm_type == "replay":
        #     return
        # else:
        #     assert self.robot.online_mode(), "online traj mode failed"
        # time.sleep(0.5)
        self._state_mode = "active"

    def enter_active_mode(self):
        # if self.robot.arm_type == "replay":
        #     return
        # else:
        #     assert self.robot.online_mode(), "online idle mode failed"
        self._state_mode = "active"

    def enter_passive_mode(self):
        if self.robot.arm_type == "replay":
            return
        else:
            assert self.robot.manual_mode(), "manual_mode failed"
        self._state_mode = "passive"

    def enter_servo_mode(self):
        # self.enter_active_mode()
        # if self.config.arm_type == "replay":
        #     return
        # else:
        #     assert self.robot.online_servo_mode(), "online_servo_mode mode failed"
        self._state_mode = "active"

    def send_action(self, action, wait=False):
        # assert self._state_mode == "active", "Robot is not in active mode"
        if self.robot.arm_type != "replay":
            assert self.robot.set_target_joint_q(
                action[:6], use_planning=wait, wait=wait
            ), "set target joint q failed"
            if self.robot.eef_type not in ["none", "E2B"]:
                assert self.robot.set_target_end(
                    action[6], wait
                ), "set target end failed"
        return action

    def get_low_dim_data(self) -> Dict[str, list]:
        data = {}
        data["/time"] = time.time()
        pose = self.robot.get_current_pose()
        data["observation/arm/joint_position"] = list(self.robot.get_current_joint_q())
        data["observation/eef/joint_position"] = [self.robot.get_current_end() or 0]
        data["observation/eef/pose"] = pose[0] + pose[1]  # xyz + quat(xyzw)
        return data

    def exit(self):
        assert not self._exited, "Robot already exited"
        self._exited = True
        logger.info("Robot exited")

    def get_state_mode(self):
        return self._state_mode

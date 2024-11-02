from dataclasses import dataclass, field, replace
from habitats.common.robot_devices.cameras.utils import Camera
from robots.airbots.airbot_play.airbot_play_3 import AIRBOTPlayConfig, AIRBOTPlay
from typing import Dict, Optional
import time


@dataclass
class AIRBOTTOKConfig(object):
    arms_cfg: Dict[str, AIRBOTPlayConfig] = field(default_factory=lambda: {})
    cameras: Dict[str, Camera] = field(default_factory=lambda: {})


class AIRBOTTOK(object):
    def __init__(self, config: Optional[AIRBOTTOKConfig] = None, **kwargs) -> None:
        if config is None:
            config = AIRBOTTOKConfig()
        self.config = replace(config, **kwargs)
        print(self.config)
        self.cameras = self.config.cameras
        self.arms_cfg = self.config.arms_cfg
        self.logs = {}
        self.__init()
        self._state_mode = "active"
        self._exited = False

    def __init(self):
        # Connect the cameras
        for name in self.cameras:
            self.cameras[name].connect()
        # Connect the robot
        self.arms = {}
        for arm_name, arm_cfg in self.arms_cfg.items():
            self.arms[arm_name] = AIRBOTPlay(**arm_cfg)
        time.sleep(0.3)
        self.reset()

    def reset(self):
        args = self.config
        robot = self.robot
        print(f"Resetting robot to start position: {args.start_arm_joint_position}")
        # set to traj mode
        if args.arm_type != "replay" and robot.get_current_state() != "ONLINE_TRAJ":
            assert robot.online_idle_mode(), "online idle mode failed"
            assert robot.online_traj_mode(), "online traj mode failed"
        time.sleep(0.3)
        # go to start position
        if args.arm_type != "replay":
            if args.start_arm_joint_position is not None:
                assert robot.set_target_joint_q(
                    args.start_arm_joint_position, wait=True
                ), "set target joint q failed"
            if args.start_eef_joint_position is not None:
                assert robot.set_target_end(
                    args.start_eef_joint_position, wait=False
                ), "set target end failed"
            # enter default mode
            if args.default_robot_mode == "ONLINE_TRAJ":
                self.enter_traj_mode()
            elif args.default_robot_mode == "ONLINE_IDLE":
                self.enter_active_mode()
            elif args.default_robot_mode == "ONLINE_SERVO":
                self.enter_servo_mode()
            else:
                raise ValueError(
                    f"Invalid default robot mode: {args.default_robot_mode}"
                )

        self._state_mode = "active"

    def enter_traj_mode(self):
        self.enter_active_mode()
        if self.config.arm_type == "replay":
            return
        else:
            assert self.robot.online_traj_mode(), "online traj mode failed"
        time.sleep(0.5)
        self._state_mode = "active"

    def enter_active_mode(self):
        if self.config.arm_type == "replay":
            return
        else:
            assert self.robot.online_idle_mode(), "online idle mode failed"
        self._state_mode = "active"

    def enter_passive_mode(self):
        if self.config.arm_type == "replay":
            return
        else:
            assert self.robot.demonstrate_prep_mode(), "demonstrate start mode failed"
        self._state_mode = "passive"

    def enter_servo_mode(self):
        self.enter_active_mode()
        if self.config.arm_type == "replay":
            return
        else:
            assert self.robot.online_servo_mode(), "online_servo_mode mode failed"
        self._state_mode = "active"

    def send_action(self, action, wait=False):
        assert self._state_mode == "active", "Robot is not in active mode"
        if self.config.arm_type == "replay":
            return
        else:
            assert self.robot.set_target_joint_q(
                action[:6], wait
            ), "set target joint q failed"
            assert self.robot.set_target_end(action[6], wait), "set target end failed"
        return action

    def get_low_dim_data(self):
        data = {}
        data["/time"] = time.time()
        pose = self.robot.get_current_pose()
        data["observation/arm/joint_position"] = list(self.robot.get_current_joint_q())
        data["observation/eef/joint_position"] = [self.robot.get_current_end()]
        data["observation/eef/pose"] = pose[0] + pose[1]  # xyz + quat(xyzw)
        return data

    def capture_observation(self):
        """The returned observations do not have a batch dimension."""
        obs_act_dict = {}
        # Capture images from cameras
        images = {}
        for name in self.cameras:
            before_camread_t = time.perf_counter()
            images[name] = self.cameras[name].async_read()
            # images[name] = torch.from_numpy(images[name])
            obs_act_dict[f"/time/{name}"] = time.time()
            self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs[
                "delta_timestamp_s"
            ]
            self.logs[f"async_read_camera_{name}_dt_s"] = (
                time.perf_counter() - before_camread_t
            )

        low_dim_data = self.get_low_dim_data()

        # Populate output dictionnaries and format to pytorch
        obs_act_dict["low_dim"] = low_dim_data
        for name in self.cameras:
            obs_act_dict[f"observation.images.{name}"] = images[name]
        return obs_act_dict

    def exit(self):
        assert not self._exited, "Robot already exited"
        for name in self.cameras:
            self.cameras[name].disconnect()
        self._exited = True
        print("Robot exited")

    def get_state_mode(self):
        return self._state_mode

import time
import numpy as np
import collections
import dm_env
from typing import List
from robots.common_robot import AssembledRobot
from dlabsim.envs.airbot_play_base import AirbotPlayCfg, AirbotPlayBase
import mujoco


class SimNode(AirbotPlayBase):
    def resetState(self):
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        if self.teleop:
            self.teleop.reset()

        self.jq = np.zeros(self.nj)
        self.jv = np.zeros(self.nj)
        self.mj_data.qpos[:self.nj] = self.init_joint_pose.copy()
        self.mj_data.ctrl[:self.nj] = self.init_joint_pose.copy()
        
        self.mj_data.qpos[self.nj+1] = 0.2 + (np.random.random() - 1) * 0.1 - 0.06
        self.mj_data.qpos[self.nj+2] = 0.1 + (np.random.random() - 0.5) * 0.1

        self.mj_data.qpos[self.nj+8] = 0.2 + (np.random.random() - 0) * 0.1 + 0.06
        self.mj_data.qpos[self.nj+9] = 0.1 + (np.random.random() - 0.5) * 0.1

        mujoco.mj_forward(self.mj_model, self.mj_data)

    def getObservation(self):
        self.obs = {
            "jq"       : self.jq.tolist(),
            "img"      : self.img_rgb_obs,
        }
        self.obs["jq"][6] *= 25.0 # gripper normalization
        return self.obs


class MujocoEnv:
    """
    Environment for real robot one-manual manipulation
    Action space:      [arm_qpos (6),             # absolute joint position
                        gripper_positions (1),]    # normalized gripper position (0: close, 1: open)

    Observation space: {"qpos": Concat[ arm_qpos (6),          # absolute joint position
                                        gripper_position (1),]  # normalized gripper position (0: close, 1: open)
                        "qvel": Concat[ arm_qvel (6),         # absolute joint velocity (rad)
                                        gripper_velocity (1),]  # normalized gripper velocity (pos: opening, neg: closing)
                        "images": {"0": (480x640x3),        # h, w, c, dtype='uint8'
                                   "1": (480x640x3),         # h, w, c, dtype='uint8'
                                   "2": (480x640x3),}  # h, w, c, dtype='uint8'
    """

    def __init__(
        self,
        robot_instances: List[AssembledRobot] = None,
        cameras=None,
    ):

        cfg = AirbotPlayCfg()
        cfg.expreriment  = "act_airbot_play"
        cfg.rb_link_list   = ["arm_base", "link1", "link2", "link3", "link4", "link5", "link6", "right", "left"]
        cfg.obj_list       = ["cup_blue", "cup_pink"]
        cfg.sync         = False
        cfg.headless     = False
        cfg.put_text     = False
        cfg.decimation   = 8
        cfg.render_set   = {
            "fps"    : 25,
            "width"  : 640,
            "height" : 480
        }
        cfg.obs_camera_id   = 1
        cfg.init_joint_pose = {
            "joint1"  :  0.06382703,
            "joint2"  : -0.71966516,
            "joint3"  :  1.2772779,
            "joint4"  : -1.5965166,
            "joint5"  :  1.72517278,
            "joint6"  :  1.80462028,
            "gripper" :  1
        }
        self.exec_node = SimNode(cfg)

        models_set = {
            "qz11/table_trans.ply",
            "qz11/qz_table_new.ply",
            "object/cup_blue.ply",
            "object/cup_pink.ply",
            "airbot_play/arm_base.ply",
            "airbot_play/link1.ply",
            "airbot_play/link2.ply",
            "airbot_play/link3.ply",
            "airbot_play/link4.ply",
            "airbot_play/link5.ply",
            "airbot_play/link6.ply",
            "airbot_play/left.ply",
            "airbot_play/right.ply",
        }
        self.exec_node.init_gs_render(models_set)

        self.exec_node.cam_id = self.exec_node.config.obs_camera_id
        self.reset_position = None

    def set_reset_position(self, reset_position):
        self.reset_position = reset_position
        print("Resetting to the given position: ", self.reset_position)

    def get_reward(self):
        return 0

    def reset(self, fake=False, sleep_time=0):
        raw_obs = self.exec_node.reset()
        # if self.reset_position is not None:
        #     # self.reset_position[-1] = 0.96
        #     # print("Resetting to the given position: ", self.reset_position)
        #     self.reset_position[-1] = 0.04  # undo the normalization
        #     self.exec_node.mj_data.ctrl[:7] = self.reset_position
        #     self.exec_node.mj_data.ctrl[7] = -self.exec_node.mj_data.ctrl[6]
        #     self.exec_node.mj_data.qpos[:7] = self.reset_position
        #     self.exec_node.mj_data.qpos[7] = -self.exec_node.mj_data.qpos[6]
        #     raw_obs, pri_obs, rew, ter, info = self.exec_node.step(self.reset_position)
        time.sleep(sleep_time)
        obs = collections.OrderedDict()
        obs["qpos"] = list(raw_obs["jq"])
        # print("obs gripper", raw_obs["jq"][-1])
        # print("pre_obs", obs["qpos"])
        # obs["qpos"][-1] *= 25  # undo the normalization
        # print("post_obs", obs["qpos"])
        # obs["qvel"] = raw_obs["jv"]
        obs["images"] = {}
        obs["images"]["0"] = raw_obs["img"][:, :, ::-1]
        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=self.get_reward(),
            discount=None,
            observation=obs,
        )

    def step(
        self,
        action,
        base_action=None,
        get_tracer_vel=False,
        get_obs=True,
        sleep_time=0,
        arm_vel=0,
    ):
        all_joints_num = (7, )
        # print("action", action)
        for index, jn in enumerate(all_joints_num):
            one_action = action[jn * index : jn * (index + 1)]
            # print("one_action", one_action)
            # print("one_action gripper", one_action[-1])
            # one_action[-1] = 0
            raw_obs, pri_obs, rew, ter, info = self.exec_node.step(one_action)
            # print("obs gripper", raw_obs["jq"][-1])
        time.sleep(sleep_time)

        if get_obs:
            obs = collections.OrderedDict()
            obs["qpos"] = list(raw_obs["jq"])
            obs["images"] = {}
            obs["images"]["0"] = raw_obs["img"][:, :, ::-1]
        else:
            obs = None
        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID,
            reward=self.get_reward(),
            discount=None,
            observation=obs,
        )


def make_env(
    setup_robots=True,
    setup_base=False,
    record_images=True,
    robot_instance=None,
    cameras=None,
):
    env = MujocoEnv(robot_instance, cameras)
    return env
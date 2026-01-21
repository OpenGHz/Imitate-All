import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image, JointState
import collections

import torch
import numpy as np
import os, time, logging, pickle, inspect
from typing import Dict
from tqdm import tqdm
from utils.utils import set_seed, save_eval_results
from configurations.task_configs.config_tools.basic_configer import (
    basic_parser,
    get_all_config,
)
from policies.common.maker import make_policy
from envs.common_env import get_image
import dm_env
import cv2
from policy_evaluate import eval_parser, get_ckpt_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Evaluate(Node):
    def __init__(self):
        self.running = True
        Node.__init__(self, 'evaluate_node')

        self.obs=collections.OrderedDict()
        self.obs["qpos"] = []
        self.obs["images"] = {}
        
        # subscriber: image, qpos
        self.bridge = CvBridge()
        # qos_profile = QoSProfile(
        #     depth=1,  # 设置队列的深度
        #     reliability=ReliabilityPolicy.RELIABLE,  # 设置为可靠的传输
        #     history=HistoryPolicy.KEEP_LAST,  # 设置为KEEP_LAST
        #     durability=DurabilityPolicy.VOLATILE  # 设置为VOLATILE
        # )
        self.head_color_suber = self.create_subscription(Image, '/camera/head_camera/color/image_raw', self.head_color_callback, 1)
        self.head_depth_suber = self.create_subscription(Image, '/camera/head_camera/aligned_depth_to_color/image_raw', self.head_depth_callback, 2)
        self.left_color_suber = self.create_subscription(Image, '/camera/left_camera/color/image_raw', self.left_color_callback, 1)
        self.right_color_suber = self.create_subscription(Image, '/camera/right_camera/color/image_raw', self.right_color_callback, 1)
        self.joint_state_suber = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 1)

        # publisher: action
        self.cmd_vel_puber = self.create_publisher(Twist, '/cmd_vel', 1)
        self.spine_cmd_puber = self.create_publisher(Float64MultiArray, '/spine_forward_position_controller/commands', 1)
        self.head_cmd_puber = self.create_publisher(Float64MultiArray, '/head_forward_position_controller/commands', 1)
        self.left_arm_cmd_puber = self.create_publisher(Float64MultiArray, '/left_arm_forward_position_controller/commands', 1)
        self.right_arm_cmd_puber = self.create_publisher(Float64MultiArray, '/right_arm_forward_position_controller/commands', 1)

    def head_color_callback(self, msg):
        self.obs["images"]["0"] = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        # self.get_logger().info('Received head color image')

    def head_depth_callback(self, msg):
        # TODO
        self.current_depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono16')
        # self.get_logger().warn('Received head depth image, Not implemented!')

    def left_color_callback(self, msg):
        # img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.obs["images"]["1"] = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        # self.obs["images"]["0"] = img
        # self.get_logger().info('Received left color image')

    def right_color_callback(self, msg):
        self.obs["images"]["2"] = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        # self.get_logger().info('Received right color image')

    def joint_state_callback(self, msg):
        self.obs["qpos"] = list(msg.position)
        # self.get_logger().info(f'Received joint state: {len(self.obs["qpos"])}')

    def publish_action(self, action):
        # tctr_base
        twist_msg = Twist()
        twist_msg.linear.x = float(action[0])  # 线速度
        twist_msg.angular.z = float(action[1])  # 角速度
        self.cmd_vel_puber.publish(twist_msg)

        # tctr_slide
        spine_cmd_msg = Float64MultiArray()
        spine_cmd_msg.data = [float(val) for val in action[2:3]]  # 上半身目标位置 
        self.spine_cmd_puber.publish(spine_cmd_msg)

        # tctr_head
        head_cmd_msg = Float64MultiArray()
        head_cmd_msg.data = [float(val) for val in action[3:5]]  # 头部目标位置
        self.head_cmd_puber.publish(head_cmd_msg)

        # tctr_left_arm
        left_arm_cmd_msg = Float64MultiArray()
        left_arm_cmd_msg.data = [float(val) for val in action[5:12]]  # 左臂目标位置命令
        self.left_arm_cmd_puber.publish(left_arm_cmd_msg)

        # tctr_right_arm
        right_arm_cmd_msg = Float64MultiArray()
        right_arm_cmd_msg.data = [float(val) for val in action[12:19]]  # 右臂目标位置命令
        self.right_arm_cmd_puber.publish(right_arm_cmd_msg)


    def evaluate(self,args):
        all_config = get_all_config(args, "eval")
        set_seed(all_config["seed"])
        ckpt_names = all_config["ckpt_names"]

        results = []

        # multiple ckpt evaluation
        for ckpt_name in ckpt_names:
            success_rate, avg_return = self.eval_bc(all_config, ckpt_name)
            results.append([ckpt_name, success_rate, avg_return])

        for ckpt_name, success_rate, avg_return in results:
            logger.info(f"{ckpt_name}: {success_rate=} {avg_return=}")

    def eval_bc(self,config, ckpt_name):
        # TODO: eval only contains the logic, data flow and visualization
        # remove other not general processing code outside in policy and env maker
        # 显式获得配置
        ckpt_dir = config["ckpt_dir"]
        stats_path = config["stats_path"]
        save_dir = config["save_dir"]
        max_timesteps = config["max_timesteps"]
        camera_names = config["camera_names"]
        max_rollouts = config["num_rollouts"]
        policy_config: dict = config["policy_config"]
        state_dim = policy_config["state_dim"]
        action_dim = policy_config["action_dim"]
        temporal_agg = policy_config["temporal_agg"]
        num_queries = policy_config["num_queries"]  # i.e. chunk_size
        dt = 1 / config["fps"]
        image_mode = config.get("image_mode", 0)
        save_all = config.get("save_all", False)
        save_time_actions = config.get("save_time_actions", False)
        filter_type = config.get("filter", None)
        ensemble: dict = config.get("ensemble", None)
        save_dir = save_dir if save_dir != "AUTO" else ckpt_dir
        result_prefix = "result_" + ckpt_name.split(".")[0]
        debug = config.get("debug", False)
        if debug:
            logger.setLevel(logging.DEBUG)
            from utils.visualization.ros1_logger import LoggerROS1

            ros1_logger = LoggerROS1("eval_debuger")

        # TODO: remove this
        ckpt_path = get_ckpt_path(ckpt_dir, ckpt_name, stats_path)
        policy_config["ckpt_path"] = ckpt_path

        # make and configure policies
        policies: Dict[str, list] = {}
        if ensemble is None:
            logger.info("policy_config:", policy_config)
            # if ensemble is not None:
            policy_config["max_timesteps"] = max_timesteps  # TODO: remove this
            policy = make_policy(policy_config, "eval")
            policies["Group1"] = (policy,)
        else:
            logger.info("ensemble config:", ensemble)
            ensembler = ensemble.pop("ensembler")
            for gr_name, gr_cfgs in ensemble.items():
                policies[gr_name] = []
                for index, gr_cfg in enumerate(gr_cfgs):

                    policies[gr_name].append(
                        make_policy(
                            gr_cfg["policies"][index]["policy_class"],
                        )
                    )

        # add action filter
        # TODO: move to policy maker as wrappers
        if filter_type is not None:
            # init filter
            from OneEuroFilter import OneEuroFilter

            config = {
                "freq": config["fps"],  # Hz
                "mincutoff": 0.01,  # Hz
                "beta": 0.05,
                "dcutoff": 0.5,  # Hz
            }
            filters = [OneEuroFilter(**config) for _ in range(action_dim)]

        # init pre/post process functions
        # TODO: move to policy maker as wrappers
        use_stats = True
        if use_stats:
            with open(stats_path, "rb") as f:
                stats = pickle.load(f)
            pre_process = lambda s_qpos: (s_qpos - stats["qpos_mean"]) / stats["qpos_std"]
            post_process = lambda a: a * stats["action_std"] + stats["action_mean"]
        else:
            pre_process = lambda s_qpos: s_qpos
            post_process = lambda a: a

        showing_images = config.get("show_images", False)

        def show_images(ts):
            images: dict = ts.observation["images"]
            for name, value in images.items():
                # logger.info(f"Showing {name}: {value}...")
                cv2.imshow(name, value)
                # cv2.imwrite(f"{name}.png", value)
            cv2.waitKey(1)

        # evaluation loop
        if hasattr(policy, "eval"):
            policy.eval()
        env_max_reward = 0
        episode_returns = []
        highest_rewards = []
        num_rollouts = 0
        policy_sig = inspect.signature(policy).parameters
        prediction_freq = 100000
        for rollout_id in range(max_rollouts):

            # evaluation loop
            all_time_actions = torch.zeros(
                [max_timesteps, max_timesteps + num_queries, action_dim]
            ).cpu()

            qpos_history = torch.zeros((1, max_timesteps, state_dim)).cpu()
            image_list = []  # for visualization
            qpos_list = []
            action_list = []
            rewards = []
            with torch.inference_mode():
                if showing_images:
                    # must show enough times to clear the black screen
                    for _ in range(10):
                        show_images(ts)
                logger.info(f"Current rollout: {rollout_id} for {ckpt_name}.")
                v = input(f"Press Enter to start evaluation or z and Enter to exit...")
                if v == "z":
                    self.running = False
                    break

                #zshtodo
                self.ts = dm_env.TimeStep(
                    step_type=dm_env.StepType.FIRST,
                    reward=0,
                    discount=None,
                    observation=self.obs,
                )

                if hasattr(policy, "reset"):
                    policy.reset()
                try:
                    for t in tqdm(range(max_timesteps)):
                        start_time = time.time()
                        image_list.append(self.ts.observation["images"])
                        if showing_images:
                            show_images(self.ts)
                        curr_image = get_image(self.ts, camera_names, image_mode)
                        qpos_numpy = np.array(self.ts.observation["qpos"])

                        logger.debug(f"raw qpos: {qpos_numpy}")
                        qpos = pre_process(qpos_numpy)  # normalize qpos
                        logger.debug(f"pre qpos: {qpos}")
                        qpos = torch.from_numpy(qpos).float().cpu().unsqueeze(0)
                        qpos_history[:, t] = qpos

                        logger.debug(f"observe time: {time.time() - start_time}")
                        start_time = time.time()
                        # wrap policy
                        target_t = t % num_queries
                        if temporal_agg or target_t == 0:
                            # (1, chunk_size, 7) for act
                            all_actions: torch.Tensor = policy(qpos, curr_image)
                        all_time_actions[[t], t : t + num_queries] = all_actions
                        index = 0 if temporal_agg else target_t
                        raw_action = all_actions[:, index]

                        # post-process predicted action
                        # dim: (1,7) -> (7,)
                        raw_action = raw_action.squeeze(0).cpu().numpy()
                        logger.debug(f"raw action: {raw_action}")
                        action = post_process(raw_action)  # de-normalize action
                        # logger.debug(f"post action: {action}")
                        if filter_type is not None:  # filt action
                            for i, filter in enumerate(filters):
                                action[i] = filter(action[i], time.time())
                        # limit the prediction frequency
                        time.sleep(max(0, 1 / prediction_freq - (time.time() - start_time)))
                        logger.debug(f"prediction time: {time.time() - start_time}")
                        # step the environment
                        if debug:
                            # dt = 1
                            ros1_logger.log_1D("joint_position", list(qpos_numpy))
                            ros1_logger.log_1D("joint_action", list(action))
                            for name, image in self.ts.observation["images"].items():
                                ros1_logger.log_2D("image_" + name, image)
                        #zshtodo ?
                        # ts: dm_env.TimeStep = env.step(action, sleep_time=dt)
                        # publish action:
                        self.publish_action(action)

                        # for visualization
                        qpos_list.append(qpos_numpy)
                        action_list.append(action)
                        rewards.append(self.ts.reward)
                        # debug
                        # input(f"Press Enter to continue...")
                        # break
                except KeyboardInterrupt:
                    logger.info(f"Current roll out: {rollout_id} interrupted by CTRL+C...")
                    continue
                else:
                    num_rollouts += 1

            rewards = np.array(rewards)
            episode_return = np.sum(rewards[rewards != None])
            episode_returns.append(episode_return)
            episode_highest_reward = np.max(rewards)
            highest_rewards.append(episode_highest_reward)
            logger.info(
                f"Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}"
            )

            # saving evaluation results
            if save_dir != "":
                dataset_name = f"{result_prefix}_{rollout_id}"
                save_eval_results(
                    save_dir,
                    dataset_name,
                    rollout_id,
                    image_list,
                    qpos_list,
                    action_list,
                    camera_names,
                    dt,
                    all_time_actions,
                    save_all=save_all,
                    save_time_actions=save_time_actions,
                )
        if num_rollouts > 0:
            success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
            avg_return = np.mean(episode_returns)
            summary_str = (
                f"\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n"
            )
            for r in range(env_max_reward + 1):
                more_or_equal_r = (np.array(highest_rewards) >= r).sum()
                more_or_equal_r_rate = more_or_equal_r / num_rollouts
                summary_str += f"Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n"

            logger.info(summary_str)

            # save success rate to txt
            if save_dir != "":
                with open(os.path.join(save_dir, dataset_name + ".txt"), "w") as f:
                    f.write(summary_str)
                    f.write(repr(episode_returns))
                    f.write("\n\n")
                    f.write(repr(highest_rewards))
                logger.info(
                    f'Success rate and average return saved to {os.path.join(save_dir, dataset_name + ".txt")}'
                )
        else:
            success_rate = 0
            avg_return = 0
        if showing_images:
            cv2.destroyAllWindows()
        return success_rate, avg_return



if __name__ == "__main__":
    parser = basic_parser()
    eval_parser(parser)
    args = parser.parse_args()
    args_dict = vars(args)

    rclpy.init()
    np.set_printoptions(precision=3, suppress=True, linewidth=500)

    exec_node = Evaluate()
    spin_thread = threading.Thread(target=lambda:rclpy.spin(exec_node))
    spin_thread.start()

    while rclpy.ok() and exec_node.running:
        exec_node.evaluate(args_dict)

    exec_node.destroy_node()
    rclpy.shutdown()
    spin_thread.join()

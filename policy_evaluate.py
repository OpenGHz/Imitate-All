import torch
import numpy as np
import os, time, logging, pickle, inspect
from typing import Dict
from tqdm import tqdm
from utils import set_seed
from visualize_episodes import save_videos
from task_configs.config_tools.basic_configer import basic_parser, get_all_config
from policies.common.maker import make_policy
from envs.common_env import get_image, CommonEnv


def main(args):

    all_config = get_all_config(args, "eval")
    set_seed(all_config["seed"])
    ckpt_names = all_config["ckpt_names"]

    # make environment
    env_config = all_config["environments"]
    env_maker = env_config.pop("environment_maker")
    env = env_maker(all_config)  # use all_config for more flexibility

    results = []
    # multiple ckpt evaluation
    for ckpt_name in ckpt_names:
        success_rate, avg_return = eval_bc(all_config, ckpt_name, env)
        results.append([ckpt_name, success_rate, avg_return])

    for ckpt_name, success_rate, avg_return in results:
        logging.info(f"{ckpt_name}: {success_rate=} {avg_return=}")

    print()


def get_ckpt_path(ckpt_dir, ckpt_name, stats_path):
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    raw_ckpt_path = ckpt_path
    if not os.path.exists(ckpt_path):
        ckpt_dir = os.path.dirname(ckpt_dir)  # check the upper dir
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        logging.warning(f"Warning: not found ckpt_path: {raw_ckpt_path}, try {ckpt_path}...")
        if not os.path.exists(ckpt_path):
            ckpt_dir = os.path.dirname(stats_path)
            ckpt_path = os.path.join(ckpt_dir, ckpt_name)
            logging.warning(f"Warning: also not found ckpt_path: {ckpt_path}, try {ckpt_path}...")
    return ckpt_path


def eval_bc(config, ckpt_name, env: CommonEnv):
    # TODO: eval only contains the logic, data flow and visualization
    # remove other not general processing code outside in policy and env maker
    # 显式获得配置
    ckpt_dir = config["ckpt_dir"]

    stats_path = config["stats_path"]

    save_dir = config["save_dir"]
    max_timesteps = config["max_timesteps"]
    camera_names = config["camera_names"]
    num_rollouts = config["num_rollouts"]
    policy_config: dict = config["policy_config"]
    state_dim = policy_config["state_dim"]
    action_dim = policy_config["action_dim"]
    temporal_agg = policy_config["temporal_agg"]
    num_queries = policy_config["num_queries"]  # i.e. chunk_size
    dt = 1 / config["fps"]
    image_mode = config.get("image_mode", 0)
    arm_velocity = config.get("arm_velocity", 6)
    save_all = config.get("save_all", False)
    save_time_actions = config.get("save_time_actions", False)
    filter_type = config.get("filter", None)
    ensemble: dict = config.get("ensemble", None)
    save_dir = save_dir if save_dir != "AUTO" else ckpt_dir
    result_prefix = "result_" + ckpt_name.split(".")[0]

    # TODO: remove this
    ckpt_path = get_ckpt_path(ckpt_dir, ckpt_name, stats_path)
    policy_config["ckpt_path"] = ckpt_path
    policy_config["stats_path"] = stats_path

    if "ckpt_dir" in config and "stats_path" in config:
        print(f"the number of model is : {len(config['ckpt_dir_n'])}")
        for i in range(len(config["ckpt_dir_n"])):
            ckpt_dir_n = config["ckpt_dir_n"][i]
            stats_path_n = config["stats_path_n"][i]
            ckpt_path = get_ckpt_path(ckpt_dir_n, ckpt_name, stats_path_n)
            policy_config[f"ckpt_path_{i}"] = ckpt_path

    # make and configure policies
    policies: Dict[str, list] = {}
    logging.info("policy_config:", policy_config)
    # if ensemble is None:
    policy_config["max_timesteps"] = max_timesteps  # TODO: remove this
    policy = make_policy(policy_config, "eval")   
    #move to policy maker 

    # if ensemble is None:
    #     logging.info("policy_config:", policy_config)
    #     # if ensemble is None:
    #     policy_config["max_timesteps"] = max_timesteps  # TODO: remove this
    #     policy = make_policy(policy_config, "eval")
    #     policies["Group1"] = (policy,)
    # else:
    #     logging.info("ensemble config:", ensemble)
    #     ensembler = ensemble.pop("ensembler")
    #     for gr_name, gr_cfgs in ensemble.items():
    #         policies[gr_name] = []
    #         for index, gr_cfg in enumerate(gr_cfgs):

    #             policies[gr_name].append(
    #                 make_policy(
    #                     gr_cfg["policies"][index]["policy_class"],
    #                 )
    #             )

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

    # evaluation loop
    if hasattr(policy, "eval"): policy.eval() #method from torch.nn.Module
    env_max_reward = 0
    episode_returns = []
    highest_rewards = []
    policy_sig = inspect.signature(policy).parameters
    for rollout_id in range(num_rollouts): #start one evaluation

        # evaluation loop
        all_time_actions = torch.zeros(
            [max_timesteps, max_timesteps + num_queries, action_dim]
        ).cuda()

        qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
        all_time_stage = []
        image_list = []  # for visualization
        qpos_list = []
        action_list = []
        rewards = []
        with torch.inference_mode():#禁用梯度计算
            logging.info("Reset environment...")
            env.reset(sleep_time=1)
            logging.info(f"Current rollout: {rollout_id} for {ckpt_name}.")
            v = input(f"Press Enter to start evaluation or z and Enter to exit...")
            if v == "z":
                break
            ts = env.reset()
            if hasattr(policy, "reset"): policy.reset()
            try:    #start evaluation
                for t in tqdm(range(max_timesteps)):
                    start_time = time.time()
                    image_list.append(ts.observation["images"])

                    # pre-process current observations
                    curr_image = get_image(ts, camera_names, image_mode)
                    qpos_numpy = np.array(ts.observation["qpos"])
                    # debug
                    # qpos_numpy = np.array(
                    #     [
                    #         -0.000190738,
                    #         -0.766194,
                    #         0.702869,
                    #         1.53601,
                    #         -0.964942,
                    #         -1.57607,
                    #         1.01381,
                    #     ]
                    # )
                    logging.debug(f"raw qpos: {qpos_numpy}")
                    qpos = pre_process(qpos_numpy)  # normalize qpos
                    logging.debug(f"pre qpos: {qpos}")
                    qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                    qpos_history[:, t] = qpos

                    # wrap policy
                    target_t = t % num_queries
                    if temporal_agg or target_t == 0:
                        #start = time.time()
                        # (1, chunk_size, 7) for act
                        # (1, chunk_size, 8) for with stage
                        all_actions: torch.Tensor = policy(qpos, curr_image)
                        #print(f"prediction time: {time.time() - start}")
                    all_time_actions[[t], t : t + num_queries] = all_actions
                    index = 0 if temporal_agg else target_t
                    raw_action = all_actions[:, index]
                    # post-process predicted action
                    # dim: (1,7) -> (7,)
                    raw_action = (
                        raw_action.squeeze(0).cpu().numpy()
                    )  
                    logging.debug(f"raw action: {raw_action}")
                    action = post_process(raw_action)  # de-normalize action
                    logging.debug(f"post action: {action}")
                    if filter_type is not None:  # filt action
                        for i, filter in enumerate(filters):
                            action[i] = filter(action[i], time.time())
                    #print("The predict stage is", action[-1])
                    # step the environment
                    sleep_time = max(0, dt - (time.time() - start_time))
                    ts = env.step(action, sleep_time=sleep_time, arm_vel=arm_velocity)

                    # for visualization
                    qpos_list.append(qpos_numpy)
                    action_list.append(action)
                    rewards.append(ts.reward)
                    # debug
                    # input(f"Press Enter to continue...")
                    # break
            except KeyboardInterrupt as e:
                logging.error(e)
                logging.error("Evaluation interrupted by user...")
                num_rollouts = rollout_id
                break
            else:
                pass

        rewards = np.array(rewards)
        episode_return = np.sum(rewards[rewards != None])
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)
        logging.info(
            f"Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}"
        )

        # saving evaluation results
        if save_dir != "":
            dataset_name = f"{result_prefix}_{rollout_id}"
            save_path = os.path.join(save_dir, dataset_name)
            if not os.path.isdir(save_dir):
                logging.info(f"Create directory for saving evaluation info: {save_dir}")
                os.makedirs(save_dir)
            save_videos(image_list, dt, video_path=f"{save_path}.mp4")
            if save_time_actions:
                np.save(f"{save_path}.npy", all_time_actions.cpu().numpy())
            if save_all:
                start_time = time.time()
                logging.info(f"Save all data to {save_path}...")
                # # save qpos
                # with open(os.path.join(save_dir, f'qpos_{rollout_id}.pkl'), 'wb') as f:
                #     pickle.dump(qpos_list, f)
                # # save actions
                # with open(os.path.join(save_dir, f'actions_{rollout_id}.pkl'), 'wb') as f:
                #     pickle.dump(action_list, f)
                # save as hdf5
                data_dict = {
                    "/observations/qpos": qpos_list,
                    "/action": action_list,
                }
                image_dict: Dict[str, list] = {}
                for cam_name in camera_names:
                    image_dict[f"/observations/images/{cam_name}"] = []
                for frame in image_list:
                    for cam_name in camera_names:
                        image_dict[f"/observations/images/{cam_name}"].append(
                            frame[cam_name]
                        )
                mid_time = time.time()
                from data_process.convert_all import compress_images, save_dict_to_hdf5

                image_dict = compress_images(image_dict)
                data_dict.update(image_dict)
                save_dict_to_hdf5(data_dict, save_path, False)
                end_time = time.time()
                logging.info(
                    f"{dataset_name}: construct data {mid_time - start_time} s and save data {end_time - mid_time} s"
                )

    success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
    avg_return = np.mean(episode_returns)
    summary_str = f"\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n"
    for r in range(env_max_reward + 1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / num_rollouts
        summary_str += f"Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n"

    logging.info(summary_str)

    # save success rate to txt
    if save_dir != "" and rollout_id > 0:
        with open(os.path.join(save_dir, dataset_name + ".txt"), "w") as f:
            f.write(summary_str)
            f.write(repr(episode_returns))
            f.write("\n\n")
            f.write(repr(highest_rewards))
        logging.info(
            f'Success rate and average return saved to {os.path.join(save_dir, dataset_name + ".txt")}'
        )
    # print("Stopping image recorder...")
    # env.image_recorder.close()
    return success_rate, avg_return


if __name__ == "__main__":

    parser = basic_parser()

    # change roll out num
    parser.add_argument(
        "-nr",
        "--num_rollouts",
        action="store",
        type=int,
        help="Maximum number of evaluation rollouts",
        required=False,
    )
    # change max time steps
    parser.add_argument(
        "-mts",
        "--max_timesteps",
        action="store",
        type=int,
        help="max_timesteps",
        required=False,
    )
    # robot config #TODO: move to robot config
    parser.add_argument(
        "-can",
        "--can_buses",
        action="store",
        nargs="+",
        type=str,
        help="can_bus",
        default=("can0", "can1"),
        required=False,
    )
    parser.add_argument(
        "-rn",
        "--robot_name",
        action="store",
        type=str,
        help="robot_name",
        required=False,
    )
    parser.add_argument(
        "-em",
        "--eef_mode",
        action="store",
        nargs="+",
        type=str,
        help="eef_mode",
        default=("gripper", "gripper"),
    )
    parser.add_argument(
        "-bat",
        "--bigarm_type",
        action="store",
        nargs="+",
        type=str,
        help="bigarm_type",
        default=("OD", "OD"),
    )
    parser.add_argument(
        "-fat",
        "--forearm_type",
        action="store",
        nargs="+",
        type=str,
        help="forearm_type",
        default=("DM", "DM"),
    )
    parser.add_argument(
        "-ci",
        "--camera_indices",
        action="store",
        nargs="+",
        type=str,
        help="camera_indices",
        default=("0",),
    )
    parser.add_argument(
        "-av",
        "--arm_velocity",
        action="store",
        type=float,
        help="arm_velocity",
        required=False,
    )
    # habitat TODO: remove this
    parser.add_argument(
        "-res",
        "--habitat",
        action="store",
        type=str,
        help="habitat",
        required=False,
    )
    # check_images
    parser.add_argument("-cki", "--check_images", action="store_true")
    # set time_stamp
    parser.add_argument(
        "-ts",
        "--time_stamp",
        action="store",
        nargs="+", #直接通过空格分隔的多个参数
        type=str,
        help="time_stamp",
        required=False,
    )
    # parser.add_argument(
    #     "-ts_1",
    #     "--time_stamp_1",
    #     action="store",
    #     type=str,
    #     help="time_stamp_1",
    #     required=False,
    # )

    # save
    parser.add_argument(
        "-sd", "--save_dir", action="store", type=str, help="save_dir", required=False
    )
    parser.add_argument("-sa", "--save_all", action="store_true", help="save_all")
    parser.add_argument(
        "-sta", "--save_time_actions", action="store_true", help="save_time_actions"
    )
    # action filter type TODO: move to post process; and will use obs filter?
    parser.add_argument(
        "-ft", "--filter", action="store", type=str, help="filter_type", required=False
    )
    # environment
    parser.add_argument('-et', "--environment", action='store', type=str, help='environment', required=False)
    args = parser.parse_args()
    args_dict = vars(args)
    # TODO: put unknown key-value pairs into args_dict
    # unknown = vars(unknown)
    # args.update(unknown)
    # print(unknown)
    main(args_dict)

import torch
import numpy as np
import os, time, logging, pickle, inspect
from typing import Dict
from tqdm import tqdm
from utils import set_seed, save_eval_results
from task_configs.config_tools.basic_configer import basic_parser, get_all_config
from policies.common.maker import make_policy
from envs.common_env import get_image, CommonEnv
from threading import Thread, Event
from policies.common.wrapper import TemporalEnsemblingWithDroppedActions as TEDA


logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)


def main(args):

    all_config = get_all_config(args, "eval")
    set_seed(all_config["seed"])
    ckpt_names = all_config["ckpt_names"]

    # make environment
    env_config = all_config["environments"]
    env_maker = env_config.pop("environment_maker")
    env = env_maker(all_config)  # use all_config for more flexibility
    assert env is not None, "Environment is not created..."

    results = []
    # multiple ckpt evaluation
    for ckpt_name in ckpt_names:
        success_rate, avg_return = eval_bc(all_config, ckpt_name, env)
        results.append([ckpt_name, success_rate, avg_return])

    for ckpt_name, success_rate, avg_return in results:
        logger.info(f"{ckpt_name}: {success_rate=} {avg_return=}")

    print()


def get_ckpt_path(ckpt_dir, ckpt_name, stats_path):
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    raw_ckpt_path = ckpt_path
    if not os.path.exists(ckpt_path):
        ckpt_dir = os.path.dirname(ckpt_dir)  # check the upper dir
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        logger.warning(
            f"Warning: not found ckpt_path: {raw_ckpt_path}, try {ckpt_path}..."
        )
        if not os.path.exists(ckpt_path):
            ckpt_dir = os.path.dirname(stats_path)
            ckpt_path = os.path.join(ckpt_dir, ckpt_name)
            logger.warning(
                f"Warning: also not found ckpt_path: {ckpt_path}, try {ckpt_path}..."
            )
    return ckpt_path


def eval_bc(config, ckpt_name, env: CommonEnv):
    # TODO: eval only contains the logic, data flow and visualization
    # remove other not general processing code outside in policy and env maker
    # conver this to a class
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

    # evaluation loop
    if hasattr(policy, "eval"):
        policy.eval()
    env_max_reward = 0
    episode_returns = []
    highest_rewards = []
    num_rollouts = 0
    action_freq = config["fps"]
    prediction_freq = 10  # TODO:config this
    chunk_size = num_queries
    teda = TEDA(
        chunk_size=chunk_size,
        action_dim=action_dim,
        max_timesteps=max_timesteps,
        dropped_num=TEDA.get_dropped_action_num(
            action_freq=action_freq, predict_freq=prediction_freq
        ),
    )

    infer_event = Event()
    for rollout_id in range(max_rollouts):
        infer_event.clear()
        act_step = 0
        next_rollout = False
        keyboard_interrupt = False

        all_time_actions = teda.get_action_buffer()
        teda.reset()

        qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
        image_list = []  # for visualization
        qpos_list = []
        action_list = []
        rewards = []

        logger.info("Reset environment...")
        env.reset(sleep_time=1)
        logger.info(f"Current rollout: {rollout_id} for {ckpt_name}.")
        v = input(f"Press Enter to start evaluation or z and Enter to exit...")
        if v == "z":
            return None, None
        ts = env.reset()

        def inference():
            global keyboard_interrupt
            # evaluation loop
            with torch.inference_mode():
                if hasattr(policy, "reset"):
                    policy.reset()
                try:
                    for t in tqdm(range(teda.prediction_step_max)):
                        infer_event.wait()
                        start_time = time.time()
                        if next_rollout:
                            break
                        image_list.append(ts.observation["images"])
                        # pre-process current observations
                        curr_image = get_image(ts, camera_names, image_mode)
                        qpos_numpy = np.array(ts.observation["qpos"])
                        logger.debug(f"raw qpos: {qpos_numpy}")
                        qpos = pre_process(qpos_numpy)  # normalize qpos
                        logger.debug(f"pre qpos: {qpos}")
                        qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)

                        # (1, chunk_size, 7) for act
                        all_actions: torch.Tensor = policy(qpos, curr_image)
                        # logger.warning(f"all_actions:{all_actions}")
                        # t is the infer_t
                        all_time_actions[[t], act_step : act_step + chunk_size] = (
                            all_actions
                        )

                        time.sleep(
                            max(0, 1 / prediction_freq - (time.time() - start_time))
                        )
                        infer_event.clear()
                except KeyboardInterrupt:
                    logger.info(
                        f"Current roll out: {rollout_id} interrupted by CTRL+C..."
                    )
                    keyboard_interrupt = True
                    return

        # start inference thread
        infer_thead = Thread(target=inference, daemon=True)
        infer_thead.start()
        last_not_done = set()
        for t in tqdm(range(max_timesteps)):
            start_time = time.time()
            act_step = t
            if keyboard_interrupt:
                break
            if teda.need_infer():
                while infer_event.is_set():
                    print("wait for last infer")
                    logger.debug("last not done")
                    last_not_done.add(t)
                    time.sleep(0.001)
                infer_event.set()
                if t == 0:
                    logger.debug("wait for first infer")
                    while infer_event.is_set():
                        time.sleep(0.001)
            raw_action = teda.update(all_time_actions)

            # post-process predicted action
            # dim: (1,7) -> (7,)
            raw_action = raw_action.squeeze(0).cpu().numpy()
            logger.debug(f"raw action: {raw_action}")
            action = post_process(raw_action)  # de-normalize action
            logger.debug(f"post action: {action}")

            # input("Press any key to continue")

            if filter_type is not None:  # filt action
                for i, filter in enumerate(filters):
                    action[i] = filter(action[i], time.time())

            # step the environment
            sleep_time = max(0, dt - (time.time() - start_time))
            ts = env.step(action, sleep_time=sleep_time, arm_vel=arm_velocity)

            # for visualization
            qpos = np.array(ts.observation["qpos"])
            qpos_history[:, t] = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
            qpos_list.append(qpos)
            action_list.append(action)
            rewards.append(ts.reward)
        else:
            # if not interrupted
            num_rollouts += 1

        # evaluation results
        rewards = np.array(rewards)
        episode_return = np.sum(rewards[rewards != None])
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)
        logger.info(
            f"Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}"
        )

        # saving evaluation results
        # TODO: configure what to save
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

        next_rollout = True
        infer_event.set()
        logger.debug(f"{last_not_done}")
        logger.debug("exiting current rollout inference")
        infer_thead.join()
        logger.debug("done")

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
        type=str,
        help="time_stamp",
        required=False,
    )
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

    args = parser.parse_args()
    args_dict = vars(args)
    # TODO: put unknown key-value pairs into args_dict
    # unknown = vars(unknown)
    # args.update(unknown)
    # print(unknown)
    main(args_dict)

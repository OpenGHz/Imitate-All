import torch
import numpy as np
import os
import pickle
import time
from typing import List, Dict
from tqdm import tqdm

from utils import set_seed
from visualize_episodes import save_videos
from policyer import make_policy, parser_common, parser_add_ACT, get_all_config
from envs.common_env import get_image, CommonEnv
from robots.custom_robot import AssembledRobot


def main(args):
    set_seed(1)

    all_config = get_all_config(args, "eval")
    ckpt_names = all_config['ckpt_names']
    robot_name = all_config['robot_name'] if not args["check_images"] else "fake_robot"
    eef_mode = all_config['eef_mode']
    bigarm_type = args['bigarm_type']
    forearm_type = args['forearm_type']
    fps = all_config['fps']
    start_joint = all_config['start_joint']
    joint_num = all_config['joint_num']
    robot_num = all_config['robot_num']
    can_list = args['can_buses']

    # get start joint
    print("stats_path:", all_config['stats_path'])
    if isinstance(start_joint, str) and start_joint == "AUTO":
        info_dir = os.path.dirname(all_config['stats_path'])
        key_info_path = os.path.join(info_dir, "key_info.pkl")
        with open(key_info_path, 'rb') as f:
            key_info = pickle.load(f)
            start_joint = key_info['init_info']["init_joint"]
    if len(start_joint) == joint_num:
        start_joint = start_joint * robot_num
    assert len(start_joint) == robot_num * joint_num, "The length of start_joint should be equal to joint_num or joint_num*robot_num"

    # init robots (sim and real are the same for airbot_play)
    robot_instances:List[AssembledRobot] = []
    if "airbot_play" in robot_name:
        # set up can
        # from utils import CAN_Tools
        import airbot

        vel = 2.0
        for i in range(robot_num):
            # if 'v' not in can:
            #     if not CAN_Tools.check_can_status(can):
            #        success, error = CAN_Tools.activate_can_interface(can, 1000000)
            #        if not success: raise Exception(error)
            airbot_player = airbot.create_agent("down", can_list[i], vel, eef_mode[i], bigarm_type[i], forearm_type[i])
            robot_instances.append(AssembledRobot(airbot_player, 1/fps, 
                                                  start_joint[joint_num*i:joint_num*(i+1)]))
    elif "fake" in robot_name or "none" in robot_name:
        from robots.custom_robot import AssembledFakeRobot
        if args["check_images"]:
            AssembledFakeRobot.real_camera = True
        for i in range(robot_num):
            robot_instances.append(AssembledFakeRobot(1/fps, start_joint[joint_num*i:joint_num*(i+1)]))
    elif "ros" in robot_name:
        from robots.custom_robot import AssembledRosRobot
        import rospy
        rospy.init_node("replay_episodes")
        namespace = "/airbot_play"
        states_topic = f"{namespace}/joint_states"
        arm_action_topic = f"{namespace}/arm_group_position_controller/command"
        gripper_action_topic = f"{namespace}/gripper_group_position_controller/command"
        for i in range(robot_num):
            robot_instances.append(
                AssembledRosRobot(
                    states_topic,
                    arm_action_topic,
                    gripper_action_topic,
                    joint_num,
                    start_joint[joint_num*i:joint_num*(i+1)],
                    1/fps,
                )
            )
    elif "mmk" in robot_name:
        from robots.custom_robot import AssembledMmkRobot
        for i in range(robot_num):
            robot_instances.append(AssembledMmkRobot())
    else:
        raise NotImplementedError(f"{robot_name} is not implemented")
    time.sleep(2)
    if args['go_zero']:
        print("Going home...")
        for robot in robot_instances:
            robot.set_joint_position_target([0, 0, 0, 0, 0, 0], [1], blocking=True)
    else:
        # load environment
        # TODO: the robots and environment should be independent
        # we should combine the robot and environment instead of passing the robot to the environment
        # so what's the name of the combination?
        environment = all_config['environment']
        if isinstance(environment, str):
            if environment == "real":
                if "airbot_play" in robot_name:
                    from envs.airbot_play_real_env import make_env
                elif "fake" in robot_name:
                    # from airbot_play_fake_env import make_env  # TODO: implement this or pass some param to make_env
                    from envs.airbot_play_real_env import make_env
                elif "ros" in robot_name:
                    from envs.airbot_play_real_env import make_env
                elif "mmk" in robot_name:
                    from envs.airbot_mmk_env import make_env
            elif environment == "mujoco":
                from envs.airbot_play_mujoco_env import make_env
            elif environment == "isaac":
                raise NotImplementedError
            else:
                raise NotImplementedError
            camera_names = all_config['camera_names']
            camera_indices = all_config['camera_indices']
            if camera_indices != "":
                cameras = {name: int(index) for name, index in zip(camera_names, camera_indices)}
            else:
                cameras = camera_names
            env = make_env(robot_instance=robot_instances, cameras=cameras)
        else:
            env = environment
        env.set_reset_position(start_joint)
        results = []
        # multiple ckpt evaluation
        for ckpt_name in ckpt_names:
            success_rate, avg_return = eval_bc(all_config, ckpt_name, env)
            results.append([ckpt_name, success_rate, avg_return])

        for ckpt_name, success_rate, avg_return in results:
            print(f'{ckpt_name}: {success_rate=} {avg_return=}')
    # 为保证安全性，退出时将机械臂归零
    if "mmk" not in robot_name:
        for robot in robot_instances:
            robot.set_joint_position_target([0, 0, 0, 0, 0, 0], [1], blocking=True)
        time.sleep(2)
        for robot in robot_instances:
            del robot
    print()

def eval_bc(config, ckpt_name, env:CommonEnv):
    set_seed(1000)
    # 显式获得配置
    ckpt_dir = config['ckpt_dir']
    stats_path = config['stats_path']
    save_dir = config['save_dir']
    max_timesteps = config['max_timesteps']
    camera_names = config['camera_names']
    num_rollouts = config['num_rollouts']
    policy_config:dict = config['policy_config']
    state_dim = policy_config['state_dim']
    action_dim = policy_config['action_dim']
    temporal_agg = policy_config['temporal_agg']
    num_queries = policy_config['num_queries']  # i.e. chunk_size
    dt = 1 / config['fps']
    image_mode = config.get('image_mode', 0)
    qpos_mode = config.get('qpos_mode', 0)
    arm_velocity = config.get('arm_velocity', 6)
    save_all = config.get('save_all', False)
    save_time_actions = config.get('save_time_actions', False)
    filter_type = config.get('filter', None)
    ensemble:dict = config.get('ensemble', None)
    save_dir = save_dir if save_dir != "AUTO" else ckpt_dir
    result_prefix = "result_" + ckpt_name.split('.')[0]

    # load policies
    policies:Dict[str, list] = {}
    if ensemble is None:
        print("policy_config:", policy_config)
        # if ensemble is not None:
        policy = make_policy(policy_config)
        policies['Group1'] = (policy,)
    else:
        print("ensemble config:", ensemble)
        ensembler = ensemble.pop("ensembler")
        for gr_name, gr_cfgs in ensemble.items():
            policies[gr_name] = []
            for index, gr_cfg in enumerate(gr_cfgs):

                policies[gr_name].append(make_policy(
                    gr_cfg["policies"][index]["policy_class"],

                ))

    # check the existence of ckpt
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    raw_ckpt_path = ckpt_path
    use_stats = stats_path not in ["", None]
    if not os.path.exists(ckpt_path):
        ckpt_dir = os.path.dirname(ckpt_dir)  # check the upper dir
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        print(f"Warning: not found ckpt_path: {raw_ckpt_path}, try {ckpt_path}...")
        if not os.path.exists(ckpt_path):
            ckpt_path = None
            if use_stats:
                ckpt_dir = os.path.dirname(stats_path)
                ckpt_path = os.path.join(ckpt_dir, ckpt_name)
                print(f"Warning: also not found ckpt_path: {ckpt_path}, try {ckpt_path}...")
                if not os.path.exists(ckpt_path):
                    ckpt_path = None
            if ckpt_path is None:
                print(f"Warning: also not found ckpt_path: {ckpt_path}, assume not using ckpt")

    # configure policy
    if hasattr(policy, 'load_state_dict'):
        assert ckpt_path is not None, "ckpt_path must exist for loading policy"
        # TODO: all policies should load the ckpt (policy maker should return a class)
        loading_status = policy.load_state_dict(torch.load(ckpt_path))
        print(loading_status)
        print(f'Loaded: {ckpt_path}')
    if hasattr(policy, 'cuda'):
        policy.cuda()
        policy.eval()

    # add action filter
    if filter_type is not None:
        # init filter
        from OneEuroFilter import OneEuroFilter
        config = {
            "freq": config['fps'],  # Hz
            "mincutoff": 0.01,  # Hz
            "beta": 0.05,
            "dcutoff": 0.5,  # Hz
        }
        filters = [OneEuroFilter(**config) for _ in range(action_dim)]

    # init pre/post process functions
    if use_stats:
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)
        pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
        post_process = lambda a: a * stats['action_std'] + stats['action_mean']
    else:
        pre_process = lambda s_qpos: s_qpos
        post_process = lambda a: a

    # evaluation loop
    query_frequency = 1 if temporal_agg else num_queries
    env_max_reward = 0
    episode_returns = []
    highest_rewards = []
    for rollout_id in range(num_rollouts):

        # evaluation loop
        if temporal_agg:
            all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, action_dim]).cuda()

        qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
        image_list = []  # for visualization
        qpos_list = []
        action_list = []
        rewards = []
        with torch.inference_mode():
            print('Reset environment...')
            env.reset(sleep_time=1)
            print(f"Current rollout: {rollout_id} for {ckpt_name}.")
            v = input(f'Press Enter to start evaluation or z and Enter to exit...')
            if v == 'z':
                break
            ts = env.reset()
            try:
                for t in tqdm(range(max_timesteps)):
                    start_time = time.time()

                    # process previous timestep to get qpos and image_list
                    obs = ts.observation
                    image_list.append(obs['images'])
                    qpos_numpy = np.array(obs['qpos'])
                    qpos = pre_process(qpos_numpy)  # normalize qpos
                    qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                    qpos_history[:, t] = qpos
                    curr_image = get_image(ts, camera_names, image_mode)

                    # query policy
                    if config['policy_class'] == "ACT":
                        if t % query_frequency == 0:
                            all_actions = policy(qpos, curr_image)  # (1, chunk_size, 7)
                        # smooth
                        if temporal_agg:
                            all_time_actions[[t], t:t+num_queries] = all_actions
                            actions_for_curr_step = all_time_actions[:, t]
                            actions_populated = torch.all(actions_for_curr_step != 0, axis=1)  # ??
                            actions_for_curr_step = actions_for_curr_step[actions_populated]
                            k = 0.01
                            exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                            exp_weights = exp_weights / exp_weights.sum()
                            exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                            raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                        else:
                            raw_action = all_actions[:, t % query_frequency]
                    else:
                        if qpos_mode == 1:  # TODO
                            qpos = torch.from_numpy(qpos_numpy[np.newaxis, :]).float()
                        raw_action = policy(qpos, curr_image)

                    # post-process actions
                    try:
                        raw_action = raw_action.squeeze(0).cpu().numpy()
                    except:
                        pass
                    action = post_process(raw_action)  # de-normalize action
                    # filter
                    if filter_type is not None:
                        for i, filter in enumerate(filters):
                            action[i] = filter(action[i], time.time())
                    # step the environment
                    sleep_time = max(0, dt - (time.time() - start_time))
                    ts = env.step(action, sleep_time=sleep_time, arm_vel=arm_velocity)

                    # for visualization
                    qpos_list.append(qpos_numpy)
                    action_list.append(action)
                    rewards.append(ts.reward)
            except KeyboardInterrupt as e:
                print(e)
                print('Evaluation interrupted by user...')
                num_rollouts = rollout_id
                break
            else:
                pass

        rewards = np.array(rewards)
        episode_return = np.sum(rewards[rewards!=None])
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)
        print(f'Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}')

        # saving evaluation results
        if save_dir != "":
            dataset_name = f'{result_prefix}_{rollout_id}'
            save_path = os.path.join(save_dir, dataset_name)
            if not os.path.isdir(save_dir):
                print(f'Create directory for saving evaluation info: {save_dir}')
                os.makedirs(save_dir)
            save_videos(image_list, dt, video_path=f'{save_path}.mp4')
            if save_time_actions:
                np.save(f"{save_path}.npy", all_time_actions.cpu().numpy())
            if save_all:
                start_time = time.time()
                print(f'Save all data to {save_path}...')
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
                image_dict:Dict[str, list] = {}
                for cam_name in camera_names:
                    image_dict[f"/observations/images/{cam_name}"] = []
                for frame in image_list:
                    for cam_name in camera_names:
                        image_dict[f"/observations/images/{cam_name}"].append(frame[cam_name])
                mid_time = time.time()
                from convert_all import compress_images, save_dict_to_hdf5
                image_dict = compress_images(image_dict)
                data_dict.update(image_dict)
                save_dict_to_hdf5(data_dict, save_path, False)
                end_time = time.time()
                print(
                    f"{dataset_name}: construct data {mid_time - start_time} s and save data {end_time - mid_time} s"
                )

    success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
    avg_return = np.mean(episode_returns)
    summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
    for r in range(env_max_reward+1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / num_rollouts
        summary_str += f'Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n'

    print(summary_str)

    # save success rate to txt
    if save_dir != "" and rollout_id > 0:
        with open(os.path.join(save_dir, dataset_name + '.txt'), 'w') as f:
            f.write(summary_str)
            f.write(repr(episode_returns))
            f.write('\n\n')
            f.write(repr(highest_rewards))
        print(f'Success rate and average return saved to {os.path.join(save_dir, dataset_name + ".txt")}')
    # print("Stopping image recorder...")
    # env.image_recorder.close()
    return success_rate, avg_return


if __name__ == '__main__':
    parser = parser_common()
    parser_add_ACT(parser)

    # change roll out num
    parser.add_argument('-nr', '--num_rollouts', action='store', type=int, help='Maximum number of evaluation rollouts', required=False)
    # change max time steps
    parser.add_argument('-mts', '--max_timesteps', action='store', type=int, help='max_timesteps', required=False)
    # just go zero
    parser.add_argument('-go', '--go_zero', action='store_true')
    # robot config
    parser.add_argument('-can', "--can_buses", action='store', nargs='+', type=str, help='can_bus', default=("can0", "can1"), required=False)
    parser.add_argument('-rn', "--robot_name", action='store', type=str, help='robot_name', required=False)
    parser.add_argument('-em', "--eef_mode", action='store', nargs='+', type=str, help='eef_mode', default=("gripper", "gripper"))
    parser.add_argument('-bat', "--bigarm_type", action='store', nargs='+', type=str, help='bigarm_type', default=("OD", "OD"))
    parser.add_argument('-fat', "--forearm_type", action='store', nargs='+', type=str, help='forearm_type', default=("DM", "DM"))
    parser.add_argument('-ci', "--camera_indices", action='store', nargs='+', type=str, help="camera_indices", default=("0",))
    # environment
    parser.add_argument('-et', "--environment", action='store', type=str, help='environment', required=False)
    # config path #TODO
    parser.add_argument('-cp', "--config_path", action='store', type=str, help='config_path', required=False)
    # save_episode
    parser.add_argument('-sd', "--save_dir", action='store', type=str, help='save_dir', required=False)
    # check_images
    parser.add_argument('-cki', "--check_images", action='store_true')
    # set time_stamp
    parser.add_argument("-ts", "--time_stamp", action="store", type=str, help="time_stamp", required=False)
    # set arm velocity
    parser.add_argument("-av", "--arm_velocity", action="store", type=float, help="arm_velocity", required=False)
    # save all
    parser.add_argument("-sa", "--save_all", action="store_true", help="save_all")
    parser.add_argument("-sta", "--save_time_actions", action="store_true", help="save_time_actions")
    # action filter type
    parser.add_argument("-ft", "--filter", action="store", type=str, help="filter_type", required=False)

    main(vars(parser.parse_args()))

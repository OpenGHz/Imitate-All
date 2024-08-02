import os
import time
import h5py
import numpy as np
import cv2


def print_dt_diagnosis(actual_dt_history):
    actual_dt_history = np.array(actual_dt_history)
    get_action_time = actual_dt_history[:, 1] - actual_dt_history[:, 0]
    step_env_time = actual_dt_history[:, 2] - actual_dt_history[:, 1]
    total_time = actual_dt_history[:, 2] - actual_dt_history[:, 0]

    dt_mean = np.mean(total_time)
    dt_std = np.std(total_time)
    freq_mean = 1 / dt_mean
    print(
        f"Avg freq: {freq_mean:.2f} Get action: {np.mean(get_action_time):.3f} Step env: {np.mean(step_env_time):.3f}"
    )
    return freq_mean


def save_one_episode(
    data,
    camera_names,
    dataset_dir,
    dataset_name,
    overwrite,
    no_base=True,
    no_effort=True,
    no_velocity=True,
    compress=True,
    states_num=None,
    slices=None,
):
    """
    Save one episode to hdf5 file. For each timestep(float64 for all data except images, which are uint8):

    Action space:      [left_arm_qpos (6),             # absolute joint position
                        left_gripper_positions (1),    # normalized gripper position (0: close, 1: open)
                        right_arm_qpos (6),            # absolute joint position
                        right_gripper_positions (1),]  # normalized gripper position (0: close, 1: open)

    Observation space: {"qpos": Concat[ left_arm_qpos (6),          # absolute joint position
                                        left_gripper_position (1),  # normalized gripper position (0: close, 1: open)
                                        right_arm_qpos (6),         # absolute joint position
                                        right_gripper_qpos (1)]     # normalized gripper position (0: close, 1: open)
                        "qvel": Concat[ left_arm_qvel (6),          # absolute joint velocity (rad)
                                        left_gripper_velocity (1),  # normalized gripper velocity (pos: opening, neg: closing)
                                        right_arm_qvel (6),         # absolute joint velocity (rad)
                                        right_gripper_qvel (1)]     # normalized gripper velocity (pos: opening, neg: closing)
                        "images": {"cam_0": (480x640x3),  # h, w, c, dtype='uint8'
                                   "cam_1": (480x640x3),  # h, w, c, dtype='uint8'
                                   "cam_2": (480x640x3),  # h, w, c, dtype='uint8'
                                   "cam_3": (480x640x3)}  # h, w, c, dtype='uint8'

    base_action: [base_linear_vel, base_angular_vel]  # base action which is the velocity of the base

    Args:
        data:
            can be list, tuple and other non-dict types(not recommended:
                data[0]: each element is a dict, containing the arm observation and base action data
                data[1]: each element is an array, containing the arm action data
                data[2]: each element is an array, containing the actual_dt_history data
            can also be dict(recommended):
                data['/observations/qpos']: list, containing the arm joint position data
                data['/observations/qvel']: list, containing the arm joint velocity data
                data['/observations/effort']: list, containing the arm joint effort data
                data['/observations/images/<camera_name>']: list, containing the camera image data
                data['/action']: list, containing the arm action data
                data['/base_action']: list, containing the base action data
                data['/base_action_t265']: list, containing the base action data for t265, not used temporarily
        camera_names: list, containing the names of the cameras
        dataset_dir: str, the directory to save the dataset (e.g. test_dataset_root/test_task_name)
        dataset_name: str, the name of the dataset (e.g. episode_0, episode_1, ..., which is a hdf5 file)
        overwrite: bool, whether to overwrite the existing dataset, default is True
        no_base: bool, whether to save the base action data, default is True
        no_effort: bool, whether to save the effort data, default is True
        no_velocity: bool, whether to save the velocity data, default is True
        compress: bool, whether to compress the image data, default is True
        states_num: int, the number of states, default is None, which will be set to the length of the first qpos data
        slices: list, containing the slices of the dataset, default is None, which will be set to [0, -1], not used temporarily
    Returns:
        bool, whether the dataset is saved successfully
    """

    # saving dataset
    print(f"Dataset name: {dataset_name}")
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    dataset_path = os.path.join(dataset_dir, dataset_name)
    if os.path.isfile(dataset_path) and not overwrite:
        print(
            f"Dataset already exist at \n{dataset_path}\nHint: set overwrite to True."
        )
        exit()
    """Data collection
    The main observation data are timesteps and actions, in addition to actual_dt_history:
    where timesteps is a list, each element is a dict, containing observation data
    actions is a list, each element is an array, containing action data
    actual_dt_history is a list, each element is an array, including the starting time, action time, and observation time of each timestep.
    """

    if not isinstance(data, dict):
        timesteps, actions, actual_dt_history = data
        max_timesteps = len(actions)
        assert (
            len(timesteps) - 1 == len(actions) == max_timesteps
        ), f"{len(timesteps)}, {len(actions)}, {max_timesteps}"
        print(
            f"Avg fps: {max_timesteps / (actual_dt_history[-1][-1] - actual_dt_history[0][0])}"
        )

        freq_mean = print_dt_diagnosis(actual_dt_history)
        if freq_mean < 30:
            print(
                f"\n\nfreq_mean is {freq_mean}, lower than 30, re-collecting... \n\n\n\n"
            )
            return False

        # TODO: change names easily
        data_dict = {
            "/observations/qpos": [],
            "/observations/qvel": [],
            "/observations/effort": [],
            "/action": [],
            "/base_action": [],
            # '/base_action_t265': [],
        }
        for cam_name in camera_names:
            data_dict[f"/observations/images/{cam_name}"] = []

        # len(action): max_timesteps, len(time_steps): max_timesteps + 1
        while actions:
            action = actions.pop(0)
            ts = timesteps.pop(0)
            data_dict["/observations/qpos"].append(ts["/observations/qpos"])
            data_dict["/observations/qvel"].append(ts["/observations/qpos"])
            data_dict["/observations/effort"].append(ts["/observations/effort"])
            data_dict["/action"].append(action)
            if not no_base:
                data_dict["/base_action"].append(ts["/base_action"])
                # data_dict['/base_action_t265'].append(ts['/base_action_t265'])
            for cam_name in camera_names:
                data_dict[f"/observations/images/{cam_name}"].append(
                    ts[f"/observations/images/{cam_name}"]
                )
    else:
        data_dict = data
        max_timesteps = len(data_dict["/action"])
        if camera_names is None:
            camera_names = [
                name.split("/")[-1] for name in data_dict.keys() if "images" in name
            ]
            print(f"camera_names: {camera_names}")

    # remove not used data
    if no_base:
        data_dict.pop("/base_action", None)
        data_dict.pop("/base_action_t265", None)
    if no_effort:
        data_dict.pop("/observations/effort", None)
    if no_velocity:
        data_dict.pop("/observations/qvel", None)

    # plot /base_action vs /base_action_t265
    # import matplotlib.pyplot as plt
    # plt.plot(np.array(data_dict['/base_action'])[:, 0], label='base_action_linear')
    # plt.plot(np.array(data_dict['/base_action'])[:, 1], label='base_action_angular')
    # plt.plot(np.array(data_dict['/base_action_t265'])[:, 0], '--', label='base_action_t265_linear')
    # plt.plot(np.array(data_dict['/base_action_t265'])[:, 1], '--', label='base_action_t265_angular')
    # plt.legend()
    # plt.savefig('record_episodes_vel_debug.png', dpi=300)

    if compress:
        # JPEG compression
        t0 = time.time()
        encode_param = [
            int(cv2.IMWRITE_JPEG_QUALITY),
            50,
        ]  # TODO: tried as low as 20, seems fine
        compressed_len = []
        for cam_name in camera_names:
            image_list = data_dict[f"/observations/images/{cam_name}"]
            compressed_list = []
            compressed_len.append([])
            for image in image_list:
                result, encoded_image = cv2.imencode(
                    ".jpg", image, encode_param
                )  # 0.02 sec # cv2.imdecode(encoded_image, 1)
                compressed_list.append(encoded_image)
                compressed_len[-1].append(len(encoded_image))
            data_dict[f"/observations/images/{cam_name}"] = compressed_list
        print(f"compression: {time.time() - t0:.2f}s")

        # pad so it has same length
        t0 = time.time()
        compressed_len = np.array(compressed_len)
        padded_size = compressed_len.max()
        for cam_name in camera_names:
            compressed_image_list = data_dict[f"/observations/images/{cam_name}"]
            padded_compressed_image_list = []
            for compressed_image in compressed_image_list:
                padded_compressed_image = np.zeros(padded_size, dtype="uint8")
                image_len = len(compressed_image)
                padded_compressed_image[:image_len] = compressed_image
                padded_compressed_image_list.append(padded_compressed_image)
            data_dict[f"/observations/images/{cam_name}"] = padded_compressed_image_list
        print(f"padding: {time.time() - t0:.2f}s")

    # HDF5
    t0 = time.time()
    with h5py.File(dataset_path + ".hdf5", "w", rdcc_nbytes=1024**2 * 2) as root:
        root.attrs["sim"] = False
        root.attrs["compress"] = compress
        obs = root.create_group("observations")
        image = obs.create_group("images")
        for cam_name in camera_names:
            if compress:
                _ = image.create_dataset(
                    cam_name,
                    (max_timesteps, padded_size),
                    dtype="uint8",
                    chunks=(1, padded_size),
                )
            else:
                _ = image.create_dataset(
                    cam_name,
                    (max_timesteps, 480, 640, 3),
                    dtype="uint8",
                    chunks=(1, 480, 640, 3),
                )
        if states_num is None:
            states_num = len(data_dict["/observations/qpos"][0])
            print(f"states_num: {states_num}")
        else:
            assert states_num == len(
                data_dict["/observations/qpos"][0]
            ), f'{states_num}, {len(data_dict["/observations/qpos"][0])}'

        _ = obs.create_dataset("qpos", (max_timesteps, states_num))
        if not no_velocity:
            _ = obs.create_dataset("qvel", (max_timesteps, states_num))
        if not no_effort:
            _ = obs.create_dataset("effort", (max_timesteps, states_num))
        _ = root.create_dataset("action", (max_timesteps, states_num))
        if not no_base:
            _ = root.create_dataset("base_action", (max_timesteps, 2))
            # _ = root.create_dataset('base_action_t265', (max_timesteps, 2))

        for name, array in data_dict.items():
            root[name][...] = array

        if compress:
            _ = root.create_dataset("compress_len", (len(camera_names), max_timesteps))
            root["/compress_len"][...] = compressed_len

    print(f"Saving: {time.time() - t0:.1f} secs")

    return True


def read_one_episode(
    camera_names,
    dataset_dir,
    dataset_name,
    compress=True,
    no_base=True,
    no_effort=True,
    no_velocity=True,
    show_info=False,
):
    if dataset_dir[-1] != "/":
        dataset_dir += "/"
    dataset_path = dataset_dir + dataset_name + ".hdf5"
    with h5py.File(dataset_path, "r") as root:
        actions = root["/action"][()]
        qpos = root["/observations/qpos"][()]
        images = {}
        for cam_name in camera_names:
            images[cam_name] = root[f"/observations/images/{cam_name}"][()]
            if compress:
                # JPEG decompression
                images_num = len(images[cam_name])
                decompressed_list = [0] * images_num
                for index, image in enumerate(images[cam_name]):
                    decompressed_list[index] = cv2.imdecode(image, 1)
                images[cam_name] = decompressed_list
        # other config
        base_actions = root["/base_action"][()] if not no_base else None
        qvel = root["/observations/qvel"][()] if not no_velocity else None
        effort = root["/observations/effort"][()] if not no_effort else None
    other_data = {"base_actions": base_actions, "qvel": qvel, "effort": effort}
    if show_info:
        print(f"qpos: {qpos.shape}, actions: {actions.shape}")
        for cam_name in camera_names:
            print(f"{cam_name}: {images[cam_name][0].shape}")
    return qpos, actions, images, other_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-dd",
        "--dataset_dir",
        type=str,
        help="The directory to save the dataset",
        default="./data/hdf5/wipe_water",
    )
    parser.add_argument(
        "-dn",
        "--dataset_name",
        type=str,
        help="The name of the dataset",
        default="episode_0",
    )
    parser.add_argument(
        "-cn",
        "--camera_names",
        type=str,
        help="Camera names",
        default="0,1,2",
    )
    parser.add_argument(
        "-ts",
        "--test_save",
        action="store_true",
        help="Test save_one_episode",
        default=False,
    )
    args, unknown = parser.parse_known_args()

    camera_names = args.camera_names.split(",")
    dataset_dir = args.dataset_dir
    dataset_name = args.dataset_name

    if args.test_save:
        ONE_SIDE = True
        # test save_one_episode
        states_num = 7 if ONE_SIDE else 14
        data_lenth = 100
        data = {
            "/observations/qpos": np.random.rand(data_lenth, states_num),
            "/observations/qvel": np.random.rand(data_lenth, states_num),
            "/observations/effort": np.random.rand(data_lenth, states_num),
            "/action": np.random.rand(data_lenth, states_num),
            "/base_action": np.random.rand(data_lenth, 2),
        }
        for name in camera_names:
            data[f"/observations/images/{name}"] = [
                np.random.randint(0, 255, (480, 640, 3), dtype="uint8")
            ] * data_lenth
        if ONE_SIDE:
            camera_names.pop()
            data.pop("/observations/images/cam_right_wrist")
        overwrite = True
        no_base = True
        no_effort = True
        no_velocity = True
        compress = True
        states_num = states_num
        slices = None
        save_one_episode(
            data,
            camera_names,
            dataset_dir,
            dataset_name,
            overwrite,
            no_base,
            no_effort,
            no_velocity,
            compress,
            states_num,
            slices,
        )
    else:
        print("Start reading...")
        # test read_one_episode
        qpos, actions, images, other_data = read_one_episode(
            camera_names,
            dataset_dir,
            dataset_name,
        )
        assert (
            qpos.shape[0] == actions.shape[0] == len(images[camera_names[0]])
        ), "The length of qpos, actions and images should be the same."
        for cam_name in camera_names:
            assert (
                images[camera_names[0]][0].shape == images[cam_name][0].shape
            ), f"The shape of images[{camera_names[0]}] and images[{cam_name}] should be the same."
        image_shape = images[camera_names[0]][0].shape
        image_size = f"{image_shape[1]}x{image_shape[0]}x{image_shape[2]}"
        print("  states_num:", qpos.shape[1])
        print("  image_size:", image_size)
        print("  episode_len:", qpos.shape[0])
        print("  start_joint:", actions[0])
        print("END")

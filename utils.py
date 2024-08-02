import numpy as np
import torch
import os
import h5py
from torch.utils.data import DataLoader
import fnmatch
import subprocess
import pickle
import re
from datetime import datetime
from typing import Tuple, List
import time
from threading import Thread, Event


class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        episode_ids,
        dataset_dir,
        camera_names,
        norm_stats,
        augmentors: dict = {},
        other_config: dict = {},
    ):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        self.other_config = other_config
        self.augmentors = augmentors
        self.augment_images = augmentors.get("image", None)
        self.__getitem__(0)  # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False  # hardcode
        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f"episode_{episode_id}.hdf5")
        with h5py.File(dataset_path, "r") as root:
            compressed = root.attrs.get("compress", False)
            original_action_shape = root["/action"].shape
            episode_len = original_action_shape[0]
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)
            # get observation at start_ts only
            qpos = root["/observations/qpos"][start_ts]
            if "qvel" in self.other_config:
                qvel = root["/observations/qvel"][start_ts]
            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f"/observations/images/{cam_name}"][
                    start_ts
                ]
            if compressed:
                for cam_name in image_dict.keys():
                    import cv2

                    decompressed_image = cv2.imdecode(image_dict[cam_name], 1)
                    image_dict[cam_name] = np.array(decompressed_image)
            # get all actions after and including start_ts
            is_sim = root.attrs["sim"]
            if is_sim:
                action = root["/action"][start_ts:]
                action_len = episode_len - start_ts
            else:
                action = root["/action"][
                    max(0, start_ts - 1) :
                ]  # hack, to make timesteps more aligned
                action_len = episode_len - max(
                    0, start_ts - 1
                )  # hack, to make timesteps more aligned

        self.is_sim = is_sim
        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(episode_len)
        is_pad[action_len:] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        # construct image metrix (the same as np.array)
        # each row is all images from one camera
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last LABEL
        image_data = torch.einsum("k h w c -> k c h w", image_data)

        # augment images
        if self.augment_images is not None and self.augment_images.activated:
            image_data = self.augment_images(image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0

        # normalize other data
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats[
            "action_std"
        ]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats[
            "qpos_std"
        ]

        return image_data, qpos_data, action_data, is_pad


def find_all_hdf5(dataset_dir, skip_mirrored_data=True):
    hdf5_files = []
    for root, dirs, files in os.walk(dataset_dir):
        for filename in fnmatch.filter(files, "*.hdf5"):
            if "features" in filename:
                continue
            if skip_mirrored_data and "mirror" in filename:
                continue
            hdf5_files.append(os.path.join(root, filename))
    print(f"Found {len(hdf5_files)} hdf5 files")
    return hdf5_files


def get_norm_stats(dataset_dir, num_episodes, other_config={}):
    all_qpos_data = []
    all_action_data = []
    for episode_idx in num_episodes:
        dataset_path = os.path.join(dataset_dir, f"episode_{episode_idx}.hdf5")
        with h5py.File(dataset_path, "r") as root:
            qpos = root["/observations/qpos"][()]
            if "qvel" in other_config:
                qvel = root["/observations/qvel"][()]
            action = root["/action"][()]
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
    all_qpos_data = torch.stack(all_qpos_data)
    all_action_data = torch.stack(all_action_data)
    all_action_data = all_action_data

    # normalize action data
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf)  # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf)  # clipping

    stats = {
        "action_mean": action_mean.numpy().squeeze(),
        "action_std": action_std.numpy().squeeze(),
        "qpos_mean": qpos_mean.numpy().squeeze(),
        "qpos_std": qpos_std.numpy().squeeze(),
        "example_qpos": qpos,
    }

    return stats


def get_key_info(path):
    if ".pkl" not in path:
        path = os.path.join(path, f"key_info.pkl")
    with open(path, "rb") as f:
        key_info = pickle.load(f)
    return key_info


def get_init_states(dir, episode_idx=None):
    if isinstance(episode_idx, int):
        dataset_path = os.path.join(dir, f"episode_{episode_idx}.hdf5")
        with h5py.File(dataset_path, "r") as root:
            qpos = root["/observations/qpos"][0]
            action = root["/action"][0]
    else:
        # dir is info dir
        key_info_path = os.path.join(dir, f"key_info.pkl")
        with open(key_info_path, "rb") as f:
            key_info = pickle.load(f)
            qpos = key_info["init_info"]["init_joint"]
            action = key_info["init_info"]["init_action"]
    return qpos, action


# def check_init_states(dataset_dir, num_episodes, other_config=()):
#     """检测所有episode的初始状态是否一致"""
#     all_qpos_data = []
#     all_action_data = []
#     for episode_idx in range(num_episodes):
#         dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
#         with h5py.File(dataset_path, 'r') as root:
#             qpos = root['/observations/qpos'][0]
#             action = root['/action'][0]
#             if 'qvel' in other_config:
#                 qvel = root['/observations/qvel'][()]
#             all_qpos_data.append(torch.from_numpy(qpos))
#             all_action_data.append(torch.from_numpy(action))
#     all_qpos_data = torch.stack(all_qpos_data)
#     all_qpos_data = all_qpos_data
#     qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
#     qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
#     qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping

#     all_action_data = torch.stack(all_action_data)
#     all_action_data = all_action_data
#     action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
#     action_std = all_action_data.std(dim=[0, 1], keepdim=True)
#     action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

#     stats = {"qpos_mean": qpos_mean.numpy().squeeze(),
#              "qpos_std": qpos_std.numpy().squeeze(),
#              "qpos_min": all_qpos_data.min(dim=[0, 1], keepdim=True).values.numpy().squeeze(),
#              "qpos_max": all_qpos_data.max(dim=[0, 1], keepdim=True).values.numpy().squeeze(),
#              "action_mean": action_mean.numpy().squeeze(),
#              "action_std": action_std.numpy().squeeze(),
#              "action_min": all_action_data.min(dim=[0, 1], keepdim=True).values.numpy().squeeze(),
#              "action_max": all_action_data.max(dim=[0, 1], keepdim=True).values.numpy().squeeze(),}
#     return stats


def load_data(
    dataset_dir,
    num_episodes,
    camera_names,
    batch_size_train,
    batch_size_val,
    augmentors=None,
    other_config={},
):
    print(f"\nData from: {dataset_dir}\n")
    # obtain train test split
    train_ratio = other_config.get("train_ratio", 0.8)
    episodes_num = len(num_episodes)
    shuffled_indices = list(num_episodes)
    np.random.shuffle(shuffled_indices)
    train_indices = shuffled_indices[: int(train_ratio * episodes_num)]
    val_indices = shuffled_indices[int(train_ratio * episodes_num) :]

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir, num_episodes, other_config)

    # construct dataset
    train_dataset = EpisodicDataset(
        train_indices, dataset_dir, camera_names, norm_stats, augmentors
    )
    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats)
    # construct dataloader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        pin_memory=True,
        num_workers=1,
        prefetch_factor=1,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size_val,
        shuffle=True,
        pin_memory=True,
        num_workers=1,
        prefetch_factor=1,
    )

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


"""env utils"""


def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])


def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose


"""helper functions"""


def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result


def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def replace_timestamp(input_str, time_stamp):
    """Replace the time format in the input string with the given time_stamp."""
    # 检查time_stamp是否符合格式要求
    if time_stamp == "now":
        time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    elif len(time_stamp) != 15:
        raise ValueError("The length of time_stamp should be 15.")
    # 使用正则表达式匹配时间格式的部分，并替换为当前时间
    result_str = re.sub(r"\d{8}-\d{6}", time_stamp, input_str)
    return result_str


def pretty_print_dict(dictionary, indent=0):
    for key, value in dictionary.items():
        # 添加适当数量的缩进
        print(" " * indent, end="")
        # 如果值是字典，则递归调用函数
        if isinstance(value, dict):
            print(f"{key}:")
            pretty_print_dict(value, indent + 4)  # 增加缩进
        else:
            print(f"{key}: {value}")


class CAN_Tools(object):

    @staticmethod
    def check_can_status(interface):
        # 使用 ip link show 命令获取网络接口状态
        result = subprocess.run(
            ["ip", "link", "show", interface], capture_output=True, text=True
        )
        output = result.stdout.strip()

        # 检查输出中是否包含 'UP' 状态
        if "UP" in output and "LOWER_UP" in output:
            return True  # 已激活
        else:
            return False  # 未激活

    @staticmethod
    def activate_can_interface(interface, bitrate):
        # 构造要执行的命令
        command = f"sudo ip link set {interface} up type can bitrate {bitrate}"

        # 使用 Popen 执行命令并自动输入密码
        proc = subprocess.Popen(
            command,
            shell=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, stderr = proc.communicate()

        # 获取命令执行结果
        return_code = proc.returncode
        if return_code == 0:
            return True, stdout.decode()
        else:
            return False, stderr.decode()


class GPUer(object):

    def __init__(self, interval=1):
        self.interval = interval
        self.gpu_info = self.get_gpu_info()
        pretty_print_dict(self.gpu_info)
        gpu_num = len(self.gpu_info["name"])
        self.utilization = {
            "current": [None] * gpu_num,
            "max": [0] * gpu_num,
            "average": [None] * gpu_num,
        }
        self.update_event = Event()
        Thread(target=self.monitor, daemon=True).start()

    def update(self):
        self.update_event.set()

    def _update(self):
        self.utilization["current"] = self.get_gpu_utilization()
        for i, util in enumerate(self.utilization["current"]):
            self.utilization["max"][i] = max(self.utilization["max"][i], util)
            if self.utilization["average"][i] is None:
                self.utilization["average"][i] = util
            else:
                self.utilization["average"][i] = (
                    self.utilization["average"][i] + util
                ) / 2

    def monitor(self):
        """Monitor the GPU utilization."""
        while True:
            self.update_event.wait()
            start_time = time.time()
            self._update()
            # print(f"GPU utilization: {self.utilization}")
            sleep_time = self.interval - (time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
            self.update_event.clear()

    def save_info(self, path):
        """Save the GPU utilization information to a file.
        Parameters
        ----------
        path: str
            The path to save the information.
        """
        with open(path, "w") as f:
            f.write(f"Max GPU utilization(%): {self.utilization['max']}\n")
            f.write(f"Average GPU utilization(%): {self.utilization['average']}\n")

    @staticmethod
    def get_gpu_info() -> dict:
        """Get the information of all GPUs.
        Returns
        -------
        info: list of str
            The information of all GPUs, each element is a string containing the index, name, total memory, free memory, and used memory of a GPU.
        """
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        gpus_info = result.stdout.strip().split("\n")
        gpus_info_dict = {
            "name": [],
            "memory.total": [],
            "memory.free": [],
            "memory.used": [],
            "memory.utilization": [],
        }
        for gpu_info in gpus_info:
            infos = gpu_info.split(",")
            gpus_info_dict["name"].append(infos[1][1:])
            gpus_info_dict["memory.total"].append(int(infos[2][1:]))
            gpus_info_dict["memory.free"].append(int(infos[3][1:]))
            gpus_info_dict["memory.used"].append(int(infos[4][1:]))
            gpus_info_dict["memory.utilization"].append(
                gpus_info_dict["memory.used"][-1]
                / gpus_info_dict["memory.total"][-1]
                * 100
            )
        return gpus_info_dict

    @staticmethod
    def get_gpu_memory_map() -> List[int]:
        """Get the current GPU memory usage.
        Returns
        -------
        usage: list of int containing the memory usage of each GPU.
        """
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        gpu_memory = [int(x) for x in result.stdout.strip().split("\n")]
        return gpu_memory

    @staticmethod
    def get_gpu_utilization() -> List[int]:
        """Get the current GPU utilization.
        Returns
        -------
        utilization: list of int containing the utilization of each GPU in percentage.
        """
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        gpu_utilization = [int(x) for x in result.stdout.strip().split("\n")]
        return gpu_utilization

    @staticmethod
    def check_all_gpus_idle(utilization_threshold=10) -> Tuple[List[int], int]:
        """Check the utilization of all GPUs and return the indices of idle GPUs.
        Parameters
        ----------
        utilization_threshold: int, optional
            The threshold of GPU utilization to determine whether a GPU is idle.
        Returns
        -------
        idle_gpus: list of int
            The indices of idle GPUs.
        total_gpus: int
            The total number of GPUs.
        """
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            gpu_usage = result.stdout.splitlines()

            idle_gpus = []
            for gpu_id, usage in enumerate(gpu_usage):
                gpu_utilization = int(usage.strip())
                if gpu_utilization < utilization_threshold:
                    idle_gpus.append(gpu_id)

            return idle_gpus, len(gpu_usage)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"Error occurred: {e}")
            return []


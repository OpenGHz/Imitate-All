import os
import subprocess
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Any, Dict, Optional, List, Callable, Union
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


def is_nested(data):
    if not isinstance(data[0], (int, float, str)):
        return True
    else:
        return False


def get_files_name_by_suffix(directory, suffix):
    all: List[str] = os.listdir(directory)
    avi_files = [f for f in all if f.endswith(suffix)]
    return avi_files


def get_folders_name(directory: str):
    all: List[str] = os.listdir(directory)
    folders = [f for f in all if os.path.isdir(f"{directory}/{f}")]
    return folders


def flatten_dict(d: dict, parent_key="", sep="/", prefix=""):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else f"{prefix}{k}"
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def flatten_list(l):
    flattened = []
    for item in l:
        if isinstance(item, list):
            flattened.extend(flatten_list(item))
        else:
            flattened.append(item)
    return flattened


def flatten_sublist_in_flat_dict(d: dict) -> dict:
    """Flatten the sublist in the flat dict.
    This will change the original dict.
    Returns the lenth of each value list.
    """
    trajs_lenth = {}
    for key, traj in d.items():
        if not isinstance(traj, list):
            trajs_lenth[key] = None
            continue
        traj_lenth = len(traj)
        if traj_lenth > 0:
            point = traj[0]
            if isinstance(point, list):
                if isinstance(point[0], list):
                    for i, p in enumerate(traj):
                        traj[i] = sum(p, [])
        trajs_lenth[key] = traj_lenth
    return trajs_lenth


class Compresser(object):

    def __init__(self, compress_method: str, compress_param: Any, padding=False):
        self.compress_param = compress_param
        self.compress_method = compress_method
        self.padding = padding
        self.pad_size = None
        assert self.compress_method in ["resize", "jpg"], "Invalid compress method."

    def compress(self, frame):
        if self.compress_method == "resize":
            return cv2.resize(
                frame,
                (
                    int(frame.shape[1] * self.compress_param),
                    int(frame.shape[0] * self.compress_param),
                ),
            )
        elif self.compress_method == "jpg":
            return cv2.imencode(".jpg", frame, self.compress_param)[1]
        else:
            raise NotImplementedError(
                f"Compress method {self.compress_method} is not implemented."
            )

    @staticmethod
    def decompress(frame, method):
        if method in ["jpg", "jpeg"]:
            return cv2.imdecode(frame, cv2.IMREAD_COLOR)
        else:
            raise NotImplementedError(f"Decompress method {method} is not implemented.")

    @staticmethod
    def pad_image_data(data: Union[list, dict], max_len: int) -> np.ndarray:
        """Pad image data so that all compressed images have the same length as the max"""
        # t0 = time.time()
        all_padded_data = {}
        dict_data = isinstance(data, dict)
        if not dict_data:
            data = {"images": data}
        for key, value in data.items():
            padded_compressed_image_list = []
            for compressed_image in value:
                padded_compressed_image = np.zeros(max_len, dtype="uint8")
                image_len = len(compressed_image)
                padded_compressed_image[:image_len] = compressed_image
                padded_compressed_image_list.append(padded_compressed_image)
            all_padded_data[key] = padded_compressed_image_list
        # logging.debug(f"padding: {time.time() - t0:.2f}s")
        if not dict_data:
            all_padded_data = all_padded_data["images"]
        return all_padded_data


def remove_after_last_dot(s):
    last_dot_index = s.rfind(".")
    if last_dot_index != -1:
        return s[:last_dot_index]
    return s


def video_to_dict(
    video_dir: str,
    video_names: Optional[List[str]] = None,
    video_type: Optional[str] = None,
    name_converter: Optional[Dict[str, str]] = None,
    compresser: Optional[Compresser] = None,
    pre_process: Optional[Callable] = None,
    max_threads: int = -1,
) -> Dict[str, list]:
    """Load the video data to a dictionary.
    Returns a dictionary with video data.

    Parameters:
        video_dir(str)          -- the directory of the video data
        video_names(list)       -- the name of the video files
    Returns:
        video_dict(dict)        -- the video data dictionary, with keys as the video names
    """
    video_dict = {}

    # Ensure video_dir is a Path object
    video_dir = Path(video_dir)

    # If video_names is not provided, get all video files in the directory
    if video_type is None:
        assert (
            video_names is not None
        ), "Please provide the video type or the video_names with type."
        assert (
            "." in video_names[0]
        ), "Please provide the video type or the video_names with type."
        typer = lambda i: video_names[i].split(".")[-1]
    else:
        typer = lambda i: video_type

    if video_names is None:
        # TODO: check why set 0 always
        video_names = get_files_name_by_suffix(video_dir, f".{typer(0)}")

    name_converter = {} if name_converter is None else name_converter
    no_suffix_names = [remove_after_last_dot(name) for name in video_names]
    for key in name_converter.keys():
        if key not in no_suffix_names:
            # TODO: fix name with suffix error
            logger.warning(f"{key} is not found in the video names.")
            name_converter.pop(key)

    compressed_len: Dict[str, list] = {}
    if pre_process is None:
        pre_process = lambda x: x

    def process_one_video(video_path, video_name):
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        compressed_len[video_name] = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if compresser is not None:
                frame = compresser.compress(frame)
                if compresser.compress_method == "jpg":
                    compressed_len[video_name].append(len(frame))
            frames.append(pre_process(frame))

        cap.release()
        name = name_converter.get(video_name, video_name)
        video_dict[f"{name}"] = frames

    futures = []
    max_threads = len(video_names) if max_threads == -1 else max_threads
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        for video_name in video_names:
            video_path = video_dir / Path(video_name)
            video_name = remove_after_last_dot(video_name)
            if not video_path.exists():
                print(f"Warning: {video_path} does not exist.")
                continue
            else:
                futures.append(
                    executor.submit(process_one_video, video_path, video_name)
                )
        as_completed(futures)

    compressed_len_list = list(compressed_len.values())
    if len(compressed_len_list[0]) > 0:
        if compresser.padding:
            compresser.pad_size = np.array(compressed_len_list).max()
            video_dict = Compresser.pad_image_data(video_dict, compresser.pad_size)
        logging.debug(f"compressed_len_list: {compressed_len_list}")
        logging.debug("compressed_len", compressed_len)
        video_dict["compressed_len"] = compressed_len

    return video_dict


def convert_avi_to_mp4(
    avi_dir: str,
    mp4_dir: Optional[str] = None,
    avi_names: Optional[List[str]] = None,
    mp4_names: Optional[List[str]] = None,
    fps: int = None,
) -> str:
    """convert `.avi` files to `.mp4` files.
    Please install ffmpeg first: `sudo apt install ffmpeg`

    Parameters:
        avi_dir(Path)       -- directory which contains data of one episode
        mp4_dir(Path)       -- the output directory for `.mp4` files
        episode(int)        -- the episode index of the current episode
        avi_names(str)      -- the camera captured this `.avi` file, default `None` for all videos.
        mp4_names(str)      -- the name of the output `.mp4` files, default `None` for all videos.
        fps(int)            -- frame rate used to collect videos.
    Returns:
        out_names(list)     -- the name of the output `.mp4` files
    """
    avi_dir: Path = Path(avi_dir)
    if avi_names is None:
        avi_names = get_files_name_by_suffix(avi_dir, ".avi")
    if mp4_dir is None:
        mp4_dir = avi_dir
    else:
        mp4_dir: Path = Path(mp4_dir)
    if mp4_names is not None:
        assert len(mp4_names) == len(
            avi_names
        ), "The length of mp4 names should be the same as avi names."

    mp4_dir.parent.mkdir(parents=True, exist_ok=True)
    fps = fps if fps is not None else 30
    out_names = []

    for index, cam_name in enumerate(avi_names):
        if ".avi" not in cam_name:
            cam_name += ".avi"
        if mp4_names is None:
            mp4_name = cam_name.replace(".avi", ".mp4")
            out_names.append(mp4_name)
        else:
            mp4_name = mp4_names[index]
        ffmpeg_cmd = (
            f"ffmpeg "
            f"-i {str(avi_dir / cam_name)} "
            f"-y {str(mp4_dir / mp4_name)} "
            f"-r {fps}"
        )
        subprocess.run(ffmpeg_cmd.split(" "), check=True)
    if mp4_names is not None:
        out_names = mp4_names
    return out_names


def raw_to_dict(
    raw_dir: str,
    state_file_names: List[str],
    video_file_names: Optional[List[str]] = None,
    flatten_mode: str = "hdf5",  # "hdf5", "hf"
    name_converter: Optional[Dict[str, str]] = None,
    pre_process: Optional[Callable] = None,
    concatenater: Optional[Dict[str, str]] = None,
    key_filter: Optional[List] = None,
) -> Dict[str, dict]:
    """Load the raw data to a dictionary.
    Parameters:
        raw_dir(Path)           -- the directory of the raw data
        state_file_names(list)  -- the name of the state files
        video_file_names(list)  -- the name of the video files
        flatten_mode(str)       -- the flatten mode for the dict keys
        name_converter(dict)    -- the name converter for the flattened dict keys
        concatenater(dict)      -- the concatenate dict to bind several flattened keys together
    Returns:
        ep_dicts(dict)          -- the raw data dictionary, with keys as the episode names
    Note:
        1. Each episode has a dict with keys as the data names and values as the array-like data.
        2. Empty values will be ignored with its' key.
    """
    episode_names = get_folders_name(raw_dir)
    raw_dir: Path = Path(raw_dir)
    ep_dicts = {}
    name_converter = {} if name_converter is None else name_converter
    if pre_process is None:
        pre_process = lambda x: x
    for ep_name in tqdm(episode_names, desc="Data Converting"):
        ep_dir = raw_dir / str(ep_name)

        # dict for each episode
        ep_dict = {}
        for state_file in state_file_names:
            with open(ep_dir / state_file, "r") as f:
                # read the raw data
                if ".json" in state_file:
                    raw_data: dict = json.load(f)
                else:
                    raise NotImplementedError(
                        f"File type {state_file} is not supported."
                    )
                # flatten the dict
                mode_to_sep_prefix = {
                    "hdf5": ("/", "/"),
                    "hf": (".", ""),
                }
                if flatten_mode is not None:
                    sep_prefix = mode_to_sep_prefix.get(flatten_mode, None)
                    assert sep_prefix is not None, f"Invalid flatten mode {flatten_mode}."
                    data_flat: Dict[str, list] = flatten_dict(raw_data, "", *sep_prefix)
                else:
                    data_flat = raw_data
                # filter the keys
                if key_filter is not None:
                    for key in key_filter:
                        data_flat.pop(key, None)
                        # if raw_data.pop(key, None) is None:
                        #     print(f"Key {key} is not found in {state_file}.")
                # flatten the sub list
                lenths = flatten_sublist_in_flat_dict(data_flat)
                for key, value in lenths.copy().items():
                    if value == 0:
                        # remove not used keys
                        # print(f"Remove not used key: {key}")
                        data_flat.pop(key)
                        lenths.pop(key)
                lenths = tuple(lenths.values())
                assert np.all(
                    np.array(lenths) == lenths[0]
                ), "The length of each value list should be the same."
                for key, value in data_flat.items():
                    name = name_converter.get(key, key)
                    ep_dict[name] = pre_process(value)

        if video_file_names is not None:
            video_type = video_file_names[0].split(".")[-1]
            video_dict = video_to_dict(
                ep_dir, video_file_names, video_type, name_converter
            )
            for key, value in video_dict.items():
                ep_dict[key] = value

        if concatenater is not None:
            for key, value in concatenater.items():
                one_dim_reshape = lambda x: x.reshape(-1, 1) if len(x.shape) == 1 else x
                ep_dict[key] = np.concatenate(
                    [one_dim_reshape(np.array(ep_dict[v])) for v in value], axis=1
                ).tolist()
                # remove the original keys if v != key
                for v in value:
                    if v != key:
                        if ep_dict.pop(v, None) is None:
                            print(f"Key {v} is not found in the episode {ep_name}.")
        ep_dicts[ep_name] = ep_dict

    return ep_dicts


import h5py


def save_dict_to_hdf5(data: dict, target_path: str, pad_max_len: Optional[int] = None):
    """Save the data dict to the target path in hdf5 format.
    Parameters:
        data(dict)          -- the data dict to be saved, with keys as the data names and values as the array-like data
        target_path(str)    -- the target path to save the data
        pad_max_len(int)    -- the max length to pad all the data to the same episode length
    """
    with h5py.File(target_path, "w", rdcc_nbytes=1024**2 * 2) as root:
        for key, value in data.items():
            is_dict_value = isinstance(value, dict)
            if is_dict_value:  # e.g. compress_len
                value = list(value.values())
                # print(f"key:{key} ||| value:{value}")
            if "images" in key:
                dtype = "uint8"
                chunks = (1, *value[0].shape)
                array_type = np.uint8
            else:
                dtype = "float32"
                chunks = None
                array_type = np.float32
            # padding the data with the last N values
            if pad_max_len is not None:

                def pad(value):
                    size_to_pad = pad_max_len - len(value)
                    if size_to_pad > 0:
                        value = value + value[-size_to_pad:]
                    return value

                if is_dict_value:
                    for i, v in enumerate(value):
                        value[i] = pad(v)
                else:
                    value = pad(value)
            root.create_dataset(
                key, data=np.array(value, dtype=array_type), dtype=dtype, chunks=chunks
            )
        root.attrs["sim"] = False
        root.attrs["compress"] = True


def save_dict_to_json_and_mp4(data: dict, target_path: str, pad_max_len: Optional[int] = None, fps: int = 30):
    """Save the data dict to the target path in hdf5 format.
    Parameters:
        data(dict)          -- the data dict to be saved, with keys as the data names and values as the array-like data
        target_path(str)    -- the target path to save the data
        pad_max_len(int)    -- the max length to pad all the data to the same episode length
    """
    # print("Call: save_dict_to_json_and_mp4.")
    target_path = Path(target_path)
    # print("target_path=", target_path)
    target_path.mkdir(parents=True, exist_ok=True)
    images_dict = {}
    # padding the data with the last N values
    keys = list(data.keys())
    for key in keys:
        if isinstance(data[key], dict):
            continue
        if pad_max_len is not None:
            def pad(value):
                size_to_pad = pad_max_len - len(value)
                if size_to_pad > 0:
                    value = value + value[-size_to_pad:]
                return value
            data[key] = pad(data[key])
        if "images" in key:
            images_dict[key] = data.pop(key)

    with open(target_path / "records.json", "w") as f:
        json.dump(data, f)
    for key, value in images_dict.items():
        # create video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        w_h = (value[0].shape[1], value[0].shape[0])
        image_path = str(target_path / f"{key.split('/')[-1]}.avi")
        # print(f"Save images to {image_path} with w_h={w_h}, fps={fps}")
        video_writer = cv2.VideoWriter(
            image_path, fourcc, fps, w_h
        )
        # save images
        for img in value:
            video_writer.write(img)

        video_writer.release()
        # print(f"Save the video data to image_path.")

def hdf5_to_dict(hdf5_path):
    def recursively_load_datasets(hdf5_group):
        data_dict = {}
        for key in hdf5_group.keys():
            item = hdf5_group[key]
            if isinstance(item, h5py.Dataset):
                data_dict[key] = item[()]
            elif isinstance(item, h5py.Group):
                data_dict[key] = recursively_load_datasets(item)
        return data_dict

    with h5py.File(hdf5_path, "r") as root:
        data = recursively_load_datasets(root)
    return data


def compress_images(images: list, compresser: Compresser) -> list:
    """Compress the images in the list (will modify the input list)."""
    compressed_len = []
    for i, img in enumerate(images):
        compressed_img = compresser.compress(img)
        compressed_len.append(len(compressed_img))
        images[i] = compressed_img
    if compresser.padding:
        compresser.pad_size = max(compressed_len)
        images = Compresser.pad_image_data(images, compresser.pad_size)
    return images


def merge_video_and_save(
    raw_data: dict,
    video_dir: str,
    video_names: list,
    saver: Callable,
    name_converter: Optional[Dict[str, str]] = None,
    compresser: Optional[Compresser] = None,
    target_path: Optional[str] = None,
    pad_max_len: Optional[int] = None,
    *args,
    **kwargs,
):
    # read images from video files
    images = video_to_dict(video_dir, video_names, None, name_converter, compresser)
    # merge the images into the raw data dict
    raw_data.update(images)
    # save the raw data dict to the target directory
    saver(raw_data, target_path, pad_max_len, *args, **kwargs)

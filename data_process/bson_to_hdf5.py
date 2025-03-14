import convert_all as crd
import os
import cv2
import argparse
from concurrent.futures import ThreadPoolExecutor
import sys


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

parser = argparse.ArgumentParser()
parser.add_argument("-tn", "--task_name", type=str)
parser.add_argument("-cn", "--camera_names", type=str, nargs="+")
parser.add_argument("-ds", "--downsampling", type=int, default=0)
parser.add_argument("-md", "--mode", type=str, default="play")
parser.add_argument("-dir", "--raw_dir", type=str, default="data/raw")
# parser.add_argument("-pad", "--padding", action="store_true")
args = parser.parse_args()

task_name = args.task_name
camera_names = args.camera_names
downsampling = args.downsampling
mode = args.mode
raw_dir = args.raw_dir
# padding = args.padding

task_dir = os.path.abspath(f"{raw_dir}/{task_name}")
assert os.path.exists(task_dir), f"task_dir {task_dir} not exists"

name_converter = {
    f"/images/{raw_name}": f"/observations/images/{i}"
    for i, raw_name in enumerate(camera_names)
}

if mode == "play":
    obs_keys_low_dim = (
        "/observation1/arm/joint_position",
        "/observation1/eef/joint_position",
    )
    act_keys = ("/action1/arm/joint_position", "/action1/eef/joint_position")
elif mode == "mmk2":
    obs_keys_low_dim = (
        "/observation/left_arm/joint_state",
        "/observation/left_arm_eef/joint_state",
        "/observation/right_arm/joint_state",
        "/observation/right_arm_eef/joint_state",
        "/observation/head/joint_state",
        "/observation/spine/joint_state",
    )
    act_keys = (
        "/action/left_arm/joint_state",
        "/action/left_arm_eef/joint_state",
        "/action/right_arm/joint_state",
        "/action/right_arm_eef/joint_state",
        "/action/head/joint_state",
        "/action/spine/joint_state",
    )
    name_converter = {
        f"/images/{raw_name}": f"/observations/images/{raw_name}"
        for raw_name in camera_names
    }
else:
    raise ValueError(f"mode {mode} not supported")
image_keys = [f"/images/{name}" for name in args.camera_names]

print(f"name_converter: {name_converter}")

pre_process = {
    key: crd.Compresser("jpg", [int(cv2.IMWRITE_JPEG_QUALITY), 50], True).compress
    for key in image_keys
}

pre_process.update(
    {
        key: lambda data: data["pos"]
        for key in [
            *obs_keys_low_dim,
            *act_keys,
        ]
    }
)

if mode == "mmk2":
    key_filter =  [
            "/observation/left_arm/pose",
            "/observation/right_arm/pose",
            # "action/eef/pose",
            # "/time",
    ]
elif mode == "play":
    pre_process.update(
        {"/action1/eef/joint_position": lambda data: data["t"]}
    )
    pre_process.update(
        {"/observation1/eef/joint_position": lambda data: data["t"]}
    )
    key_filter = [
        "/action1/eef/pose",
        "/observation1/eef/pose",
    ]
else:
    raise ValueError(f"mode {mode} not supported")

concatenater = {
    "/observations/qpos": obs_keys_low_dim,
    "/action": act_keys,
}

padding = {key: 0 for key in image_keys}

# crd.downsample(data, downsampling)

# merge high_dim data and save
target_dir = f"data/hdf5/{task_name}/"
target_namer = lambda i: f"episode_{i}.hdf5"


# create target dir
os.makedirs(target_dir, exist_ok=True)

print(f"Try to find all episode files in {task_dir}...")
if mode == "mmk2":
    episode_names = crd.get_files_name_by_suffix(task_dir, ".bson")
elif mode == "play":
    episode_names = [f"{fd}/data.bson" for fd in os.listdir(task_dir)]
else:
    raise ValueError(f"mode {mode} not supported")
print(f"episode_names: {episode_names}")


def save_one(index, ep_name):
    bson_dict, stamps = crd.raw_bson_to_dict(
        f"{task_dir}/{ep_name}",
        None,
        name_converter,
        pre_process,
        concatenater,
        key_filter,
        padding,
    )
    print(bson_dict.keys())
    return crd.save_dict_to_hdf5(bson_dict, target_dir + target_namer(index), None)


# save one first to print logs
print("Try saving one data to check if everything is ok...")
index = 0
ep_name = episode_names[index]
save_one(index, ep_name)

# save all data
print(f"Start saving all data to {target_dir}...")
futures = []
with ThreadPoolExecutor(max_workers=25) as executor:
    for index, ep_name in enumerate(episode_names):
        futures.append(executor.submit(save_one, index, ep_name))
print(f"All data saved to {target_dir}")

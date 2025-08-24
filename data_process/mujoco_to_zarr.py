import argparse
import os
import sys
import convert_all as crd
import zarr
import tqdm
import numpy as np
import torch
import pytorch3d.ops as torch3d_ops
import torchvision
from termcolor import cprint
from typing import Dict


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

parser = argparse.ArgumentParser()
parser.add_argument("-tn", "--task_name", type=str)
parser.add_argument("-ds", "--downsampling", type=int, default=0)
parser.add_argument("-md", "--mode", type=str, default="real3")
parser.add_argument("-pad", "--padding", action="store_true")
parser.add_argument("-dir", "--raw_dir", type=str, default="data/raw")
parser.add_argument("-vn", "--video_names", type=str, nargs="+")
parser.add_argument(
    "--num_points", type=int, default=1024, help="Number of points after sampling"
)
parser.add_argument("--no_crop", action="store_true", help="Disable workspace cropping")
parser.add_argument(
    "--no_transform", action="store_true", help="Disable extrinsic transform + scaling"
)
parser.add_argument(
    "--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device for FPS"
)
args = parser.parse_args()

task_name = args.task_name
downsampling = args.downsampling
mode = args.mode
padding = args.padding
raw_dir = args.raw_dir
video_names = args.video_names
num_points_cfg = args.num_points
disable_crop = args.no_crop
disable_transform = args.no_transform
device = args.device
use_cuda_global = device == "cuda" and torch.cuda.is_available()
assert len(video_names) == 1, "Please provide exactly one video name."

task_dir = os.path.abspath(f"{raw_dir}/{task_name}")
assert os.path.exists(task_dir), f"task_dir {task_dir} not exists"


WORK_SPACE = [
    [0.65, 1.1],  # x
    [0.45, 0.66],  # y
    [-0.7, 0.0],  # z
]

extrinsics_matrix = np.array(
    [
        [0.5213259, -0.84716441, 0.10262438, 0.04268034],
        [0.25161211, 0.26751035, 0.93012341, 0.15598059],
        [-0.81542053, -0.45907589, 0.3526169, 0.47807532],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)

def load_raw_mujoco_data(raw_dir, downsampling=0):
    data = crd.raw_to_dict(
        raw_dir,
        ["obs_action.json"],
        video_file_names=None,
        flatten_mode="hdf5",
        name_converter={
            "/obs/jq": "state",
            "/act": "action",
        },
        pre_process=None,
        concatenater=None,
        key_filter=["/time"],
    )
    return crd.downsample(data, downsampling)


def farthest_point_sampling(points, num_points=1024, use_cuda=True):
    K = [num_points]
    if use_cuda:
        points = torch.from_numpy(points).float().cuda()
        sampled_points, indices = torch3d_ops.sample_farthest_points(
            points=points.unsqueeze(0), K=K
        )
        sampled_points = sampled_points.squeeze(0).cpu().numpy()
    else:
        points = torch.from_numpy(points).float()
        sampled_points, indices = torch3d_ops.sample_farthest_points(
            points=points.unsqueeze(0), K=K
        )
        sampled_points = sampled_points.squeeze(0).numpy()

    return sampled_points, indices


def preprocess_point_cloud(points, use_cuda=True):
    """Preprocess a single frame point cloud.

    Steps:
      1. (Optional) scale + extrinsic transform (kept as original logic)
      2. Crop to workspace
      3. Farthest Point Sampling to num_points (with fallbacks if not enough)
      4. Return concatenated XYZ + RGB (RGB zeros if not provided)

    Robustness additions:
      - Handle empty input or post-crop empty
      - If remaining points < num_points, sample with replacement (no FPS) to avoid crash
      - Ensure float32 dtype
    """

    num_points = num_points_cfg

    if points is None or len(points) == 0:
        # Return zeros placeholder
        return np.zeros((num_points, 6), dtype=np.float32)

    points = points.astype(np.float32, copy=False)

    # Ensure at least XYZ exist
    assert points.shape[1] >= 3, (
        f"Point cloud must have at least 3 dims, got shape {points.shape}"
    )

    # Copy so in-place modifications don't affect upstream caches
    points = points.copy()

    # 1) scale + transform
    if not disable_transform:
        try:
            point_xyz = points[..., :3] * 0.0002500000118743628
            point_homogeneous = np.hstack(
                (point_xyz, np.ones((point_xyz.shape[0], 1), dtype=np.float32))
            )
            point_homogeneous = point_homogeneous @ extrinsics_matrix
            point_xyz = point_homogeneous[..., :-1]
            points[..., :3] = point_xyz
        except Exception as e:
            print(
                f"[preprocess_point_cloud] Transform failed: {e}. Proceeding without transform."
            )

    # 2) crop
    if not disable_crop:
        before_crop = points.shape[0]
        mask = (
            (points[:, 0] > WORK_SPACE[0][0])
            & (points[:, 0] < WORK_SPACE[0][1])
            & (points[:, 1] > WORK_SPACE[1][0])
            & (points[:, 1] < WORK_SPACE[1][1])
            & (points[:, 2] > WORK_SPACE[2][0])
            & (points[:, 2] < WORK_SPACE[2][1])
        )
        cropped = points[mask]
        if cropped.shape[0] == 0:
            # Fallback: if crop removed everything, skip cropping (likely calibration mismatch for mujoco)
            print(
                f"[preprocess_point_cloud] Warning: crop removed all {before_crop} points. Using uncropped points."
            )
            cropped = points
        points = cropped

    # 3) sampling
    xyz = points[:, :3]
    has_color = points.shape[1] >= 6  # assume layout xyzrgb(...)
    colors = None
    if has_color:
        colors = points[:, 3:6]
    else:
        # create dummy zeros
        colors = np.zeros((points.shape[0], 3), dtype=np.float32)

    if points.shape[0] >= num_points and points.shape[0] > 0:
        try:
            sampled_xyz, sample_indices = farthest_point_sampling(
                xyz, num_points=num_points, use_cuda=use_cuda
            )
            # sample_indices shape: (1, K)
            if isinstance(sample_indices, torch.Tensor):
                sample_indices = sample_indices.squeeze(0).cpu().numpy()
            else:
                sample_indices = np.array(sample_indices).squeeze(0)
            sampled_colors = colors[sample_indices]
        except Exception as e:
            print(
                f"[preprocess_point_cloud] FPS failed ({e}); falling back to random sampling."
            )
            choice = np.random.choice(
                points.shape[0], size=num_points, replace=points.shape[0] < num_points
            )
            sampled_xyz = xyz[choice]
            sampled_colors = colors[choice]
    else:
        # Not enough points for FPS; sample with replacement
        if points.shape[0] == 0:
            sampled_xyz = np.zeros((num_points, 3), dtype=np.float32)
            sampled_colors = np.zeros((num_points, 3), dtype=np.float32)
        else:
            choice = np.random.choice(points.shape[0], size=num_points, replace=True)
            sampled_xyz = xyz[choice]
            sampled_colors = colors[choice]

    processed = np.concatenate([sampled_xyz, sampled_colors], axis=1).astype(np.float32)
    return processed


def preproces_image(image):
    img_size = 84

    image = image.astype(np.float32)
    image = torch.from_numpy(image).cuda()
    image = image.permute(2, 0, 1)  # HxWx4 -> 4xHxW
    image = torchvision.transforms.functional.resize(image, (img_size, img_size))
    image = image.permute(1, 2, 0)  # 4xHxW -> HxWx4
    image = image.cpu().numpy()
    return image


low_dim_data: Dict[str, Dict] = load_raw_mujoco_data(task_dir, downsampling)
save_data_path = f"data/zarr/{task_name}.zarr"

# get max episode length
episode_lens = {}
for ep, low_d in low_dim_data.items():
    episode_lens[ep] = len(list(low_d.values())[0])
print(
    f"min episode length: {min(episode_lens.values())}, max episode length: {max(episode_lens.values())}"
)
max_pad_length = max(episode_lens.values()) if padding else None

print(f"Episode flatten keys: {list(low_dim_data.values())[0].keys()}")
episode_names = list(low_dim_data.keys())
print(f"Episode number: {len(episode_names)}")
print(f"Max episode length: {max_pad_length}")
print(f"All episodes: {episode_names}")


# storage
total_count = 0
img_arrays = []
point_cloud_arrays = []
depth_arrays = []
state_arrays = []
action_arrays = []
episode_ends_arrays = []


if os.path.exists(save_data_path):
    cprint("Data already exists at {}".format(save_data_path), "red")
    cprint("If you want to overwrite, delete the existing directory first.", "red")
    cprint("Do you want to overwrite? (y/n)", "red")
    user_input = "y"
    if user_input == "y":
        cprint("Overwriting {}".format(save_data_path), "red")
        os.system("rm -rf {}".format(save_data_path))
    else:
        cprint("Exiting", "red")
        exit()
os.makedirs(save_data_path, exist_ok=True)


for ep, demo in low_dim_data.items():
    print(f"Processing episode {ep}")
    pc_dir = os.path.join(task_dir, f"{ep}/point_cloud/{video_names[0]}")
    demo["point_cloud"] = []
    demo_length = episode_lens[ep]
    for i in range(demo_length):
        demo["point_cloud"].append(np.load(os.path.join(pc_dir, f"{i}.npy")))
    pc_arr = np.array(demo["point_cloud"])
    print(pc_arr.shape)
    demo["point_cloud"] = pc_arr

    # dict_keys(['point_cloud', 'state', 'action'])
    for step_idx in tqdm.tqdm(range(demo_length)):
        total_count += 1
        obs_pointcloud = demo["point_cloud"][step_idx]
        robot_state = demo["state"][step_idx]
        action = demo["action"][step_idx]

        obs_pointcloud = preprocess_point_cloud(
            obs_pointcloud, use_cuda=use_cuda_global
        )
        action_arrays.append(action)
        point_cloud_arrays.append(obs_pointcloud)
        state_arrays.append(robot_state)

    # mark end index for this episode
    episode_ends_arrays.append(total_count)


# create zarr file
zarr_root = zarr.group(save_data_path)
zarr_data = zarr_root.create_group("data")
zarr_meta = zarr_root.create_group("meta")

# img_arrays = np.stack(img_arrays, axis=0)
# if img_arrays.shape[1] == 3: # make channel last
#     img_arrays = np.transpose(img_arrays, (0,2,3,1))
point_cloud_arrays = np.stack(point_cloud_arrays, axis=0)
# depth_arrays = np.stack(depth_arrays, axis=0)
action_arrays = np.stack(action_arrays, axis=0)
state_arrays = np.stack(state_arrays, axis=0)
episode_ends_arrays = np.array(episode_ends_arrays)

compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
# img_chunk_size = (100, img_arrays.shape[1], img_arrays.shape[2], img_arrays.shape[3])
point_cloud_chunk_size = (100, point_cloud_arrays.shape[1], point_cloud_arrays.shape[2])
# depth_chunk_size = (100, depth_arrays.shape[1], depth_arrays.shape[2])
if len(action_arrays.shape) == 2:
    action_chunk_size = (100, action_arrays.shape[1])
elif len(action_arrays.shape) == 3:
    action_chunk_size = (100, action_arrays.shape[1], action_arrays.shape[2])
else:
    raise NotImplementedError
# zarr_data.create_dataset('img', data=img_arrays, chunks=img_chunk_size, dtype='uint8', overwrite=True, compressor=compressor)
zarr_data.create_dataset(
    "point_cloud",
    data=point_cloud_arrays,
    chunks=point_cloud_chunk_size,
    dtype="float64",
    overwrite=True,
    compressor=compressor,
)
# zarr_data.create_dataset('depth', data=depth_arrays, chunks=depth_chunk_size, dtype='float64', overwrite=True, compressor=compressor)
zarr_data.create_dataset(
    "action",
    data=action_arrays,
    chunks=action_chunk_size,
    dtype="float32",
    overwrite=True,
    compressor=compressor,
)
zarr_data.create_dataset(
    "state",
    data=state_arrays,
    chunks=(100, state_arrays.shape[1]),
    dtype="float32",
    overwrite=True,
    compressor=compressor,
)
zarr_meta.create_dataset(
    "episode_ends",
    data=episode_ends_arrays,
    chunks=(100,),
    dtype="int64",
    overwrite=True,
    compressor=compressor,
)

# print shape
# cprint(f'img shape: {img_arrays.shape}, range: [{np.min(img_arrays)}, {np.max(img_arrays)}]', 'green')
cprint(
    f"point_cloud shape: {point_cloud_arrays.shape}, range: [{np.min(point_cloud_arrays)}, {np.max(point_cloud_arrays)}]",
    "green",
)
# cprint(f'depth shape: {depth_arrays.shape}, range: [{np.min(depth_arrays)}, {np.max(depth_arrays)}]', 'green')
cprint(
    f"action shape: {action_arrays.shape}, range: [{np.min(action_arrays)}, {np.max(action_arrays)}]",
    "green",
)
cprint(
    f"state shape: {state_arrays.shape}, range: [{np.min(state_arrays)}, {np.max(state_arrays)}]",
    "green",
)
cprint(
    f"episode_ends shape: {episode_ends_arrays.shape}, range: [{np.min(episode_ends_arrays)}, {np.max(episode_ends_arrays)}]",
    "green",
)
cprint(f"total_count: {total_count}", "green")
cprint(f"Saved zarr file to {save_data_path}", "green")

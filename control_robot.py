"""
Utilities to control a robot.

Useful to record a dataset, replay a recorded episode, run the policy on your robot
and record an evaluation dataset, and to recalibrate your robot if needed.

Examples of usage:

- Recalibrate your robot:
```bash
python lerobot/scripts/control_robot.py calibrate
```

- Unlimited teleoperation at highest frequency (~200 Hz is expected), to exit with CTRL+C:
```bash
python lerobot/scripts/control_robot.py teleoperate

# Remove the cameras from the robot definition. They are not used in 'teleoperate' anyway.
python lerobot/scripts/control_robot.py teleoperate --robot-overrides '~cameras'
```

- Unlimited teleoperation at a limited frequency of 30 Hz, to simulate data recording frequency:
```bash
python lerobot/scripts/control_robot.py teleoperate \
    --fps 30
```

- Record one episode in order to test replay:
```bash
python lerobot/scripts/control_robot.py record \
    --fps 30 \
    --root tmp/data \
    --repo-id $USER/koch_test \
    --num-episodes 1 \
    --run-compute-stats 0
```

- Visualize dataset:
```bash
python lerobot/scripts/visualize_dataset.py \
    --root tmp/data \
    --repo-id $USER/koch_test \
    --episode-index 0
```

- Replay this test episode:
```bash
python lerobot/scripts/control_robot.py replay \
    --fps 30 \
    --root tmp/data \
    --repo-id $USER/koch_test \
    --episode 0
```

- Record a full dataset in order to train a policy, with 2 seconds of warmup,
30 seconds of recording for each episode, and 10 seconds to reset the environment in between episodes:
```bash
python lerobot/scripts/control_robot.py record \
    --fps 30 \
    --root data \
    --repo-id $USER/koch_pick_place_lego \
    --num-episodes 50 \
    --warmup-time-s 2 \
    --episode-time-s 30 \
    --reset-time-s 10
```

**NOTE**: You can use your keyboard to control data recording flow.
- Tap right arrow key '->' to early exit while recording an episode and go to resseting the environment.
- Tap right arrow key '->' to early exit while resetting the environment and got to recording the next episode.
- Tap left arrow key '<-' to early exit and re-record the current episode.
- Tap escape key 'esc' to stop the data recording.
This might require a sudo permission to allow your terminal to monitor keyboard events.

**NOTE**: You can resume/continue data recording by running the same data recording command twice.
To avoid resuming by deleting the dataset, use `--force-override 1`.

- Train on this dataset with the ACT policy:
```bash
DATA_DIR=data python lerobot/scripts/train.py \
    policy=act_koch_real \
    env=koch_real \
    dataset_repo_id=$USER/koch_pick_place_lego \
    hydra.run.dir=outputs/train/act_koch_real
```

- Run the pretrained policy on the robot:
```bash
python lerobot/scripts/control_robot.py record \
    --fps 30 \
    --root data \
    --repo-id $USER/eval_act_koch_real \
    --num-episodes 10 \
    --warmup-time-s 2 \
    --episode-time-s 30 \
    --reset-time-s 10
    -p outputs/train/act_koch_real/checkpoints/080000/pretrained_model
```
"""

import argparse
import concurrent.futures
import json
import logging
import os
import platform
import shutil
import time
import traceback
from contextlib import nullcontext
from pathlib import Path
from threading import Event

import cv2
import torch
import tqdm
from omegaconf import DictConfig
from PIL import Image
from termcolor import colored
from dataclasses import dataclass, field, replace

# from safetensors.torch import load_file, save_file
# from le_studio.common.datasets.compute_stats import compute_stats
# from le_studio.common.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
# from le_studio.common.datasets.push_dataset_to_hub.aloha_hdf5_format import to_hf_dataset
# from le_studio.common.datasets.push_dataset_to_hub.utils import concatenate_episodes, get_default_encoding
# from le_studio.common.datasets.utils import calculate_episode_data_index, create_branch

from le_studio.common.datasets.video_utils import encode_video_frames

# from le_studio.common.policies.factory import make_policy
# from le_studio.common.robot_devices.robots.factory import make_robot
# from le_studio.common.robot_devices.robots.utils import Robot

from le_studio.common.robot_devices.utils import busy_wait
from le_studio.common.utils.utils import (
    get_safe_torch_device,
    init_hydra_config,
    init_logging,
    set_global_seed,
)

# from le_studio.scripts.eval import get_pretrained_policy_path
# from le_studio.scripts.push_dataset_to_hub import (
# push_dataset_card_to_hub,
# push_meta_data_to_hub,
# push_videos_to_hub,
# save_meta_data,
# )

from typing import Optional
from data_process.dataset.raw_dataset import RawDataset
from data_process.convert_all import save_dict_to_json_and_mp4
from robots.common import Robot, make_robot, make_robot_from_yaml

########################################################################################
# Utilities
########################################################################################


def say(text, blocking=False):
    # Check if mac, linux, or windows.
    if platform.system() == "Darwin":
        cmd = f'say "{text}"'
    elif platform.system() == "Linux":
        cmd = f'spd-say "{text}"'
    elif platform.system() == "Windows":
        cmd = (
            'PowerShell -Command "Add-Type -AssemblyName System.Speech; '
            f"(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{text}')\""
        )

    if not blocking and platform.system() in ["Darwin", "Linux"]:
        # TODO(rcadene): Make it work for Windows
        # Use the ampersand to run command in the background
        cmd += " &"

    os.system(cmd)


def save_image(img_tensor, frame_index, images_dir):
    img = Image.fromarray(img_tensor.numpy())
    path = images_dir / f"frame_{frame_index:06d}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(path), quality=100)


def none_or_int(value):
    if value == "None":
        return None
    return int(value)


def log_control_info(robot, dt_s, episode_index=None, frame_index=None, fps=None):
    log_items = []
    if episode_index is not None:
        log_items.append(f"ep:{episode_index}")
    if frame_index is not None:
        log_items.append(f"frame:{frame_index}")

    def log_dt(shortname, dt_val_s):
        nonlocal log_items, fps
        info_str = f"{shortname}:{dt_val_s * 1000:5.2f} ({1/ dt_val_s:3.1f}hz)"
        if fps is not None:
            actual_fps = 1 / dt_val_s
            if actual_fps < fps - 1:
                info_str = colored(info_str, "yellow")
        log_items.append(info_str)

    # total step time displayed in milliseconds and its frequency
    log_dt("dt", dt_s)

    for name in robot.leader_arms:
        key = f"read_leader_{name}_pos_dt_s"
        if key in robot.logs:
            log_dt("dtRlead", robot.logs[key])

    for name in robot.follower_arms:
        key = f"write_follower_{name}_goal_pos_dt_s"
        if key in robot.logs:
            log_dt("dtWfoll", robot.logs[key])

        key = f"read_follower_{name}_pos_dt_s"
        if key in robot.logs:
            log_dt("dtRfoll", robot.logs[key])

    for name in robot.cameras:
        key = f"read_camera_{name}_dt_s"
        if key in robot.logs:
            log_dt(f"dtR{name}", robot.logs[key])

    info_str = " ".join(log_items)
    logging.info(info_str)


def is_headless():
    """Detects if python is running without a monitor."""
    try:
        import pynput  # noqa

        return False
    except Exception:
        print(
            "Error trying to import pynput. Switching to headless mode. "
            "As a result, the video stream from the cameras won't be shown, "
            "and you won't be able to change the control flow with keyboards. "
            "For more info, see traceback below.\n"
        )
        traceback.print_exc()
        print()
        return True


########################################################################################
# Control modes
########################################################################################


def teleoperate(
    robot: Robot, fps: Optional[int] = None, teleop_time_s: Optional[float] = None
):
    # TODO(rcadene): Add option to record logs
    if not robot.is_connected:
        robot.connect()

    start_teleop_t = time.perf_counter()
    while True:
        start_loop_t = time.perf_counter()
        robot.teleop_step()

        if fps is not None:
            dt_s = time.perf_counter() - start_loop_t
            busy_wait(1 / fps - dt_s)

        dt_s = time.perf_counter() - start_loop_t
        log_control_info(robot, dt_s, fps=fps)

        if (
            teleop_time_s is not None
            and time.perf_counter() - start_teleop_t > teleop_time_s
        ):
            break


def record(
    robot: Robot,
    policy: Optional[torch.nn.Module] = None,
    hydra_cfg: Optional[DictConfig] = None,
    fps: Optional[int] = None,
    root="data",
    repo_id="lerobot/debug",
    warmup_time_s=2,
    episode_time_s=10,
    reset_time_s=5,
    num_episodes=50,
    video=True,
    run_compute_stats=True,
    push_to_hub=True,
    tags=None,
    num_image_writers_per_camera=4,
    force_override=False,
    start_episode=None,
):
    # TODO(rcadene): Add option to record logs
    # TODO(rcadene): Clean this function via decomposition in higher level functions

    # _, dataset_name = repo_id.split("/")
    # if dataset_name.startswith("eval_") and policy is None:
    #     raise ValueError(
    #         f"Your dataset name begins by 'eval_' ({dataset_name}) but no policy is provided ({policy})."
    #     )

    if not video:
        raise NotImplementedError()

    # if not robot.is_connected:
    #     robot.connect()

    local_dir = Path(root) / repo_id  # data/raw
    if local_dir.exists() and force_override:
        shutil.rmtree(local_dir)

    # episodes_dir = local_dir / "episodes"
    episodes_dir = local_dir
    episodes_dir.mkdir(parents=True, exist_ok=True)

    # videos_dir = local_dir / "videos"
    # videos_dir = episodes_dir
    # videos_dir.mkdir(parents=True, exist_ok=True)

    # Logic to resume data recording
    if start_episode is None:
        rec_info_path = episodes_dir / "data_recording_info.json"
        if rec_info_path.exists():
            with open(rec_info_path) as f:
                rec_info = json.load(f)
            episode_index = rec_info["last_episode_index"] + 1
        else:
            episode_index = 0
        start_episode = episode_index
    else:
        episode_index = start_episode

    if is_headless():
        logging.info(
            "Headless environment detected. On-screen cameras display and keyboard inputs will not be available."
        )

    # Allow to exit early while recording an episode or resetting the environment,
    # by tapping the right arrow key '->'. This might require a sudo permission
    # to allow your terminal to monitor keyboard events.

    class KeyboardHandler(object):
        def __init__(self) -> None:
            self.exit_early: bool = False
            self.rerecord_episode: bool = False
            self.stop_recording: bool = False
            self.save_event: Event = Event()

        def wait_save_once(self):
            self.save_event.wait()
            self.save_event.clear()

    keyer = KeyboardHandler()
    # Only import pynput if not in a headless environment
    if not is_headless():
        from pynput import keyboard

        def on_press(key):
            try:
                if key == "s":
                    print("\nSave current episode right now")
                    keyer.exit_early = True
                    keyer.save_event.set()
                elif key == "q":
                    print("Exiting loop and rerecord the last episode...")
                    keyer.exit_early = True
                    keyer.rerecord_episode = True
                    keyer.save_event.set()
                elif key == keyboard.Key.esc:
                    print("Escape key pressed. Stopping data recording...")
                    keyer.exit_early = True
                    keyer.stop_recording = True
            except Exception as e:
                print(f"Error handling key press: {e}")

        listener = keyboard.Listener(on_press=on_press)
        listener.start()

    # Load policy if any
    # if policy is not None:
    #     # Check device is available
    #     device = get_safe_torch_device(hydra_cfg.device, log=True)

    #     policy.eval()
    #     policy.to(device)

    #     torch.backends.cudnn.benchmark = True
    #     torch.backends.cuda.matmul.allow_tf32 = True
    #     set_global_seed(hydra_cfg.seed)

    #     # override fps using policy fps
    #     fps = hydra_cfg.env.fps

    get_videos_dir = lambda episode_index: episodes_dir / f"episode_{episode_index}"
    get_video_path = (
        lambda episode_index, key: get_videos_dir(episode_index) / f"{key}.mp4"
    )

    # Execute a few seconds without recording data, to give times
    # to the robot devices to connect and start synchronizing.
    timestamp = 0
    start_warmup_t = time.perf_counter()
    is_warmup_print = False
    while timestamp < warmup_time_s:
        if not is_warmup_print:
            logging.info("Warming up (no data recording)")
            say("Warming up")
            is_warmup_print = True

        start_loop_t = time.perf_counter()

        if policy is None:
            observation, action = robot.teleop_step(record_data=True)
        else:
            observation = robot.capture_observation()

        if not is_headless():
            image_keys = [key for key in observation if "image" in key]
            for key in image_keys:
                cv2.imshow(
                    key, cv2.cvtColor(observation[key].numpy(), cv2.COLOR_RGB2BGR)
                )
            cv2.waitKey(1)

        dt_s = time.perf_counter() - start_loop_t
        busy_wait(1 / fps - dt_s)

        dt_s = time.perf_counter() - start_loop_t
        log_control_info(robot, dt_s, fps=fps)

        timestamp = time.perf_counter() - start_warmup_t

    # Save images using threads to reach high fps (30 and more)
    # Using `with` to exist smoothly if an execption is raised.
    futures = []
    num_image_writers = num_image_writers_per_camera * len(robot.cameras)
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=num_image_writers
    ) as executor:
        # Start recording all episodes
        while episode_index < num_episodes:
            logging.info(f"Recording episode {episode_index}")
            say(f"Recording episode {episode_index}")
            videos_dir = get_videos_dir(episode_index)
            ep_dict = {}
            ep_dict["low_dim"] = {}
            frame_index = 0
            timestamp = 0
            start_episode_t = time.perf_counter()
            # Record one episode
            while timestamp < episode_time_s:
                start_loop_t = time.perf_counter()

                if policy is None:
                    observation, action = robot.teleop_step(record_data=True)
                else:
                    observation = robot.capture_observation()

                image_keys = [key for key in observation if "image" in key]
                # obs_not_image_keys = [key for key in observation if "image" not in key]
                low_dim_keys = list(observation["low_dim"].keys())

                get_tmp_imgs_dir = (
                    lambda episode_index, key: videos_dir
                    / f"{key}_episode_{episode_index:06d}"
                )

                # save temporal images as jpg files
                for key in image_keys:
                    tmp_imgs_dir = get_tmp_imgs_dir(episode_index, key)
                    futures += [
                        executor.submit(
                            save_image,
                            observation[key],
                            frame_index,
                            tmp_imgs_dir,
                        )
                    ]

                # show current images
                if not is_headless():
                    image_keys = [key for key in observation if "image" in key]
                    for key in image_keys:
                        cv2.imshow(
                            key,
                            cv2.cvtColor(observation[key].numpy(), cv2.COLOR_RGB2BGR),
                        )
                    cv2.waitKey(1)

                # add low dim observations and actions to the episode dict
                for key in low_dim_keys:
                    if key not in ep_dict["low_dim"]:
                        ep_dict["low_dim"][key] = []
                    ep_dict["low_dim"][key].append(observation[key])
                for key in action:
                    if key not in ep_dict:
                        ep_dict["low_dim"][key] = []
                    ep_dict["low_dim"][key].append(action[key])

                frame_index += 1

                dt_s = time.perf_counter() - start_loop_t
                busy_wait(1 / fps - dt_s)

                dt_s = time.perf_counter() - start_loop_t
                log_control_info(robot, dt_s, fps=fps)

                timestamp = time.perf_counter() - start_episode_t
                if keyer.exit_early:
                    keyer.exit_early = False
                    break

            if not keyer.stop_recording:
                # Start resetting env while the executor are finishing
                logging.info("Reset the environment")
                say("Reset the environment")

            timestamp = 0
            start_vencod_t = time.perf_counter()
            # During env reset we save the data and encode the videos

            with open(videos_dir / "low-dim.json", "w") as f:
                json.dump(ep_dict["low_dim"], f)

            num_frames = frame_index
            print(f"num_frames:{num_frames}")
            for key in image_keys:
                tmp_imgs_dir = get_tmp_imgs_dir(key)
                # fname = f"{key}_episode_{episode_index:06d}.mp4"
                video_path = get_video_path(episode_index, key)
                fname = video_path.stem + video_path.suffix
                if video_path.exists():  # overwrite existing video
                    video_path.unlink()
                # Store the reference to the video frame, even tho the videos are not yet encoded
                ep_dict[key] = []
                for i in range(num_frames):
                    ep_dict[key].append({"path": f"{fname}", "timestamp": i / fps})

            # save record information
            rec_info = {
                "last_episode_index": episode_index,
            }
            with open(rec_info_path, "w") as f:
                json.dump(rec_info, f)

            is_last_episode = keyer.stop_recording or (
                episode_index == (num_episodes - 1)
            )

            # Wait if necessary
            with tqdm.tqdm(total=reset_time_s, desc="Waiting") as pbar:
                while timestamp < reset_time_s and not is_last_episode:
                    time.sleep(1)
                    timestamp = time.perf_counter() - start_vencod_t
                    pbar.update(1)
                    if keyer.exit_early:
                        keyer.exit_early = False
                        break

            # Skip updating episode index which forces re-recording episode
            if keyer.rerecord_episode:
                keyer.rerecord_episode = False
                continue
            else:
                episode_index += 1
                if is_last_episode:
                    logging.info("Done recording")
                    say("Done recording", blocking=True)
                    if not is_headless():
                        listener.stop()

                    logging.info(
                        "Waiting for threads writing the images on disk to terminate..."
                    )
                    for _ in tqdm.tqdm(
                        concurrent.futures.as_completed(futures),
                        total=len(futures),
                        desc="Writting images",
                    ):
                        pass
                    break

    robot.exit()
    if not is_headless():
        cv2.destroyAllWindows()

    num_episodes = episode_index

    logging.info("Encoding all episode videos")
    say("Encoding videos")
    # Use ffmpeg to convert frames stored as .png files into mp4 videos
    for episode_index in tqdm.tqdm(range(num_episodes)):
        for key in image_keys:
            tmp_imgs_dir = get_tmp_imgs_dir(key)
            video_path = get_video_path(episode_index, key)
            if video_path.exists():
                # Skip if video is already encoded. Could be the case when resuming data recording.
                continue
            # note: `encode_video_frames` is a blocking call. Making it asynchronous shouldn't speedup encoding,
            # since video encoding with ffmpeg is already using multithreading.
            encode_video_frames(tmp_imgs_dir, video_path, fps, overwrite=True)
            shutil.rmtree(tmp_imgs_dir)

    logging.info("Exiting")
    say("Exiting")


def replay(
    robot: Robot,
    episode: int,
    fps: Optional[int] = None,
    root="data",
    repo_id="lerobot/debug",
):
    # TODO(rcadene): Add option to record logs
    local_dir = Path(root) / repo_id
    if not local_dir.exists():
        raise ValueError(local_dir)

    dataset = RawDataset(repo_id, root=root)
    dataset.warm_up_episodes([episode])
    if not hasattr(dataset, "select_columns"):
        setattr(dataset, "select_columns", dataset.hf_dataset.select_columns)
    items = dataset.select_columns("action")
    from_idx = dataset.episode_data_index["from"][episode]
    to_idx = dataset.episode_data_index["to"][episode]

    # if not robot.is_connected:
    #     robot.connect()

    logging.info("Replaying episode")
    say("Replaying episode", blocking=True)
    for idx in range(from_idx, to_idx):
        start_episode_t = time.perf_counter()

        action = items[idx]["action"]
        robot.send_action(action)

        dt_s = time.perf_counter() - start_episode_t
        busy_wait(1 / fps - dt_s)

        dt_s = time.perf_counter() - start_episode_t
        log_control_info(robot, dt_s, fps=fps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Set common options for all the subparsers
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument(
        "--robot-path",
        type=str,
        default="lerobot/configs/robot/koch.yaml",
        help="Path to robot yaml file used to instantiate the robot using `make_robot` factory function.",
    )
    base_parser.add_argument(
        "--robot-overrides",
        type=str,
        nargs="*",
        help="Any key=value arguments to override config values (use dots for.nested=overrides)",
    )

    parser_calib = subparsers.add_parser("calibrate", parents=[base_parser])
    parser_calib.add_argument(
        "--arms",
        type=str,
        nargs="*",
        help="List of arms to calibrate (e.g. `--arms left_follower right_follower left_leader`)",
    )

    parser_teleop = subparsers.add_parser("teleoperate", parents=[base_parser])
    parser_teleop.add_argument(
        "--fps",
        type=none_or_int,
        default=None,
        help="Frames per second (set to None to disable)",
    )

    parser_record = subparsers.add_parser("record", parents=[base_parser])
    parser_record.add_argument(
        "--fps",
        type=none_or_int,
        default=None,
        help="Frames per second (set to None to disable)",
    )
    parser_record.add_argument(
        "--root",
        type=Path,
        default="data",
        help="Root directory where the dataset will be stored locally at '{root}/{repo_id}' (e.g. 'data/hf_username/dataset_name').",
    )
    parser_record.add_argument(
        "--repo-id",
        type=str,
        default="lerobot/test",
        help="Dataset identifier. By convention it should match '{hf_username}/{dataset_name}' (e.g. `lerobot/test`).",
    )
    parser_record.add_argument(
        "--warmup-time-s",
        type=int,
        default=10,
        help="Number of seconds before starting data collection. It allows the robot devices to warmup and synchronize.",
    )
    parser_record.add_argument(
        "--episode-time-s",
        type=int,
        default=60,
        help="Number of seconds for data recording for each episode.",
    )
    parser_record.add_argument(
        "--reset-time-s",
        type=int,
        default=60,
        help="Number of seconds for resetting the environment after each episode.",
    )
    parser_record.add_argument(
        "--num-episodes", type=int, default=50, help="Number of episodes to record."
    )
    parser_record.add_argument(
        "--run-compute-stats",
        type=int,
        default=1,
        help="By default, run the computation of the data statistics at the end of data collection. Compute intensive and not required to just replay an episode.",
    )
    parser_record.add_argument(
        "--push-to-hub",
        type=int,
        default=1,
        help="Upload dataset to Hugging Face hub.",
    )
    parser_record.add_argument(
        "--tags",
        type=str,
        nargs="*",
        help="Add tags to your dataset on the hub.",
    )
    parser_record.add_argument(
        "--num-image-writers-per-camera",
        type=int,
        default=4,
        help=(
            "Number of threads writing the frames as png images on disk, per camera. "
            "Too much threads might cause unstable teleoperation fps due to main thread being blocked. "
            "Not enough threads might cause low camera fps."
        ),
    )
    parser_record.add_argument(
        "--force-override",
        type=int,
        default=0,
        help="By default, data recording is resumed. When set to 1, delete the local directory and start data recording from scratch.",
    )
    parser_record.add_argument(
        "-p",
        "--pretrained-policy-name-or-path",
        type=str,
        help=(
            "Either the repo ID of a model hosted on the Hub or a path to a directory containing weights "
            "saved using `Policy.save_pretrained`."
        ),
    )
    parser_record.add_argument(
        "--policy-overrides",
        type=str,
        nargs="*",
        help="Any key=value arguments to override config values (use dots for.nested=overrides)",
    )

    parser_replay = subparsers.add_parser("replay", parents=[base_parser])
    parser_replay.add_argument(
        "--fps",
        type=none_or_int,
        default=None,
        help="Frames per second (set to None to disable)",
    )
    parser_replay.add_argument(
        "--root",
        type=Path,
        default="data",
        help="Root directory where the dataset will be stored locally at '{root}/{repo_id}' (e.g. 'data/hf_username/dataset_name').",
    )
    parser_replay.add_argument(
        "--repo-id",
        type=str,
        default="lerobot/test",
        help="Dataset identifier. By convention it should match '{hf_username}/{dataset_name}' (e.g. `lerobot/test`).",
    )
    parser_replay.add_argument(
        "--episode", type=int, default=0, help="Index of the episode to replay."
    )

    args = parser.parse_args()

    init_logging()

    control_mode = args.mode
    robot_path = args.robot_path
    robot_overrides = args.robot_overrides
    kwargs = vars(args)
    del kwargs["mode"]
    del kwargs["robot_path"]
    del kwargs["robot_overrides"]

    # robot_cfg = init_hydra_config(robot_path, robot_overrides)
    # robot = make_robot(robot_cfg)
    robot = make_robot_from_yaml(robot_path, robot_overrides)
    # robot = make_robot(robot_path)

    if control_mode == "calibrate":
        raise NotImplementedError()
    elif control_mode == "teleoperate":
        teleoperate(robot, **kwargs)
    elif control_mode == "record":
        record(robot, **kwargs)
    elif control_mode == "replay":
        replay(robot, **kwargs)
    robot.exit()
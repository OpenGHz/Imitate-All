"""
Utilities to control a robot.

Useful to record a dataset, replay a recorded episode, run the policy on your robot
and record an evaluation dataset, and to recalibrate your robot if needed.
"""


import argparse
import concurrent.futures
import json
import logging
import shutil
import time
import traceback
from pathlib import Path
from threading import Event
from functools import partial
import cv2
import tqdm
from omegaconf import DictConfig
from PIL import Image
from termcolor import colored

from habitats.common.datasets.video_utils import encode_video_frames
from habitats.common.robot_devices.utils import busy_wait
from habitats.common.utils.utils import (
    init_logging,
)

from typing import Optional, Callable
from data_process.dataset.raw_dataset import RawDataset
from data_process.convert_all import concatenate_by_key, replace_keys
from robots.common import Robot, make_robot_from_yaml
from pprint import pprint
import numpy as np


########################################################################################
# Utilities
########################################################################################


def save_image(img: np.ndarray, frame_index, images_dir: Path):
    img = Image.fromarray(img)
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

    # for name in robot.leader_arms:
    #     key = f"read_leader_{name}_pos_dt_s"
    #     if key in robot.logs:
    #         log_dt("dtRlead", robot.logs[key])

    # for name in robot.follower_arms:
    #     key = f"write_follower_{name}_goal_pos_dt_s"
    #     if key in robot.logs:
    #         log_dt("dtWfoll", robot.logs[key])

    #     key = f"read_follower_{name}_pos_dt_s"
    #     if key in robot.logs:
    #         log_dt("dtRfoll", robot.logs[key])

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
    root: str,
    repo_id: str,
    fps: Optional[int] = None,
    episode_time_s=None,
    num_frames_per_episode=None,
    warmup_time_s=2,
    reset_time_s=5,
    num_episodes=50,
    video=True,
    num_image_writers_per_camera=4,
    force_override=False,
    start_episode=-1,
    policy: Optional[Callable] = None,
    hydra_cfg: Optional[DictConfig] = None,
    run_compute_stats=True,
    push_to_hub=True,
    tags=None,
    *args,
    **kwargs,
):
    # allow to record data within a specific time or number of frames
    assert (episode_time_s, num_frames_per_episode).count(None) == 1
    if episode_time_s is None:
        episode_time_s = np.inf
    elif num_frames_per_episode is None:
        num_frames_per_episode = np.inf

    if not video:
        raise NotImplementedError()

    local_dir = Path(root) / repo_id  # data/raw
    if local_dir.exists() and force_override:
        shutil.rmtree(local_dir)

    # episodes_dir = local_dir / "episodes"
    episodes_dir = local_dir
    episodes_dir.mkdir(parents=True, exist_ok=True)

    # Logic to resume data recording
    raw_start_episode = start_episode
    rec_info_path = episodes_dir / "data_recording_info.json"
    if start_episode < 0:
        start_episode += 1
        if rec_info_path.exists():
            with open(rec_info_path) as f:
                rec_info = json.load(f)
            episode_index = rec_info["last_episode_index"] + 1 + start_episode
        else:
            if start_episode < 0:
                logging.warning(
                    "No data recording info found. Starting from episode 0."
                )
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
            self._rerecord_episode: bool = False
            self._stop_recording: bool = False
            self.record_event: Event = Event()
            self._is_waiting_start_recording: bool = False
            self.is_dragging_mode: bool = False

        def show_instruction(self):
            print(
                """(Press:
                'Space Bar' to start recording the data,
                'q' to discard current recording or rerecording the last episode,
                'p' to print current arms' states,
                'g' to start/stop teaching mode,
                '0' to reset arms,
                'z' to exit the program after converting all saved images to mp4 videos,
                'ESC': to exit this program without converting data,
                'i' to show this instructions again.
            )"""
            )

        def wait_start_recording(self):
            self._is_waiting_start_recording = True
            self.record_event.wait()
            self.record_event.clear()
            self._is_waiting_start_recording = False
            return self._rerecord_episode, self._stop_recording

        def is_recording(self):
            return not self._is_waiting_start_recording

        def clear_rerecord(self):
            self._rerecord_episode = False

        def set_record_event(self):
            if not self.record_event.is_set():
                self.record_event.set()
                return True
            else:
                print("\n Something went wrong, recording data is already started")
                return False

        def on_press(self, key, robot: Robot = None):
            try:
                print()
                if key == keyboard.Key.space:
                    if (not self.is_recording()) and self.set_record_event():
                        robot.enter_passive_mode()
                        print("Start recording data")
                    else:
                        print("Still recording data, please wait or press 's' to save right now...")
                elif (key == keyboard.Key.esc) or (key.char == "z"):
                    print("Stopping data recording...")
                    self.exit_early = True
                    self._stop_recording = True
                    if key == keyboard.Key.esc:
                        self.no_convert = True
                    else:
                        self.no_convert = False
                    if not self.is_recording():
                        self.set_record_event()
                elif key.char == "s":
                    if self.is_recording():
                        print("Save current episode right now")
                        self.exit_early = True
                    else:
                        print("Not recording data, no need to save")
                elif key.char == "q":
                    print("Rerecord current episode...")
                    self._rerecord_episode = True
                    if self.is_recording():
                        self.exit_early = True
                    else:
                        self.set_record_event()
                        robot.enter_passive_mode()
                        print("Start recording data")
                elif key.char == "i":
                    self.show_instruction()
                elif key.char == "p":
                    pprint(robot.get_low_dim_data())
                elif key.char == "g":
                    if not self.is_recording():
                        if robot.get_state_mode() == "passive":
                            print("Stop teaching mode")
                            robot.enter_active_mode()
                        elif robot.get_state_mode() == "active":
                            print("Start teaching mode")
                            robot.enter_passive_mode()
                        else:
                            raise ValueError()
                    else:
                        print("Cannot switch mode while recording data")
                elif key.char == "0":
                    if not self.is_recording():
                        print("Reset robots")
                        robot.reset()
                    else:
                        print("Cannot reset robots while recording data")
                elif key.char == "c":
                    # used for clearing boundary errors
                    # robot.enter_active_mode()
                    robot.enter_passive_mode()
                else:
                    print(
                        "Unknown key pressed:",
                        key,
                        f"type:{type(key)}, str value {str(key)}",
                    )
            except Exception as e:
                # print(f"Error handling key press: {e}")
                print(
                    "Unknown key pressed:",
                    key,
                    f"type:{type(key)}, str value {str(key)}",
                )

        @property
        def rerecord_episode(self):
            return self._rerecord_episode

        @property
        def stop_recording(self):
            return self._stop_recording

    keyer = KeyboardHandler()
    # Only import pynput if not in a headless environment
    if not is_headless():
        from pynput import keyboard

        listener = keyboard.Listener(on_press=partial(keyer.on_press, robot=robot))
        listener.start()

    get_videos_dir = lambda episode_index: episodes_dir / f"{episode_index}"
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
            # say("Warming up")
            is_warmup_print = True

        start_loop_t = time.perf_counter()

        # if policy is None:
        #     observation, action = robot.teleop_step(record_data=True)
        # else:
        # observation = robot.get_low_dim_data()
        observation = robot.capture_observation()

        # logging.warning(f"0: get observation time: {((time.perf_counter() - start_loop_t) * 1000):.2f} ms")

        if not is_headless():
            image_keys = [key for key in observation if "image" in key]
            for key in image_keys:
                cv2.imshow(
                    key, cv2.cvtColor(observation[key], cv2.COLOR_RGB2BGR)
                )
            cv2.waitKey(1)

        dt_s = time.perf_counter() - start_loop_t
        busy_wait(1 / fps - dt_s)
        # logging.warning(f"0: warmup time: {((time.perf_counter() - start_loop_t) * 1000):.2f} ms")
        dt_s = time.perf_counter() - start_loop_t
        log_control_info(robot, dt_s, fps=fps)

        timestamp = time.perf_counter() - start_warmup_t

    # Save images using threads to reach high fps (30 and more)
    # Using `with` to exist smoothly if an execption is raised.
    futures = []
    camera_num = len(robot.cameras)
    num_image_writers = num_image_writers_per_camera * camera_num
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=num_image_writers
    ) as executor:

        def before_exit():
            logging.info("Done recording")
            # say("Done recording", blocking=True)
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

        # Show the instructions to the user
        keyer.show_instruction()
        # Start recording all episodes
        while episode_index < num_episodes:
            logging.info(
                f"Press 'Space Bar' to start recording episode {episode_index} or press 'q' to re-record the last episode {max(episode_index - 1, 0)}."
            )
            re_record, stop_record = keyer.wait_start_recording()
            if stop_record:
                before_exit()
                # episode_index = max(1, episode_index)
                break
            elif re_record:
                episode_index = max(episode_index - 1, 0)
                keyer.clear_rerecord()
                logging.info(f"Rerecording last episode {episode_index}")
            logging.info(f"Recording episode {episode_index}")
            videos_dir = get_videos_dir(episode_index)
            get_tmp_imgs_dir = (
                lambda episode_index, key: get_videos_dir(episode_index)
                / f"{key}_episode_{episode_index:06d}"
            )
            # say(f"Recording episode {episode_index}")
            ep_dict = {
                "low_dim": {},
                "images_timestamp": {},
            }
            frame_index = 0
            timestamp = 0
            start_episode_t = time.perf_counter()
            # Record one episode
            while timestamp < episode_time_s:
                start_loop_t = time.perf_counter()

                # if policy is None:
                #     observation, action = robot.teleop_step(record_data=True)
                # else:
                observation = robot.capture_observation()

                # logging.warning(f"1: get observation time: {((time.perf_counter() - start_loop_t) * 1000):.2f} ms")

                image_keys = [key for key in observation if "image" in key]
                # obs_not_image_keys = [key for key in observation if "image" not in key]
                low_dim_keys = list(observation["low_dim"].keys())
                low_dim_time_keys = [key for key in observation["low_dim"] if "time" in key]

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
                            cv2.cvtColor(observation[key], cv2.COLOR_RGB2BGR),
                        )
                    cv2.waitKey(1)

                # logging.warning(f"2: save image time + 1: {((time.perf_counter() - start_loop_t) * 1000):.2f} ms")

                # add low dim observations and actions to the episode dict
                for key in low_dim_keys:
                    if key not in ep_dict["low_dim"]:
                        ep_dict["low_dim"][key] = []
                    ep_dict["low_dim"][key].append(observation["low_dim"][key])
                for key in robot.cameras:
                    if key not in ep_dict["images_timestamp"]:
                        ep_dict["images_timestamp"][key] = []
                    ep_dict["images_timestamp"][key].append(
                        observation["/time/" + key]
                    )
                # for key in action:
                #     if key not in ep_dict:
                #         ep_dict["low_dim"][key] = []
                #     ep_dict["low_dim"][key].append(action[key])

                frame_index += 1

                dt_s = time.perf_counter() - start_loop_t
                busy_wait(1 / fps - dt_s)

                dt_s = time.perf_counter() - start_loop_t
                log_control_info(robot, dt_s, fps=fps)

                timestamp = time.perf_counter() - start_episode_t
                if keyer.exit_early:
                    keyer.exit_early = False
                    break
                elif frame_index >= num_frames_per_episode:
                    break

            if not keyer.stop_recording:
                # Start resetting env while the executor are finishing
                logging.info("Reset the environment")
                # say("Reset the environment")
                robot.reset()

            timestamp = 0
            start_vencod_t = time.perf_counter()
            # During env reset we save the data and encode the videos
            low_dim_timestamps = {}
            for key in low_dim_time_keys:
                low_dim_timestamps[key.replace("/time", "")] = ep_dict["low_dim"].pop(key)
            if len(low_dim_time_keys) == 1:
                low_dim_timestamps = low_dim_timestamps.popitem()[1]
            with open(videos_dir / "low_dim.json", "w") as f:
                json.dump(ep_dict["low_dim"], f)
            with open(videos_dir / "timestamps.json", "w") as f:
                timestamps = {
                    "low_dim": low_dim_timestamps,
                    "images": ep_dict["images_timestamp"],
                }
                json.dump(timestamps, f)
            num_frames = frame_index
            with open(videos_dir / "meta.json", "w") as f:
                json.dump(
                    {
                        "length": num_frames,
                        "fps": fps,
                    },
                    f,
                )
            print(f"episode_{episode_index} frame number: {num_frames}")
            for key in image_keys:
                tmp_imgs_dir = get_tmp_imgs_dir(episode_index, key)
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

            # check if current episode is the last one
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
                keyer.clear_rerecord()
                continue
            else:
                episode_index += 1
                if is_last_episode:
                    before_exit()
                    break

    if not is_headless():
        cv2.destroyAllWindows()

    num_episodes = episode_index

    if keyer.no_convert:
        logging.info("Exiting without converting images to videos")
        return

    # Encode all videos finally
    if start_episode < num_episodes - 1:
        logging.info(
            f"Encoding videos from episode {start_episode} to {num_episodes - 1}"
        )
    elif num_episodes == 0:
        logging.info("No episode recorded.")
    else:
        logging.info(f"Encoding video of episode {episode_index - 1}")
    # say("Encoding videos")
    # Use ffmpeg to convert frames stored as .png files into mp4 videos
    for episode_index in tqdm.tqdm(range(num_episodes)):
        for key in image_keys:
            tmp_imgs_dir = get_tmp_imgs_dir(episode_index, key)
            video_path = get_video_path(episode_index, key)
            if video_path.exists() and (
                raw_start_episode >= 0 and episode_index < raw_start_episode
            ):
                # Skip if video is already encoded. Could be the case when resuming data recording.
                logging.warning(
                    f"Video {video_path} already exists. Skipping encoding. If you want to re-encode, delete the video file before recording."
                )
                continue
            else:
                if not tmp_imgs_dir.exists():
                    logging.error(
                        f"Episode {episode_index} images path {tmp_imgs_dir} not found."
                    )
                else:
                    logging.info(
                        f"Encoding epidode_{episode_index} video to {video_path}"
                    )
                    # note: `encode_video_frames` is a blocking call. Making it asynchronous shouldn't speedup encoding,
                    # since video encoding with ffmpeg is already using multithreading.
                    encode_video_frames(
                        tmp_imgs_dir, video_path, fps, vcodec="libx264", overwrite=False
                    )
                    shutil.rmtree(tmp_imgs_dir)

    logging.info("Exiting")


def replay(
    robot: Robot,
    root:str,
    repo_id: str,
    start_episode: int,
    num_episodes:int,
    num_rollouts: int,
    fps: int,
):
    # TODO(rcadene): Add option to record logs
    local_dir = Path(root) / repo_id
    assert local_dir.exists(), f"Local directory not found: {local_dir}"
    logging.info(f"Loading dataset from {local_dir}")
    dataset = RawDataset(repo_id, root=root)

    for episode_index in range(start_episode, start_episode + num_episodes):
        logging.info(f"Replaying episode {episode_index}")

        dataset.warm_up_episodes([start_episode], low_dim_only=True)

        meta = dataset.raw_data[start_episode]["meta"]
        low_dim = dataset.raw_data[start_episode]["low_dim"]

        # concatenate different arms
        obs_keys = ["observation/arm/joint_position", "observation/eef/joint_position"]
        act_keys = replace_keys(obs_keys.copy(), "observation", "action")
        if obs_keys[0] not in low_dim:
            low_dim_concat:dict = robot.low_dim_concat
            low_dim_concat.update(replace_keys(low_dim_concat.copy(), "observation", "action"))
            low_dim = concatenate_by_key(low_dim, low_dim_concat, remove_ori=True)

        for roll in range(num_rollouts):
            # go to first frame using trajectory mode
            arm_qpos = low_dim[obs_keys[0]][0]
            eef_qpos = low_dim[obs_keys[1]][0]
            action = arm_qpos + eef_qpos
            logging.info("Moving to the first frame of the episode")
            robot.enter_traj_mode()
            robot.send_action(action)
            # time.sleep(1)
            key = input(f"Press Enter to replay. Episode: {episode_index} Number: {roll} or 'z and Enter' to exit current episode or 'x and Enter' to exit all episodes")
            if key in ["z", "Z"]:
                break
            elif key in ["x", "X"]:
                return
            logging.info("Replaying episode")
            robot.enter_servo_mode()
            for i in tqdm.tqdm(range(meta["length"])):
                start_episode_t = time.perf_counter()
                arm_qpos = low_dim[obs_keys[0]][i]
                eef_qpos = low_dim[obs_keys[1]][i]
                action = arm_qpos + eef_qpos
                # print("current joint:", robot.get_low_dim_data()["observation/arm/joint_position"])
                # print("target action:", action)
                robot.send_action(action)
                dt_s = time.perf_counter() - start_episode_t
                busy_wait(1.0 / fps - dt_s)
                dt_s = time.perf_counter() - start_episode_t
                # log_control_info(robot, dt_s, fps=fps)


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
    base_parser.add_argument(
        "--fps",
        type=none_or_int,
        help="Frames per second (set to None to disable)",
    )

    dataused_parser = argparse.ArgumentParser(add_help=False)
    dataused_parser.add_argument(
        "--root",
        type=Path,
        default="data",
        help="Root directory where the dataset will be stored locally at '{root}/{repo_id}' (e.g. 'data/hf_username/dataset_name').",
    )
    dataused_parser.add_argument(
        "--repo-id",
        type=str,
        default="raw/example",
        help="Dataset identifier. By convention it should match '{hf_username}/{dataset_name}' (e.g. `lerobot/test`).",
    )
    dataused_parser.add_argument(
        "--num-episodes", type=int, default=1, help="Number of episodes to record."
    )
    dataused_parser.add_argument(
        "--start-episode",
        type=int,
        help="Index of the first episode to record; value < 0 means get the last episode index from 'data_recording_info.json' and add (value + 1) to it.",
    )

    parser_teleop = subparsers.add_parser("teleoperate", parents=[base_parser])

    parser_record = subparsers.add_parser("record", parents=[base_parser, dataused_parser])
    parser_record.add_argument(
        "--warmup-time-s",
        type=int,
        default=10,
        help="Number of seconds before starting data collection. It allows the robot devices to warmup and synchronize.",
    )
    parser_record_length = parser_record.add_mutually_exclusive_group(required=True)
    parser_record_length.add_argument(
        "--episode-time-s",
        type=int,
        help="Number of seconds for data recording for each episode.",
    )
    parser_record_length.add_argument(
        "--num-frames-per-episode",
        type=int,
        help="Number of frames for data recording for each episode.",
    )
    parser_record.add_argument(
        "--reset-time-s",
        type=int,
        default=0,
        help="Number of seconds for resetting the environment after each episode.",
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

    parser_replay = subparsers.add_parser("replay", parents=[base_parser, dataused_parser])
    parser_replay.add_argument(
        "--num-rollouts",
        type=int,
        default=50,
        help="Number of times to replay each episode.",
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

    robot = make_robot_from_yaml(robot_path, robot_overrides)

    if control_mode == "teleoperate":
        teleoperate(robot, **kwargs)
    elif control_mode == "record":
        record(robot, **kwargs)
    elif control_mode == "replay":
        replay(robot, **kwargs)
    robot.exit()

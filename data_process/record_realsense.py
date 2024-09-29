import time
import pyrealsense2 as rs
import numpy as np
import cv2
import os
from pathlib import Path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
import subprocess
import argparse
from typing import Optional


def save_images_concurrently(
    imgs_array: np.ndarray, out_dir: Path, max_workers: int = 4
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def save_image(img_array, i, out_dir):
        img = Image.fromarray(img_array)
        img.save(str(out_dir / f"frame_{i:06d}.png"), quality=100)

    num_images = len(imgs_array)
    # for i in range(num_images):
    #     save_image(imgs_array[i], i, out_dir)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        [
            executor.submit(save_image, imgs_array[i], i, out_dir)
            for i in range(num_images)
        ]


def encode_video_frames(
    imgs_dir: Path,
    video_path: Path,
    fps: int,
    vcodec: str = "libsvtav1",
    pix_fmt: str = "yuv420p",
    g: Optional[int] = 2,
    crf: Optional[int] = 30,
    fast_decode: int = 0,
    log_level: Optional[str] = "error",
    overwrite: bool = False,
) -> None:
    """More info on ffmpeg arguments tuning on `benchmark/video/README.md`"""
    video_path = Path(video_path)
    video_path.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg_args = OrderedDict(
        [
            ("-f", "image2"),
            ("-r", str(fps)),
            ("-i", str(imgs_dir / "frame_%06d.png")),
            ("-vcodec", vcodec),
            ("-pix_fmt", pix_fmt),
        ]
    )

    if g is not None:
        ffmpeg_args["-g"] = str(g)

    if crf is not None:
        ffmpeg_args["-crf"] = str(crf)

    if fast_decode:
        key = "-svtav1-params" if vcodec == "libsvtav1" else "-tune"
        value = f"fast-decode={fast_decode}" if vcodec == "libsvtav1" else "fastdecode"
        ffmpeg_args[key] = value

    if log_level is not None:
        ffmpeg_args["-loglevel"] = str(log_level)

    ffmpeg_args = [item for pair in ffmpeg_args.items() for item in pair]
    if overwrite:
        ffmpeg_args.append("-y")

    ffmpeg_cmd = ["ffmpeg"] + ffmpeg_args + [str(video_path)]
    # redirect stdin to subprocess.DEVNULL to prevent reading random keyboard inputs from terminal
    subprocess.run(ffmpeg_cmd, check=True, stdin=subprocess.DEVNULL)

    if not video_path.exists():
        raise OSError(
            f"Video encoding did not work. File not found: {video_path}. "
            f"Try running the command manually to debug: `{''.join(ffmpeg_cmd)}`"
        )


class Camera(object):
    """
    realsense相机处理类
    """

    def __init__(self, width=1280, height=720, fps=30):
        self.width = width
        self.height = height
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(
            rs.stream.color, self.width, self.height, rs.format.bgr8, fps
        )
        self.config.enable_stream(
            rs.stream.depth, self.width, self.height, rs.format.z16, fps
        )
        # self.align = rs.align(rs.stream.color) # depth2rgb
        self.pipeline.start(self.config)  # 开始连接相机

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()  # 获得frame (包括彩色，深度图)
        # 创建对齐对象
        align_to = rs.stream.color  # rs.align允许我们执行深度帧与其他帧的对齐
        align = rs.align(align_to)  # “align_to”是我们计划对齐深度帧的流类型。
        aligned_frames = align.process(frames)
        # 获取对齐的帧
        aligned_depth_frame = (
            aligned_frames.get_depth_frame()
        )  # aligned_depth_frame是对齐的深度图
        color_frame = aligned_frames.get_color_frame()
        colorizer = rs.colorizer()
        depthx_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        colorizer_depth = np.asanyarray(
            colorizer.colorize(aligned_depth_frame).get_data()
        )
        return color_image, depthx_image, colorizer_depth

    def release(self):
        self.pipeline.stop()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-in", "--imgs_dir", type=str, default="data/Realsense/images")
    parser.add_argument(
        "-out", "--video_dir", type=str, default="data/Realsense/rgb_data"
    )
    parser.add_argument("--fps", type=int, required=True)
    parser.add_argument("--vcodec", type=str, default="libx264")
    parser.add_argument("--pix_fmt", type=str, default="yuv420p")
    parser.add_argument("--g", type=int, default=2)
    parser.add_argument("--crf", type=int, default=30)
    parser.add_argument("--fast_decode", type=int, default=0)
    parser.add_argument("--log_level", type=str, default="error")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    # 视频保存路径
    imgs_dir = args.imgs_dir
    video_dir = args.video_dir
    fps = args.fps
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs("data/Realsense/depth_data", exist_ok=True)
    os.makedirs("data/Realsense/depthcolor_data", exist_ok=True)
    os.makedirs("data/Realsense/camera_colordepth", exist_ok=True)
    os.makedirs(imgs_dir, exist_ok=True)
    video_path = f"{video_dir}/{int(time.time())}.mp4"
    video_depth_path = f"data/Realsense/depth_data/{int(time.time())}_depth.mp4"
    video_depthcolor_path = (
        f"data/Realsense/depthcolor_data/{int(time.time())}_depthcolor.mp4"
    )
    video_depthcolor_camera_path = (
        f"data/Realsense/camera_colordepth/{int(time.time())}_depthcolor.mp4"
    )
    # 初始化参数
    w, h = 640, 480
    mp4 = cv2.VideoWriter_fourcc(*"mp4v")  # 视频格式
    # 视频保存而建立对象
    color_images = []
    # wr_color = cv2.VideoWriter(video_path, mp4, fps, (w, h), isColor=True)
    wr_depth = cv2.VideoWriter(video_depth_path, mp4, fps, (w, h), isColor=False)
    wr_depthcolor = cv2.VideoWriter(
        video_depthcolor_path, mp4, fps, (w, h), isColor=True
    )
    wr_camera_colordepth = cv2.VideoWriter(
        video_depthcolor_camera_path, mp4, fps, (w, h), isColor=True
    )

    cam = Camera(w, h, fps)
    print("录制视频请按: s, 保存视频或退出请按：q")
    flag_V = 0
    cv2.namedWindow("RealSense", cv2.WINDOW_AUTOSIZE)
    while True:
        # 读取图像帧，包括RGB图和深度图
        color_image, depth_image, colorizer_depth = cam.get_frame()
        # 深度图数据格式转换，uint16→uint8
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
        )
        # images = np.hstack((color_image, depth_colormap))
        cv2.imshow("RealSense", color_image)

        key = cv2.waitKey(1)
        if key == ord("s"):
            flag_V = 1
            print("...录制视频中...")
        elif key == ord("q"):
            print(f"Saving images concurrently to {imgs_dir}, frames:", len(color_images))
            images_array = np.array(color_images)
            save_images_concurrently(
                images_array, out_dir=Path(imgs_dir), max_workers=4
            )
            # Encode a video from a directory of images
            encode_video_frames(
                Path(imgs_dir),
                Path(video_path),
                fps,
                args.vcodec,
                args.pix_fmt,
                args.g,
                args.crf,
                args.fast_decode,
                args.log_level,
                args.overwrite,
            )
            print("...录制结束...")
            break
        elif key == 27:
            print("不保存退出")
            break
        elif key != -1:
            print(f"按键{key}无功能")
        if flag_V == 1:
            color_images.append(color_image.copy())
            # wr_depth.write(depth_image)  # 保存基于灰度深度图
            # wr_depthcolor.write(depth_colormap)  # 保存计算所得着色深度图
            # wr_camera_colordepth.write(colorizer_depth)  # 保存相机自行计算的着色深度图

    # 释放资源
    cv2.destroyAllWindows()
    wr_depthcolor.release()
    wr_depth.release()
    wr_camera_colordepth.release()
    cam.release()

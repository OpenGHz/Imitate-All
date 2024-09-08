import time
import cv2
import tkinter as tk
from tkinter import ttk
from threading import Thread
import os
import numpy as np
import json

# 视频路径
video_folder = 'demonstrations/raw/put_objects_into_bowl_2/35/'
video_path = video_folder + '0.avi'
json_path = video_folder + 'records.json'

class VideoPlayer:
    def __init__(self, root, video_path, json_path):
        self.root = root
        self.video_path = video_path
        self.json_path = json_path
        self.cap = cv2.VideoCapture(video_path)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.duration = self.frame_count / self.fps

        self.current_frame = 0
        self.is_playing = True
        self.is_running = True

        # 从 JSON 文件中读取标记数组
        self.mark_array = self.load_mark_array_from_json()

        # Tkinter window settings
        self.root.title("Video Player")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Create video display window
        self.video_label = tk.Label(self.root)
        self.video_label.pack()

        # Create progress bar
        self.progress = tk.IntVar()
        self.progress_bar = ttk.Scale(self.root, from_=0, to=self.frame_count, variable=self.progress, orient='horizontal')
        self.progress_bar.pack(fill=tk.X, padx=10, pady=5)

        # Create Play/Pause button
        self.play_pause_button = tk.Button(self.root, text="Pause", command=self.toggle_play_pause)
        self.play_pause_button.pack()

        # Bind keyboard events
        self.root.bind("<KeyPress>", self.on_key_press)

        # Create thread to play video
        self.play_thread = Thread(target=self.play_video)
        self.play_thread.start()

        # Bind progress bar change event
        self.progress_bar.bind('<ButtonRelease-1>', self.on_progress_change)

    def load_mark_array_from_json(self):
        """从 JSON 文件中读取标记数组"""
        if not os.path.exists(self.json_path):
            print(f"File {self.json_path} not found.")
            return np.zeros(self.frame_count, dtype=int)

        try:
            with open(self.json_path, 'r') as f:
                data = json.load(f)
                return np.array(data["/observations/stage"], dtype=int)
        except Exception as e:
            print(f"Failed to load JSON: {e}")
            return np.zeros(self.frame_count, dtype=int)

    def toggle_play_pause(self):
        # Play/pause the video
        if self.is_playing:
            self.is_playing = False
            self.play_pause_button.config(text="Play")
        else:
            self.is_playing = True
            self.play_pause_button.config(text="Pause")

    def play_video(self):
        while self.is_running and self.cap.isOpened():
            if self.is_playing:
                ret, frame = self.cap.read()
                if not ret:
                    break

                # 获取当前帧的索引，并确保它不会超出边界
                self.current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

                # 防止当前帧超出标记数组的边界
                if self.current_frame >= len(self.mark_array):
                    self.current_frame = len(self.mark_array) - 1

                # 获取当前帧的标记值
                mark_value = self.mark_array[self.current_frame]

                # Update the frame in the progress bar
                self.progress.set(self.current_frame)

                # Convert the frame to display in tkinter
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = cv2.resize(frame, (800, 600))
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                # 在帧上显示标记值
                cv2.putText(img, f"Mark: {mark_value}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                self.photo = tk.PhotoImage(data=cv2.imencode('.png', img)[1].tobytes())
                self.video_label.config(image=self.photo)

                time.sleep(0.1)

            self.root.update_idletasks()

    def on_progress_change(self, event):
        # Seek to the selected frame
        new_frame = int(self.progress.get())
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)

    def on_key_press(self, event):
        # 绑定空格键来暂停和播放视频
        if event.keysym == 'space':
            self.toggle_play_pause()

    def on_close(self):
        # Stop the video and close the window
        self.is_running = False
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    # Check if video file exists
    if not os.path.exists(video_path):
        print(f"Video file {video_path} not found!")
    else:
        # Start the video player
        root = tk.Tk()
        player = VideoPlayer(root, video_path, json_path)
        root.mainloop()
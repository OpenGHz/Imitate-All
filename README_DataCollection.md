# Data Collection and Replay


## Environment Setup

!!! tip "Supported Operation Systems"

    * Ubuntu 20.04 LTS AMD64(x86_64)

Install the required python packages:
```bash
pip install -r requirements/data_collection.txt
```

## Data Collection

> The task name should be reasonable, and it is recommended to include time in the name to distinguish the same task data collected at different times.

### Starting Robotic Arms
1. Prepare all teacher-follower robotic arms.
2. Connect the power sources of all robotic arms (order doesn't matter).
3. **First**, connect the teacher arm via Type-C data cable (CAN0), **then** connect the follower arm (CAN1). For dual-arm operations, follow the above sequence for the left-side robotic arm first, then the right-side arm.
4. Long-press the power button on each robotic arm to turn them on.
5. Ensure that the robotic arms are at the zero pose; otherwise, perform a zero calibration.
6. Start the control service of all the robots.

> Note: Other devices connected to your computer may occupy the CAN interfaces, you may need to change the default can interfaces manually. Please refer to [Explanation of Parameters](#explanation-of-parameters).

### Connecting Cameras
Data collection typically requires multiple cameras, and the connection order can be arbitrary, for example:

- Single-arm task: Left arm-mounted camera -> Right arm-mounted camera -> Environment camera
- Dual-arm task: Left arm camera -> Right arm camera -> Environment top camera -> Environment bottom camera

> Note: Your computer may not be able to activate too many cameras at the same time.

### Modify Default Configurations
The defualt configurations are in `configurations/basic_configs/example/robot/airbot_play_demonstration.yaml` which is used for one long airbot_play with a E2B encoder to teleoprate one long airbot_play with a G2 gripper and one USB camera. You can modify these according to your usage, for example:

- One short Play teacher with one short follower:
    ```
    leader_arm_type: ["play_short"]
    follower_arm_type: ["play_short"]
    ```
- Two short Play teachers with two short followers:
    ```
    leader_arm_type: ["play_short"， "play_short"]
    follower_arm_type: ["play_short"， "play_short"]
    leader_end_effector: ["E2B"， "E2B"]
    follower_end_effector: ["G2"， "G2"]
    leader_can_interface: ["can0", "can2"]
    follower_can_interface: ["can1", "can3"]
    leader_domain_id: [33, 34]
    follower_domain_id: [44, 45]
    start_arm_joint_position: [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    start_eef_joint_position: [0.0, 0.0]
    ```

### Starting Data Collection / Demonstration

The common usage of parameters for data collection command is as follows:

```bash
python3 control_robot.py record \
  --robot-path configurations/basic_configs/example/robot/airbot_play_demonstration.yaml \
  --fps 25 \
  --root data \
  --repo-id raw/example \
  --warmup-time-s 1 \
  --num-frames-per-episode 500 \
  --reset-time-s 1 \
  --num-episodes 5 \
  --start-episode 0 \
  --num-image-writers-per-camera 1
```

#### Explanation of Parameters

  - --`robot-path`: path to the robot yaml file used to instantiate the robot
  - --`fps`: frames per second
  - --`root`: root directory where the dataset will be stored locally at `{root}/{repo_id}`
  - --`repo-id`: dataset identifier
  - --`warmup-time-s`: number of seconds before starting data collection
  - --`num-frames-per-episode`: number of frames for data recording for each episode
  - --`reset-time-s`: number of seconds for resetting the environment after each episode
  - --`num-episodes`: number of episodes to record
  - --`start-episode`: index of the first episode to record
  - --`num-image-writers-per-camera`: number of threads writing the frames as png images on disk, per camera

#### Key Descriptions

> Note: Avoid continuous key presses to prevent repeated behavior.

After excuting the command above, you can use the keyboard of your computer to control. The key descriptions are as follows:

- `Space Bar`: Start recording one episode data. The data will be automatically saved upon recording all frames.
- `s`: Stop current recording and save the data before recording all frames. This will result in different recorded data lengths
- `q`: Discard current recording or rerecording the last episode
- `g`: Toggle demonstration mode on/off
- `0`: Set the robotic arms to the initial states
- `p`: Print current robot states information in terminal
- `i`: Show these instructions again
- `z`: Exit this program after converting all saved images to mp4 videos
- `ESC`: Exit this program without converting data

#### Operational Steps

1. Start the program, and the real-time windows of each camera will appear (if not all cameras start, try adjusting device numbers or checking camera connections; try connecting only one camera per docking station; some computers may support only 1-2 external cameras when USB ports share the same bus, consider changing the computer).
2. Press `Space Bar` to start recording data and teleoperating the robotic arms to complete the target task.
3. After task completion, wait to collect the specified number of frames (the number of frames actually spent to complete the task should be as close as possible to the maximum collection frames):
    - If the demonstration opration is not acceptable, press `q` to discard the current recording process, then press `0` to control the robotic arm to return to the initial position.
    - If it is acceptable, press `Space Bar` to save the current teaching record. After saving, the robotic arm will automatically return to the initial position.
4. (Optional) In the `data/raw/<task_name>` folder in the current directory, check the recorded episodes. Each collected episode data includes:
    - `mp4` videos of the cameras
    - `low_dim.json` contains the robots' states
    - `meta.json` contains number of frames and collecting fps

**Additional Notes:**

1. Try to ensure the task are completed **just before reaching the maximum frame count**, i.e., do not end the action too early.

2. The robotic arm movement speed should not be too fast; otherwise, the collected information will be sparse, and the image quality will not be high.

3. It is recommended to store the collected task data folder <task_name> in the same directory structure on a **portable hard drive** as a backup.

## Data Replay

### Modify Default Configurations

The defualt configurations are in `configurations/basic_configs/example/robot/airbots/play/airbot_play.yaml` which is used for one long airbot_play with a G2 gripper. You can modify these according to your usage, for example:



### Starting Data Replay

> Data replay can be used to verify if there are issues with collected data, init states of the environment, etc. (requires prior [Environment Setup](#environment-setup)).

The data replay command and its parameters are as follows:

```bash
python3 control_robot.py replay \       
  --robot-path configurations/basic_configs/example/robot/airbot_play.yaml \
  --fps 25 \
  --root data \
  --repo-id raw/example \
  --episode 0
```

**Parameter explanation:**

- `-tn`: Specify the task name.
- `-ei`: Specify the ID corresponding to the HDF file.
- `-can`: CAN ID of the follower arms. For dual-arm tasks, you can set `-can 1 3` if you followed the connection order in [Starting-Robotic-Arms](#starting-robotic-arms). Also, you can unplug all the Type-C wires first and only connect the follower arms, then set `-can 0 1`.
- `-ii`: Do not replay camera data.
- `-ia`: Do not replay action data.
- `-rn`: For dual-arm tasks, specify `-rn` 2.
- `-cn`: The name of the cameras data to be replayed. For example, if there are two cameras, specify `-cn 0 1`.

After executing the command above, you can see the instruction ``Press Enter to start replay or z and Enter to exit...`` in the terminal. Then press `Enter` to replay. And you will see the same prompt after the current replay ends.

# ACT: Action Chunking with Transformers

## Introduction

**ALOHA Project Website**: https://tonyzhaozh.github.io/aloha/

This repository contains the codes for training and evaluating the ACT model. Make sure your computer has NVIDIA graphics card (memory less than 16G may not be able to train the model) and the `nvidia-smi` command is ready (driver installed).

The data for training should be collected by [airbot_aloha](https://discover-robotics.github.io/docs/sdk/aloha/).

It is recomended to use **anaconda** to manage python environments. You can download and install it by running the following commands(if download very slowly, you can [click here](https://mirrors.bfsu.edu.cn/anaconda/miniconda/Miniconda3-py38_4.9.2-Linux-x86_64.sh) to download manually):

```bash
wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py38_4.9.2-Linux-x86_64.sh
chmod u+x Miniconda3-py38_4.9.2-Linux-x86_64.sh && ./Miniconda3-py38_4.9.2-Linux-x86_64.sh
```

Restart your terminal and you can now use conda:
```bash
conda config --set auto_activate_base false && conda deactivate
```

## Repo Structure

- ``imitate_episodes.py`` Train and Evaluate ACT
- ``policy.py`` An adaptor for ACT policy
- ``detr`` Model definitions of ACT, modified from DETR
- ``sim_env.py`` Mujoco + DM_Control environments with joint space control
- ``ee_sim_env.py`` Mujoco + DM_Control environments with EE space control
- ``scripted_policy.py`` Scripted policies for sim environments
- ``constants.py`` Constants shared across files
- ``utils.py`` Utils such as data loading and helper functions
- ``visualize_episodes.py`` Save videos from a .hdf5 dataset
- ``robot_utils.py`` Useful robot tools to record images and process data.
- ``task_configs`` Configs for tasks training and evaluating.

## Installation

It is recommended to use a conda python environment. If you do not have one, create and activate it by using the following commands:

```bash
conda create -n aloha python=3.8.10 && conda activate aloha
```

Get the `act.zip` package from our customer service and then install the necessary packages by running the following commands:

```bash
unzip act.zip && cd act
pip install opencv-python h5py scipy robotics_tools -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -e ./detr -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

What's more, for policy evaluation, make sure you have finished the [AIRBOT Play ALOHA Environment Setup](https://discover-robotics.github.io/docs/sdk/aloha/#environment-setup).

## Parameter Configuration (important)

**Before training or inference**, parameter configuration is necessary. **Create a Python file in the act/task_configs directory with the same name as the task ( not recommended to modify or rename the example_task.py file directly)** to configure the task. This configuration mainly involves modifying various paths (using the replace_task_name function to **use default paths** or manually specifying paths), camera names (camera_names), robot number (robot_num, **set to 2 for dual-arm tasks**), and so on. Below is an example from example_task.py, which demonstrates how to modify configs based on the default configuration in template.py without needing to rewrite everything (for more adjustable configurations, refer to act/task_configs/template.py):

![](images/basic_config.png)

When training with default paths, place the converted HDF5 format data in the act/data/hdf5/<task_name> folder. You can create the directory with the following command:

```bash
mkdir -p data/hdf5
```

You can then copy the data manually or using a command like this (remember to modify the paths in the command):

```bash
mv path/to/your/task/hdf5_file data/hdf5
```

If CKPT_DIR and STATS_PATH don't exist, they will be automatically created and relevant files will be written.

## Model Training

> Please complete [Model Training Environment Setup](#installation) and [Parameter Configuration](#parameter-configuration) first (training with at least 2 data instances is required; otherwise, an error will occur due to the inability to split the training and validation sets).

Navigate to the act folder and activate the Conda environment:

```bash
conda activate aloha
```

Then run the training command:

```bash
python3 policy_train.py -tn example_task
```

The above commands, with just `-tn` args, will **use the configurations from the `.py` config files in the `task_configs` folder corresponding to the given task name**. If you use command-line parameters (not all parameters support command-line configuration, use `-h` arg to show all supported args), they will override the configurations in the config file. This allows for temporary parameter changes but is not recommended for regular use.

After training, by default, you can find two folders in `./my_ckpt/<task_name>/<time_stamp>` directory. The `ckpt` folder contains all weight files (referred to as the **process folder**), while the folder with the same name as `<task_name>` (called the **core folder**) contains the following files:

- Final weights and optimal weights: `policy_best.ckpt` and `policy_last.ckpt` respectively.
- Statistical data: `dataset_stats.pkl`.
- Crucial training information (including initial joint angles, training parameter configurations, etc.): `key_info.pkl`.
- The training loss curves: `train_val_kl_seed_0.png`, `train_val_l1_seed_0.png` and `train_val_loss_seed_0.png`.
- The simple description of the training result: `description.txt`, such as `Best ckpt: val loss 0.174929 @ epoch9499 with seed 0`.

<p align="center">
  <img src="images/train_output_structure.png" />
</p>

For ease of use in the future, it's recommended to **store the core folder in the specified disk's ALOHA/my_ckpt folder**.

**Training Parameter Tuning Recommendations:**
>
> \- Chunk size is the most important param to tune when applying ACT to a new environment. One chunk should correspond to \~1 secs wall-clock robot motion. 
>
> \- High KL weight (10 or 100), or train ***without*** CVAE encoder.
>
> \- Consider removing temporal\_agg and increase query frequency [here](https://github.com/tonyzhaozh/act/blob/main/imitate_episodes.py#L193) to be the same as your chunk size. I.e. each chunk is executed fully.
>
> \- train for **very long** (well after things plateaus, see picture)
>
> \- Try to increase batch size as much as possible, and increase lr accordingly. E.g. batch size 64 with learning rate 5e-5 versus batch size 8 and learning rate 1e-5
>
> \- Have separate backbones for each camera (requires changing the code, see [this commit](https://github.com/tonyzhaozh/act/commit/20fc6e990698534b89a41c2c85f54b7ff4b0c280))
>
> \- L1 loss > L2 loss (not precise enough)
>
> \- Abs position control > delta/velocity control (harder to recover)
>
> \- Try multiple checkpoints
>
> For real-world experiments:
>
> \- Train for even longer (5k - 8k epochs, especially if multi-camera)
>
> \- If inference is too slow -> robot moving slowly: disable temporal\_agg and increase query frequency [here](https://github.com/tonyzhaozh/act/blob/main/imitate_episodes.py#L193). We tried as high as 20.

## Model Evaluating

### Environment Preparation
- First, unplug both the teaching arm and execution arm's USB interfaces to refresh the CAN interface. Then, only connect the execution arm's USB interface (this way, the execution arm will use CAN0).
- Connect the cameras in the same order as that of data collection and so if you haven't unplugged them since data collection, you can skip this step.
- Long-press the power button of each robotic arm to turn them on.

<p align="center">
  <img src="images/robot_arm_connect.png" />
</p>

### Executing Commands

Navigate to the act folder and activate the conda environment:

```bash
conda activate aloha
```

Here are the evaluation command and its parameters:

```bash
python3 policy_evaluate.py -tn example_task -ci 0 -ts 20240322-194244
```

- -ci: Camera device numbers, corresponding to the device order of the configured camera names. For example, if two cameras are used and their id are 2 and 4, specify `-ci 2 4`.
- -ts: Timestamp corresponding to the task (check the path where model training results are saved, e.g., ```act/my_ckpt/example_task/20240325-153007```).
- -can: Specify which CAN to use for control; default is CAN0. Change to CAN1 with -can can1, for example. For dual-arm tasks, specify multiple cans like ```-can can0 can1```.
- -cki: Don't start the robotic arm, only show captured camera images, useful for verifying if the camera order matches the data collection order.

After the robotic arm starts and moves to the initial pose defined by the task, you can see the instructions in the terminal: **press Enter to start inference** and press z and then press Enter to end inference and exit. The robotic arm will return to the zero pose before the program exiting.

After each evaluation, you can find evaluation-related files (including process videos) in the corresponding timestamp folder inside the eval_results folder in the current directory.

## Information Viewing

After model training, key information is stored in the key_info.pkl file, which can be viewed using the following steps.

Navigate to the act folder and activate the conda environment:

```bash
conda activate aloha
```

Then, use the following command to view information for a specified timestamp where training results are saved:

```bash
python3 policy_train.py -tn example_task -sti 20240420-214215
```

You will see information related to that task in the terminal, including:

![](images/train_info.png)

This includes the absolute path to the HDF5 data used during training, training parameter configurations, initial joint angles for inference, and other information. 

This information ensures experiment reproducibility. If the camera is rigidly attached to the robotic arm, replicating the robotic arm's behavior is relatively straightforward. Object placement can be determined through retraining data replication.

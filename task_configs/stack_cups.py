from task_configs.template import (
    replace_task_name,
    TASK_CONFIG_DEFAULT,
    set_paths,
    activator,
)
from task_configs.config_augmentation.image.basic import color_transforms_2

def policy_maker(policy_class, policy_config):
    print("custom policy_maker")
    return None

def policy_maker(config:dict, stage=None):
    from policies.act.act import ACTPolicy
    from policies.common.maker import post_init_policies
    import logging
    import torch
    # TODO: add stage param to the policy class for convenience and simplicity
    # that is, do the following in the policy class __init__ method.
    policy = ACTPolicy(config)
    post_init_policies([policy], stage, [config["ckpt_path"]])
    return policy

def environment_maker(config:dict):
    from envs.make_env import make_environment
    env_config = config["environments"]
    # TODO: use env_config only
    return make_environment(config)

@activator(False)
def augment_images(image):
    return color_transforms_2(image)

# replace the task name in the default paths when using the default paths such as this example
# auto replace by the file name
TASK_NAME = __file__.split("/")[-1].replace(".py", "")
replace_task_name(TASK_NAME, stats_name="dataset_stats.pkl", time_stamp="now")
# but we also show how to replace the paths manually
# DATA_DIR = f"./data/hdf5/{TASK_NAME}"
# CKPT_DIR = f"./my_ckpt/{TASK_NAME}/ckpt"
# STATS_PATH = f"./my_ckpt/{TASK_NAME}/dataset_stats.pkl"
# replace_paths(DATA_DIR, CKPT_DIR, STATS_PATH)  # replace the default data and ckpt paths


chunk_size = 25
joint_num = 7

TASK_CONFIG_DEFAULT["common"]["camera_names"] = ["0"]
TASK_CONFIG_DEFAULT["common"]["robot_num"] = 1
TASK_CONFIG_DEFAULT["common"]["policy_config"]["temporal_agg"] = True

TASK_CONFIG_DEFAULT["common"]["policy_config"]["policy_maker"] = policy_maker

TASK_CONFIG_DEFAULT["common"]["state_dim"] = joint_num
TASK_CONFIG_DEFAULT["common"]["action_dim"] = joint_num
TASK_CONFIG_DEFAULT["common"]["policy_config"]["chunk_size"] = chunk_size
TASK_CONFIG_DEFAULT["common"]["policy_config"]["num_queries"] = chunk_size
TASK_CONFIG_DEFAULT["common"]["policy_config"]["kl_weight"] = 10


TASK_CONFIG_DEFAULT["train"]["num_episodes"] = (200, 299)
TASK_CONFIG_DEFAULT["train"]["num_epochs"] = 7000
TASK_CONFIG_DEFAULT["train"]["learning_rate"] = 2e-5
TASK_CONFIG_DEFAULT["train"]["pretrain_ckpt_path"] = "/home/ghz/airbot_play/act/my_ckpt/stack_cups/20240621-022635/stack_cups/policy_best.ckpt"
TASK_CONFIG_DEFAULT["train"]["pretrain_epoch_base"] = "AUTO"
TASK_CONFIG_DEFAULT["train"]["batch_size"] = 16

TASK_CONFIG_DEFAULT["eval"]["robot_num"] = 1
TASK_CONFIG_DEFAULT["eval"]["joint_num"] = joint_num
TASK_CONFIG_DEFAULT["eval"]["start_joint"] = "AUTO"
TASK_CONFIG_DEFAULT["eval"]["max_timesteps"] = 400
TASK_CONFIG_DEFAULT["eval"]["ensemble"] = None
TASK_CONFIG_DEFAULT["eval"]["environments"]["environment_maker"] = environment_maker

# final config
TASK_CONFIG = TASK_CONFIG_DEFAULT

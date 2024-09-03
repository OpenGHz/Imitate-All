from policies.common.maker import post_init_policies
from task_configs.template import (
    get_task_name,
    replace_task_name,
    set_paths,
    activator,
    TASK_CONFIG_DEFAULT,
)

# def policy_maker(config:dict, stage=None):
#     from policies.act.act import ACTPolicy
#     from policies.common.maker import post_init_policies
#     import logging
#     import torch
#     # TODO: add stage param to the policy class for convenience and simplicity
#     # that is, do the following in the policy class __init__ method.
#     policy = ACTPolicy(config)
#     post_init_policies([policy], stage, [config["ckpt_path"]])
#     return policy
def policy_maker(config:dict, stage=None):
    from policies.act.act import ACTPolicy
    from policies.traditionnal.cnnmlp import CNNMLPPolicy
    # from policies.diffusion.diffusion_policy import DiffusionPolicy
    import logging
    import torch
    policy = ACTPolicy(config)
    post_init_policies([policy], stage, [config["ckpt_path"]])
    policy_1 = CNNMLPPolicy(config)
    post_init_policies([policy], stage, [config["ckpt_path_1"]])
     
    if stage == "train":
        return policy

    elif stage == "eval":
        if TASK_CONFIG_DEFAULT["eval"]["ensemble"] == None:
            return policy
        else:
            ckpt_path = config["ckpt_path"]
            assert ckpt_path is not None, "ckpt_path must exist for loading policy"
            # TODO: all policies should load the ckpt (policy maker should return a class)

            policy.cuda()
            policy.eval()
            def ensemble_policy(*args, **kwargs):
                actions = policy(*args, **kwargs)
                actions_2 = policy_1(*args, **kwargs)
                # average the actions
                actions = (actions + actions_2) / 2
                return actions
            return ensemble_policy
        

def environment_maker(config:dict):
    from envs.make_env import make_environment
    env_config = config["environments"]
    # TODO: use env_config only
    return make_environment(config)

@activator(False)
def augment_images(image):
    from task_configs.config_augmentation.image.basic import color_transforms_1
    return color_transforms_1(image)

# auto replace the task name in the default paths accoring to the file name
TASK_NAME = get_task_name(__file__)
replace_task_name(TASK_NAME, stats_name="dataset_stats.pkl", time_stamp="now")
# but we also show how to set the whole paths manually
# DATA_DIR = f"./data/hdf5/{TASK_NAME}"
# CKPT_DIR = f"./my_ckpt/{TASK_NAME}/ckpt"
# STATS_PATH = f"./my_ckpt/{TASK_NAME}/dataset_stats.pkl"
# set_paths(DATA_DIR, CKPT_DIR, STATS_PATH)  # replace the default data and ckpt paths

chunk_size = 25
joint_num = 7
robot_num = 2
TASK_CONFIG_DEFAULT["common"]["camera_names"] = ["0"]
TASK_CONFIG_DEFAULT["common"]["state_dim"] = joint_num * robot_num
TASK_CONFIG_DEFAULT["common"]["action_dim"] = joint_num * robot_num
TASK_CONFIG_DEFAULT["common"]["policy_config"]["temporal_agg"] = True
TASK_CONFIG_DEFAULT["common"]["policy_config"]["chunk_size"] = chunk_size
TASK_CONFIG_DEFAULT["common"]["policy_config"]["num_queries"] = chunk_size
TASK_CONFIG_DEFAULT["common"]["policy_config"]["kl_weight"] = 10
TASK_CONFIG_DEFAULT["common"]["policy_config"]["policy_maker"] = policy_maker

TASK_CONFIG_DEFAULT["train"]["num_episodes"] = "ALL"
TASK_CONFIG_DEFAULT["train"]["num_epochs"] = 8000
TASK_CONFIG_DEFAULT["train"]["batch_size"] = 24
TASK_CONFIG_DEFAULT["train"]["learning_rate"] = 3e-5
TASK_CONFIG_DEFAULT["train"]["pretrain_ckpt_path"] = ""
TASK_CONFIG_DEFAULT["train"]["pretrain_epoch_base"] = "AUTO"

TASK_CONFIG_DEFAULT["eval"]["robot_num"] = 2
TASK_CONFIG_DEFAULT["eval"]["joint_num"] = joint_num
TASK_CONFIG_DEFAULT["eval"]["start_joint"] = "AUTO"
TASK_CONFIG_DEFAULT["eval"]["max_timesteps"] = 400
TASK_CONFIG_DEFAULT["eval"]["ensemble"] = None
TASK_CONFIG_DEFAULT["eval"]["environments"]["environment_maker"] = environment_maker

# final config
TASK_CONFIG = TASK_CONFIG_DEFAULT

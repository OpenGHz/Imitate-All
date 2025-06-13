from configurations.task_configs.template import (
    TASK_CONFIG_DEFAULT,
    activator,
    get_task_name,
    is_valid_module_name,
    replace_task_name,
    set_paths,
)


def policy_maker(config: dict, stage=None):
    from policies.act.act import ACTPolicy
    from policies.common.maker import post_init_policies

    policy = ACTPolicy(config)
    # TODO: now the ckpt_path are set automatically in train and eval
    post_init_policies([policy], stage, [config["ckpt_path"]])
    return policy


def environment_maker(config: dict):
    from envs.airbot_com_mmk_env import make_env

    return make_env(config)


@activator(False)
def augment_images(image):
    from configurations.task_configs.config_augmentation.image.basic import (
        color_transforms_1,
    )

    return color_transforms_1(image)


# auto replace the task name in the default paths accoring to the file name
TASK_NAME = get_task_name(__file__)
assert is_valid_module_name(TASK_NAME), f"Invalid task name {TASK_NAME}"

replace_task_name(TASK_NAME, stats_name="dataset_stats.pkl", time_stamp="now")
# but we also show how to set the whole paths manually
# DATA_DIR = f"./data/hdf5/{TASK_NAME}"
# CKPT_DIR = f"./my_ckpt/{TASK_NAME}/ckpt"
# STATS_PATH = f"./my_ckpt/{TASK_NAME}/dataset_stats.pkl"
# set_paths(DATA_DIR, CKPT_DIR, STATS_PATH)  # replace the default data and ckpt paths

chunk_size = 25
joint_num = 17
TASK_CONFIG_DEFAULT["common"]["camera_names"] = ["head_camera"]
TASK_CONFIG_DEFAULT["common"]["state_dim"] = joint_num
TASK_CONFIG_DEFAULT["common"]["action_dim"] = joint_num
TASK_CONFIG_DEFAULT["common"]["policy_config"]["temporal_agg"] = False
TASK_CONFIG_DEFAULT["common"]["policy_config"]["chunk_size"] = chunk_size
TASK_CONFIG_DEFAULT["common"]["policy_config"]["num_queries"] = chunk_size
TASK_CONFIG_DEFAULT["common"]["policy_config"]["kl_weight"] = 10
TASK_CONFIG_DEFAULT["common"]["policy_config"]["policy_maker"] = policy_maker

TASK_CONFIG_DEFAULT["train"]["load_data"]["num_episodes"] = "ALL"
TASK_CONFIG_DEFAULT["train"]["load_data"]["batch_size_train"] = 4
TASK_CONFIG_DEFAULT["train"]["load_data"]["batch_size_validate"] = 4
TASK_CONFIG_DEFAULT["train"]["load_data"]["observation_slice"] = (0, 17)
TASK_CONFIG_DEFAULT["train"]["load_data"]["action_slice"] = (0, 17)
TASK_CONFIG_DEFAULT["train"]["load_data"]["mcap_state_topics"] = [
    "/mmk/observation/left_arm/joint_state/position",
    "/mmk/observation/left_arm_eef/joint_state/position",
    "/mmk/observation/right_arm/joint_state/position",
    "/mmk/observation/right_arm_eef/joint_state/position",
    "/mmk/observation/spine/joint_state/position",
    "/mmk/observation/head/joint_state/position",
]
TASK_CONFIG_DEFAULT["train"]["load_data"]["mcap_action_topics"] = [
    "/mmk/action/left_arm/joint_state/position",
    "/mmk/action/left_arm_eef/joint_state/position",
    "/mmk/action/right_arm/joint_state/position",
    "/mmk/action/right_arm_eef/joint_state/position",
    "/mmk/action/spine/joint_state/position",
    "/mmk/action/head/joint_state/position",
]
TASK_CONFIG_DEFAULT["train"]["load_data"]["mcap_camera_topics"] = [
    "/mmk/head_camera/color/video"
]

TASK_CONFIG_DEFAULT["train"]["num_epochs"] = 500
TASK_CONFIG_DEFAULT["train"]["learning_rate"] = 2e-5
TASK_CONFIG_DEFAULT["train"]["pretrain_ckpt_path"] = ""
TASK_CONFIG_DEFAULT["train"]["pretrain_epoch_base"] = "AUTO"

TASK_CONFIG_DEFAULT["eval"]["start_joint"] = "AUTO"
TASK_CONFIG_DEFAULT["eval"]["max_timesteps"] = 300
TASK_CONFIG_DEFAULT["eval"]["ensemble"] = None
TASK_CONFIG_DEFAULT["eval"]["environments"]["environment_maker"] = environment_maker
TASK_CONFIG_DEFAULT["eval"]["ckpt_names"] = ["policy_best.ckpt"]

# final config
TASK_CONFIG = TASK_CONFIG_DEFAULT

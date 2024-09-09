from concurrent.futures import ThreadPoolExecutor
import logging
import pickle
import torch
from collections import Counter
from policies.common.maker import post_init_policies
from task_configs.template import (
    replace_task_name,
    TASK_CONFIG_DEFAULT,
    set_paths,
    activator,
)
from task_configs.config_augmentation.image.basic import color_transforms_2


def policy_maker(config:dict, stage=None):
    from policies.act.act import ACTPolicy
    from policies.traditionnal.cnnmlp import CNNMLPPolicy
    # from policies.diffusion.diffusion_policy import DiffusionPolicy
    with torch.inference_mode():  # 禁用梯度计算
        policies = []
        for key in config:
            if key.startswith("ckpt_path_"):
                policy = ACTPolicy(config)
                post_init_policies([policy], stage, [config[key]])
                policies.append(policy)
                logging.info(len(policies))
        
        if stage == "train":
            return policy

        elif stage == "eval":
            if TASK_CONFIG_DEFAULT["eval"]["ensemble"] == None:
                return policy
            else:
                ckpt_path = config["ckpt_path"]
                stats_path = config["stats_path"]

                assert ckpt_path is not None, "ckpt_path must exist for loading policy"
                # TODO: all policies should load the ckpt (policy maker should return a class)、

                for policy in policies:
                    policy.cuda()
                    policy.eval()

                #串行推理
                # def ensemble_policy(*args, **kwargs):
                #     actions = policy(*args, **kwargs)
                #     actions_2 = policy_1(*args, **kwargs)
                #     # average the actions
                #     actions = (actions + actions_2) / 2
                #     return actions
                
                #并行推理
                def run_policy(policy, *args, **kwargs):
                    with torch.inference_mode():
                        a_hat = policy(*args, **kwargs)
                        #a_hat = a_hat.clone()  # 在更新之前克隆
                        return a_hat
                    

                def ensemble_policy(*args, **kwargs):
                    with ThreadPoolExecutor(max_workers=len(policies)) as executor:
                        futures = []
                        for policy in policies:
                            future = executor.submit(run_policy, policy, *args, **kwargs)
                            futures.append(future)
                        with open(stats_path, "rb") as f:
                            stats = pickle.load(f)

                        post_process = lambda a: a * stats["action_std"][-1] + stats["action_mean"][-1]

                        results = [future.result() for future in futures]
                        # 获取每个结果的最后一位并四舍五入
                        last_elements = [round(post_process(result[0,0,-1].item())) for result in results] 
                        # 统计每个最后一位的出现次数
                        counter = Counter(last_elements)
                        # 找到出现次数最多的stage
                        most_common_last_element, count = counter.most_common(1)[0]

                        state_descriptions = {
                            0: "移动到胶水上方(夹抓由闭到开)",
                            1: "抓住胶水 (机械臂向下运动抓取，夹抓由开到闭)",
                            2: "将胶水移动到蓝色碗里（机械臂向上运动，夹抓一直为闭）",
                            3: "松开夹爪 (机械臂向下放置，夹抓由闭到开)",
                            4: "移动到物块上方（机械臂向上运动，夹抓一直为开）",
                            5: "抓住物块 (机械臂向下抓取，夹抓由开到闭)",
                            6: "将物块放到粉色碗里 （机械臂向上运动，夹抓一直为闭）",
                            7: "松开夹爪  (机械臂向下放置，夹抓由闭到开)",
                            8: "移动到初始位置"
                        }
                        # 获取对应的状态描述
                        state_description = state_descriptions.get(most_common_last_element, "未知状态")
                        print(f"{most_common_last_element}: {state_description}")

                        # 过滤出最后一位是多数的结果
                        filtered_results = [result for result in results if round(post_process(result[0,0,-1].item())) == most_common_last_element]
        
                        actions_sum = torch.zeros_like(filtered_results[0])

                        for result in filtered_results:
                            actions_sum += result

                        actions_avg = actions_sum / len(filtered_results)

                        return actions_avg
                    #return size: (1, chunksize, 7)
                
                # 定义 reset 方法
                def reset():
                    for policy in policies:
                        if hasattr(policy, "reset"):policy.reset()


                # 将 reset 方法绑定到 ensemble_policy 函数上
                ensemble_policy.reset = reset

                return ensemble_policy

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
TASK_CONFIG_DEFAULT["common"]["action_dim"] = joint_num + 1
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
TASK_CONFIG_DEFAULT["eval"]["start_joint"] = [0, -0.766, 0.704, 1.537, -0.965, -1.576, 0.710] 
TASK_CONFIG_DEFAULT["eval"]["max_timesteps"] = 400
TASK_CONFIG_DEFAULT["eval"]["ensemble"] = True
TASK_CONFIG_DEFAULT["eval"]["environments"]["environment_maker"] = environment_maker

# final config
TASK_CONFIG = TASK_CONFIG_DEFAULT

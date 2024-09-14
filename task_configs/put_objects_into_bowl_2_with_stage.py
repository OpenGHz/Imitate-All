from concurrent.futures import ThreadPoolExecutor
import logging
import pickle
from PIL import Image  
import numpy as np
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

import os
import base64
import openai
from io import BytesIO
from openai import OpenAI

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
                    
                def check_stage_error(all_time_stage):
                    """
                    检查 all_time_stage 中每个模型的 stage, 返回list of bool,表示每个模型是否有问题,True 表示有问题,False 表示正常。
                    如果当前的 stage 小于上一次的 stage, 则认为该模型有问题。
                    
                    """
                    valid_transitions = {
                        0: [0,1],
                        1: [0,1,2],
                        2: [1,2,3],
                        3: [0,1,3,4],
                        4: [4,5],
                        5: [4,5,6],
                        6: [5,6,7],
                        7: [4,5,7,8],
                        8: [0,1,2,3,4,5,6,7,8]
                    }
                        # 定义每个 stage 的最大允许次数
                    stage_limits = {
                        0: 50,
                        1: 50,
                        2: 50,
                        3: 50,
                        4: 50,
                        5: 50,
                        6: 50,
                        7: 50,
                        8: 50
                    }
                    error_signals = []

                    for model_stages in all_time_stage:
                        
                        has_error = False
                        if len(model_stages) > 2:
                            curr_stage = model_stages[-1]
                            prev_stage = model_stages[-2]
                            if curr_stage not in valid_transitions.get(prev_stage, []):
                                has_error = True
                            else:
                                same_stage_count = 0
                                for i in range(len(model_stages)-1, -1, -1):
                                    if model_stages[i] == curr_stage:
                                        same_stage_count += 1
                                    else:
                                        break

                                if same_stage_count > stage_limits[curr_stage]:
                                    has_error = True

                        error_signals.append(has_error)

                    return error_signals
                    
                def ensemble_policy(*args, **kwargs):
                    if not hasattr(ensemble_policy, 'all_time_stage'):
                        ensemble_policy.all_time_stage = [[] for _ in range(len(policies))]
                    if not hasattr(ensemble_policy, 'all_time_action'):
                        ensemble_policy.all_time_action = []
                    if not hasattr(ensemble_policy, 'skip_gpt_analysis'):
                        ensemble_policy.skip_gpt_analysis = False

                    stage_descriptions = {
                        0: "移动到胶水上方",
                        1: "抓胶水",
                        2: "将胶水移动到蓝色碗上方",
                        3: "放置胶水",
                        4: "移动到物块上方",
                        5: "抓物块",
                        6: "将物块移动到粉色碗上方",
                        7: "放置物块",
                        8: "移动到初始位置"
                    }
                    with ThreadPoolExecutor(max_workers=len(policies)) as executor:
                        futures = []
                        for policy in policies:
                            future = executor.submit(run_policy, policy, *args, **kwargs)
                            futures.append(future)

                        with open(stats_path, "rb") as f:
                            stats = pickle.load(f)

                        post_process = lambda a: a * stats["action_std"][-1] + stats["action_mean"][-1]

                        results = [future.result() for future in futures]
                        last_elements = [round(post_process(result[0,0,-1].item())) for result in results] # 获取每个模型的stage                        
                        for i, last_element in enumerate(last_elements):
                            ensemble_policy.all_time_stage[i].append(last_element)
                        # size of all_time_stage ： [
                        #         [stage_1_time_1, stage_1_time_2, ..., stage_1_time_t],  # 模型1的stage输出
                        #         [stage_2_time_1, stage_2_time_2, ..., stage_2_time_t],  # 模型2的stage输出
                        #         ...
                        #         [stage_n_time_1, stage_n_time_2, ..., stage_n_time_t]   # 模型n的stage输出
                        #     ]

                        error_signals = check_stage_error(ensemble_policy.all_time_stage)

                        # 计算异常模型的比例
                        num_policies = len(policies)
                        num_errors = sum(error_signals)
                        error_ratio = num_errors / num_policies
                        print(f"error ratio: {error_ratio}")

                        valid_results = [result for i, result in enumerate(results) if not error_signals[i]]
                        counter = Counter([round(post_process(result[0,0,-1].item())) for result in valid_results])  # 统计每个最后一位的出现次数
                        most_common_last_element, count = counter.most_common(1)[0]  # 找到出现次数最多的 stage
                        curr_stage = most_common_last_element
                        if error_ratio <= 0.5:
                            if error_ratio >= 0.3:print("Warning: 0.3 < error_ratio <= 0.5")

                            stage_description = stage_descriptions.get(curr_stage, "unknow stage")
                            print(f"{curr_stage}: {stage_description}")

                            # 过滤出最后一位是多数的结果
                            filtered_results = [result for result in results if round(post_process(result[0,0,-1].item())) == most_common_last_element]
            
                            actions_sum = torch.zeros_like(filtered_results[0])
                            #actions_sum = torch.zeros((1, 50, 8))
                            for result in filtered_results:
                                actions_sum += result

                            actions_avg = actions_sum / len(filtered_results)

                            ensemble_policy.all_time_action.append(actions_avg[0,0,0:7])

                            return actions_avg  #return size: (1, chunksize, 8)                        
                        
                        else :
                            # 异常模型超过50%，GPT analysis
                            print("error_ratio > 50%, GPT analysis is called")
                            if ensemble_policy.skip_gpt_analysis:
                                # 直接返回最后一个模型的输出
                                result = results[-1]
                                handled_action = torch.zeros_like(result)
                                handled_action[0, 0, 0:7] = result[0, 0, 0:7]
                                ensemble_policy.all_time_action.append(handled_action[0, 0, 0:7])
                                return handled_action
                            image_tensor = args[1] 
                            output = handle_exceptions_with_gpt(image_tensor)
                            handled_action = torch.zeros_like(results[0])
                            if output == 0:
                                # 状态0：没抓住物体，重现到上一个状态
                                last_stage = curr_stage - 1
                                print(f"now turn to last stage: {last_stage}", stage_descriptions[last_stage])
                                for i, stages in enumerate(ensemble_policy.all_time_stage):
                                    index = stages.index(last_stage)
                                    corresponding_action = ensemble_policy.all_time_action[index]
                                    handled_action[0, 0, 0:7] = corresponding_action
                                    ensemble_policy.all_time_stage.append(handled_action[0, 0, 0:7])
                                    break

                                return handled_action
                            elif output == 1:
                                # 状态1：图像中杂物太多，需要换到另一个带杂物的ACT
                                # 假设最后一个模型是杂物模型
                                result = results[-1]
                                handled_action[0, 0, 0:7] = result[0, 0, 0:7]
                                ensemble_policy.all_time_action.append(handled_action[0, 0, 0:7])
                                # 设置标志，之后不再进入GPT分析
                                ensemble_policy.skip_gpt_analysis = True
                                return handled_action
                            else:
                                # 状态2：物体错误/不存在
                                # 需要人为干预
                                input("Please adjust the object in the scene and press Enter to continue.")
                                print(f"now turn to stage:{curr_stage}")
                                filtered_results = [result for result in results if round(post_process(result[0,0,-1].item())) == most_common_last_element]
                
                                actions_sum = torch.zeros_like(filtered_results[0])
                                #actions_sum = torch.zeros((1, 50, 8))
                                for result in filtered_results:
                                    actions_sum += result
                                actions_avg = actions_sum / len(filtered_results)
                                ensemble_policy.all_time_action.append(actions_avg[0,0,0:7])
                                return actions_avg 


                def handle_exceptions_with_gpt(image_tensor):
                    if image_tensor.numel() == 0:
                        raise ValueError("No image provided in args")
                    #print(image_tensor)
                    # 调用外部函数获取GPT响应数据
                    response_data = get_gpt_response(image_tensor)
                    print(f"THE GPT RESPOND: {response_data}")
                    # 假设GPT返回的状态在response_data中
                    gpt_status = response_data

                    if "0" in gpt_status:
                        return 0
                    elif "1" in gpt_status:
                        return 1
                    elif "2" in gpt_status:
                        return 2
                    else:
                        return 3


                def get_gpt_response(image_tensor):
                    # Set up proxy if needed
                    os.environ["http_proxy"] = "http://localhost:7890"
                    os.environ["https_proxy"] = "http://localhost:7890"

                    def tensor_to_base64_image(tensor):
                        # 检查张量的形状
                        if tensor.ndimension() == 5:
                            # 假设张量的形状是 (1, 1, C, H, W)
                            tensor = tensor.squeeze(0).squeeze(0)  # 去掉前两个维度
                        elif tensor.ndimension() == 4:
                            # 假设张量的形状是 (1, C, H, W)
                            tensor = tensor.squeeze(0)  # 去掉第一个维度
                        elif tensor.ndimension() == 3:
                            # 假设张量的形状是 (C, H, W)
                            pass
                        else:
                            raise ValueError("Unsupported tensor shape: {}".format(tensor.shape))
                        
                        # 转置张量的轴顺序
                        tensor = tensor.cpu().numpy()
                        tensor = np.transpose(tensor, (1, 2, 0))  # 转置为 (H, W, C)
                        
                        # 将张量转换为图像
                        tensor = (tensor * 255).astype(np.uint8)
                        image = Image.fromarray(tensor)
                        
                        # 将图像编码为 base64
                        buffer = BytesIO()
                        image.save(buffer, format="JPEG")
                        base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
                        return base64_image

                    # Getting the base64 string
                    base64_image = tensor_to_base64_image(image_tensor)
                    client = openai.OpenAI(api_key="")

                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": (
                                        "Our task is to use a robotic arm to place a yellow glue stick into a blue bowl "
                                        "and a wooden block into a pink bowl. Based on the image, please tell me the current situation. "
                                        "Return a number: If the gripper is not gripping the object, return 0; "
                                        "if there are many other distracting items in the scene, return 1; "
                                        "if there is no yellow glue stick or no wooden block, return 2; "

                                    )},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{base64_image}",
                                        },
                                    },
                                ],
                            }
                        ],
                        max_tokens=300,
                    )

                    #print(response.choices[0])
                    return response.choices[0].message.content
                
                # 定义 reset 方法
                def reset():
                    for policy in policies:
                        if hasattr(policy, "reset"):policy.reset()
                    # 重置 all_time_stage 和 all_time_action
                    if hasattr(ensemble_policy, 'all_time_stage'):
                        ensemble_policy.all_time_stage = [[] for _ in range(len(policies))]
                    if hasattr(ensemble_policy, 'all_time_action'):
                        ensemble_policy.all_time_action = []

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

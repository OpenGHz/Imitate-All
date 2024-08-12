import torch
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
import shutil
import time
import argparse

from utils import load_data, compute_dict_mean, set_seed, detach_dict, GPUer, pretty_print_dict
from task_configs.config_tools.basic_configer import basic_parser, get_all_config
from policies.common.maker import make_policy


def main(args:dict):
    # get all config
    all_config = get_all_config(args, 'train')
    ckpt_dir = all_config['ckpt_dir']
    stats_path = all_config['stats_path']
    dataset_dir = all_config['dataset_dir']
    image_augmentor = all_config['image_augmentor']
    if image_augmentor.activated:
        print("Use image augmentor")
    # 加载数据及统计信息
    train_dataloader, val_dataloader, stats = load_data(dataset_dir, all_config["num_episodes"], all_config['camera_names'], all_config['batch_size'], all_config['batch_size'], {'image':image_augmentor}, all_config)
    # 创建保存路径
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    stats_dir = os.path.dirname(stats_path)
    if not os.path.isdir(stats_dir):
        os.makedirs(stats_dir)
    # 复制配置文件到stats_dir
    shutil.copy(all_config["config_file_sys_path"], stats_dir)
    # 保存统计数据
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl') if stats_path == '' else stats_path
    print(f'Saving stats into {stats_path}...')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)
    # 保存关键信息（must pop functions)
    key_info_path = os.path.join(stats_dir, f'key_info.pkl')
    print(f'Saving key info into {key_info_path}...')
    all_config_cp = deepcopy(all_config)
    all_config_cp["policy_config"].pop('policy_maker')
    all_config_cp.pop('image_augmentor')
    key_info = {
        "init_info": {"init_joint": all_config_cp["start_joint"], "init_action": all_config_cp["start_action"]},
        "all_config": all_config_cp,
    }
    # pretty_print_dict(key_info)
    with open(key_info_path, 'wb') as f:
        pickle.dump(key_info, f)
    # wait for free GPU
    target_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    while True:
        free_gpus, gpus_num = GPUer.check_all_gpus_idle(5)
        if len(free_gpus) > 0:
            if target_gpu is not None:
                target_gpu = int(target_gpu)
                if target_gpu >= gpus_num:
                    raise ValueError(f'Target GPU id (from 0) {target_gpu} is not valid, only {gpus_num} gpus available')
                if target_gpu not in free_gpus:
                    print(f'Target GPU {target_gpu} is not free, waiting...')
                    time.sleep(60)
                    continue
                else:
                    free_gpu = target_gpu
            else:
                free_gpu = free_gpus[0]
            os.environ["CUDA_VISIBLE_DEVICES"] = str(free_gpu)
            print(f'Using GPU {free_gpu}')
            break
        else:
            print('No free GPU, waiting...')
            time.sleep(60)
    # train policy
    set_seed(all_config['seed'])
    best_ckpt_info = train_bc(train_dataloader, val_dataloader, all_config)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    print(f'Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}')

def make_optimizer(policy):
    if hasattr(policy, 'configure_optimizers'):
        optimizer = policy.configure_optimizers()
    else:
        print('Warning: Using default optimizer')
    return optimizer

def forward_pass(data:torch.Tensor, policy):
    image_data, qpos_data, action_data, is_pad = data
    image_data, qpos_data, action_data, is_pad = image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda()
    return policy(qpos_data, image_data, action_data, is_pad)  # TODO remove None

def train_bc(train_dataloader, val_dataloader, config):
    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_config = config['policy_config']
    stats_dir = os.path.dirname(config['stats_path'])
    eval_every = 3.14 if config["eval_every"] == 0 else config["eval_every"]
    parallel = config["parallel"]
    if config["eval_every"] != 0:
        from policy_evaluate import eval_bc

    policy = make_policy(policy_config)
    # load pretrain model if needed
    pretrain_path = config["pretrain_ckpt_path"]
    if pretrain_path != "":  # load pretrained model
        if pretrain_path in ["best", "last"]:  # 默认使用ckpt路径下的policy_best.ckpt继续训练
            from utils import replace_timestamp
            # TODO: 目前仅支持指定完整路径
            pretrain_dir = replace_timestamp(ckpt_dir, config['time_stamp'])
            loading_status = policy.deserialize(torch.load(os.path.join(pretrain_dir, f'policy_{pretrain_path}.ckpt')))
            print(f'loaded! {loading_status}')
        elif pretrain_path != "":  # 使用指定完整路径的ckpt继续训练
            loading_status = policy.deserialize(torch.load(pretrain_path))
            print(f'Resume policy from: {pretrain_path}, Status: {loading_status}')
        epoch_base = config['pretrain_epoch_base']
        if epoch_base == 'AUTO':
            if pretrain_path in ["best", "last"] or "best" in pretrain_path or "last" in pretrain_path:
                epoch_base = 0
            else:
                try:
                    epoch_base = int(pretrain_path.split('_')[-3])
                except:
                    try:
                        epoch_base = int(pretrain_path.split('_')[-2])
                    except:
                        raise ValueError(f'Invalid pretrain_ckpt_path to auto get epoch bias: {pretrain_path}')
    else:
        epoch_base = 0
    # set GPU device
    policy.cuda()
    if parallel is not None:
        if parallel["mode"] == "DP":
            policy = torch.nn.DataParallel(policy, device_ids=parallel["device_ids"])
        elif parallel["mode"] == "DDP":
            # TODO: can not use DDP for now
            raise NotImplementedError
            policy = torch.nn.parallel.DistributedDataParallel(policy, device_ids=parallel["device_ids"])
        else:
            raise ValueError(f'Invalid parallel mode: {parallel["mode"]}')
    # make optimizer
    optimizer = make_optimizer(policy)

    # training loop
    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    for epoch in tqdm(range(epoch_base, num_epochs + epoch_base)):
        print(f'\nEpoch {epoch}')
        step = epoch - epoch_base + 1
        # validation
        if step % config["validate_every"] == 0:
            print('validating')
            with torch.inference_mode():
                policy.eval()
                epoch_dicts = []
                for batch_idx, data in enumerate(val_dataloader):
                    forward_dict = forward_pass(data, policy)
                    epoch_dicts.append(forward_dict)
                epoch_summary = compute_dict_mean(epoch_dicts)
                validation_history.append(epoch_summary)

                epoch_val_loss = epoch_summary['loss']
                if epoch_val_loss < min_val_loss:
                    min_val_loss = epoch_val_loss
                    best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
            print(f'Val loss:   {epoch_val_loss:.5f}')
            summary_string = ''
            for k, v in epoch_summary.items():
                summary_string += f'{k}: {v.item():.3f} '
            print(summary_string)

        # evaluation #TODO: 目前不支持训练过程中的评估
        if step % eval_every == 0:
            # first save then eval
            ckpt_name = f'policy_step_{step}_seed_{seed}.ckpt'
            ckpt_path = os.path.join(ckpt_dir, ckpt_name)
            torch.save(policy.serialize(), ckpt_path)
            success, _ = eval_bc(config, ckpt_name, save_episode=True, num_rollouts=10)

        # training
        policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy)
            # backward
            loss = forward_dict['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
        epoch_summary = compute_dict_mean(train_history[(batch_idx+1)*(epoch-epoch_base):(batch_idx+1)*(epoch-epoch_base+1)])
        epoch_train_loss = epoch_summary['loss']
        print(f'Train loss: {epoch_train_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        if step % config["save_every"] == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
            torch.save(policy.serialize(), ckpt_path)
            plot_history(train_history, validation_history, epoch, stats_dir, seed)

    # training finished
    # save last and best ckpts
    ckpt_path = os.path.join(stats_dir, f'policy_last.ckpt')
    torch.save(policy.serialize(), ckpt_path)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    policy_best_path = os.path.join(stats_dir, f'policy_best.ckpt')
    torch.save(best_state_dict, policy_best_path)

    # save training curves
    plot_history(train_history, validation_history, num_epochs, stats_dir, seed)

    # show and save result description
    description = f'Best ckpt: val loss {min_val_loss:.6f} @ epoch{best_epoch} with seed {seed}'
    print("Training finished.")
    print(description)
    with open(os.path.join(stats_dir, 'description.txt'), 'w') as f:
        f.write(description)
    return best_ckpt_info

def plot_history(train_history, validation_history, num_epochs, stats_dir, seed):
    """save training curves to stats_dir"""
    for key in train_history[0]:
        plot_path = os.path.join(stats_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(np.linspace(0, num_epochs-1, len(train_history)), train_values, label='train')
        plt.plot(np.linspace(0, num_epochs-1, len(validation_history)), val_values, label='validation')
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    print(f'Saved plots to {stats_dir}')

def parser_add_train(parser:argparse.ArgumentParser):
    parser.add_argument('-bs', '--batch_size', action='store', type=int, help='batch_size', required=False)
    parser.add_argument('-lr', '--learning_rate', action='store', type=float, help='learning_rate', required=False)
    parser.add_argument('-ne', '--num_epochs', action='store', type=int, help='num_epochs', required=False)
    parser.add_argument('-pcp','--pretrain_ckpt_path', action='store', type=str, help='pretrain_ckpt_path', required=False)
    parser.add_argument('-peb','--pretrain_epoch_base', action='store', type=str, help='pretrain_epoch_base', required=False)
    parser.add_argument('-ee', '--eval_every', action='store', type=int, help='eval_every')
    parser.add_argument('-ve', '--validate_every', action='store', type=int, help='validate_every', required=False)
    parser.add_argument('-se', '--save_every', action='store', type=int, help='save_every', required=False)
    parser.add_argument('-smd','--skip_mirrored_data', action='store', type=bool, help='skip_mirrored_data', required=False)
    # set time_stamp  # TODO: used to load pretrain model
    parser.add_argument("-ts", "--time_stamp", action="store", type=str, help="time_stamp", required=False)
    # parser.add_argument('--actuator_network_dir', action='store', type=str, help='actuator_network_dir', required=False)
    # parser.add_argument('--history_len', action='store', type=int)
    # parser.add_argument('--future_len', action='store', type=int)
    # parser.add_argument('--prediction_len', action='store', type=int)
    return parser


if __name__ == '__main__':
    """
    参数优先级：命令行 > config文件
    """
    parser = basic_parser()
    parser_add_train(parser)
    main(vars(parser.parse_args()))

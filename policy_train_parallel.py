import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.amp import GradScaler, autocast
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
import shutil
import argparse
import logging

from utils.utils import load_data, LoadDataConfig, compute_dict_mean, set_seed, detach_dict
from configurations.task_configs.config_tools.basic_configer import basic_parser, get_all_config
from policies.common.maker import make_policy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(args: dict):
    all_config = get_all_config(args, 'train')
    ckpt_dir = all_config['ckpt_dir']
    stats_path = all_config['stats_path']
    os.makedirs(ckpt_dir, exist_ok=True)
    stats_dir = os.path.dirname(stats_path)
    os.makedirs(stats_dir, exist_ok=True)
    shutil.copy(all_config["config_file_sys_path"], stats_dir)

    _, _, stats = load_data(
        LoadDataConfig(
            **all_config["load_data"],
            camera_names=all_config["camera_names"],
            chunk_sizes={"action": all_config["policy_config"]["chunk_size"]},
        )
    )
    stats_path_final = os.path.join(
        ckpt_dir, 'dataset_stats.pkl') if stats_path == '' else stats_path
    tqdm.write(f'Saving stats into {stats_path_final}...')
    with open(stats_path_final, 'wb') as f:
        pickle.dump(stats, f)

    key_info_path = os.path.join(stats_dir, 'key_info.pkl')
    tqdm.write(f'Saving key info into {key_info_path}...')
    all_config_cp = deepcopy(all_config)
    all_config_cp["policy_config"].pop('policy_maker', None)
    all_config_cp["environments"].pop('environment_maker', None)
    all_config_cp["load_data"].pop("augmentors", None)
    key_info = {
        "init_info": {"init_joint": all_config_cp["start_joint"], "init_action": all_config_cp["start_action"]},
        "all_config": all_config_cp,
    }
    with open(key_info_path, 'wb') as f:
        pickle.dump(key_info, f)

    num_gpus = torch.cuda.device_count()
    if num_gpus < 1:
        raise RuntimeError("No GPUs available for training.")
    all_config["target_gpus"] = [0, 1]
    num_gpus = len(all_config["target_gpus"])

    if num_gpus > 1:
        mp.spawn(main_worker, nprocs=num_gpus, args=(num_gpus, all_config))
    else:
        main_worker(0, 1, all_config)


def main_worker(rank: int, world_size: int, config: dict):
    dist.init_process_group(
        backend="nccl", init_method="tcp://127.0.0.1:29500", world_size=world_size, rank=rank)

    gpu_id = config["target_gpus"][rank]
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    set_seed(config['seed'])

    train_dataloader, val_dataloader, _ = load_data(
        LoadDataConfig(
            **config["load_data"],
            camera_names=config["camera_names"],
            chunk_sizes={"action": config["policy_config"]["chunk_size"]},
        )
    )
    train_sampler = DistributedSampler(
        train_dataloader.dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(
        val_dataloader.dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataloader.dataset,
        batch_size=train_dataloader.batch_size,
        sampler=train_sampler,
        collate_fn=train_dataloader.collate_fn,
        num_workers=train_dataloader.num_workers,
        pin_memory=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataloader.dataset,
        batch_size=val_dataloader.batch_size,
        sampler=val_sampler,
        collate_fn=val_dataloader.collate_fn,
        num_workers=val_dataloader.num_workers,
        pin_memory=True
    )

    best_ckpt_info = train_bc(
        train_dataloader, val_dataloader, config, rank, device)
    if rank == 0 and best_ckpt_info is not None:
        tqdm.write(
            f'Best ckpt: val loss {best_ckpt_info[1]:.6f} @ epoch {best_ckpt_info[0]}')
    dist.destroy_process_group()


def make_optimizer(policy: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
    if hasattr(policy, 'configure_optimizers'):
        optimizer = policy.configure_optimizers()
    else:
        tqdm.write('Warning: Using default optimizer (Adam)')
        optimizer = torch.optim.Adam(
            policy.parameters(), lr=config['learning_rate'])
    return optimizer


def forward_pass(data: tuple, policy: torch.nn.Module, device: torch.device) -> dict:
    image_data, qpos_data, action_data, is_pad = data
    image_data = image_data.to(device, non_blocking=True)
    qpos_data = qpos_data.to(device, non_blocking=True)
    action_data = action_data.to(device, non_blocking=True)
    is_pad = is_pad.to(device, non_blocking=True)
    return policy(qpos_data, image_data, action_data, is_pad)


def get_epoch_base(pretrain_path: str, epoch_base: str) -> int:
    if pretrain_path == "":
        return 0
    elif epoch_base == 'AUTO':
        try:
            return int(pretrain_path.split('_')[-3])
        except:
            try:
                return int(pretrain_path.split('_')[-2])
            except:
                raise ValueError(
                    f'Invalid pretrain_ckpt_path: {pretrain_path}')
    return int(epoch_base)


def train_bc(train_dataloader, val_dataloader, config, rank, device):
    num_epochs = config["num_epochs"]
    ckpt_dir = config["ckpt_dir"]
    seed = config["seed"]
    stats_dir = os.path.dirname(config["stats_path"])
    validate_every = config["validate_every"]
    save_every = config["save_every"]
    use_amp = True
    if use_amp:
        print("Using amp training mode!")
    else:
        print("Not using amp training mode!")

    pretrain_path = config.get("pretrain_ckpt_path", "")
    policy_config = config["policy_config"]
    policy_config["ckpt_path"] = pretrain_path
    policy = make_policy(policy_config, "train")
    policy = policy.to(device)
    policy = torch.nn.parallel.DistributedDataParallel(
        policy, device_ids=[device.index], find_unused_parameters=True)
    optimizer = make_optimizer(policy, config)

    # 记录每个epoch的训练loss及其它指标
    train_history = []
    # 验证历史保存为字典，键为epoch
    validation_history = {}
    min_val_loss = np.inf
    best_ckpt_info = None

    scaler = GradScaler()
    epoch_base = get_epoch_base(
        pretrain_path, config.get("pretrain_epoch_base", 0))
    num_stages = (num_epochs + save_every - 1) // save_every

    for stage_idx in range(num_stages):
        start_epoch = epoch_base + stage_idx * save_every
        end_epoch = min(start_epoch + save_every, epoch_base + num_epochs)
        if rank == 0:
            tqdm.write(
                f'\n[Stage {stage_idx + 1} / {num_stages}] Epochs {start_epoch} to {end_epoch - 1}')
            # 一个stage内用同一个进度条
            stage_pbar = tqdm(total=(end_epoch - start_epoch),
                              desc=f"Stage {stage_idx + 1}", leave=False)
        for epoch in range(start_epoch, end_epoch):
            policy.train()
            epoch_train_dicts = []
            for batch in train_dataloader:
                optimizer.zero_grad()
                if use_amp:
                    with autocast(device_type="cuda"):
                        forward_dict = forward_pass(batch, policy, device)
                        loss = forward_dict['loss']
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    forward_dict = forward_pass(batch, policy, device)
                    loss = forward_dict['loss']
                    loss.backward()
                    optimizer.step()
                epoch_train_dicts.append(detach_dict(forward_dict))
            if rank == 0:
                epoch_summary = compute_dict_mean(epoch_train_dicts)
                train_history.append((epoch, epoch_summary))
                tqdm.write(f"Epoch {epoch} Train: " +
                           " ".join(f"{k}: {v.item():.3f}" for k, v in epoch_summary.items()))
                stage_pbar.update(1)
            # 如果满足验证条件，则执行验证
            if (epoch + 1) % validate_every == 0:
                policy.eval()
                val_epoch_dicts = []
                if rank == 0:
                    val_pbar = tqdm(total=len(val_dataloader),
                                    desc=f"Validation Epoch {epoch}", leave=False)
                for batch in val_dataloader:
                    with torch.inference_mode():
                        if use_amp:
                            with autocast(device_type="cuda"):
                                forward_dict = forward_pass(
                                    batch, policy, device)
                        else:
                            forward_dict = forward_pass(batch, policy, device)
                    val_epoch_dicts.append(forward_dict)
                    if rank == 0:
                        val_pbar.update(1)
                if rank == 0:
                    val_pbar.close()
                    val_summary = compute_dict_mean(val_epoch_dicts)
                    validation_history[epoch] = val_summary
                    val_loss = val_summary['loss']
                    tqdm.write(
                        f"Epoch {epoch} Validation loss: {val_loss:.5f}")
                    tqdm.write("Validation: " +
                               " ".join(f"{k}: {v.item():.3f}" for k, v in val_summary.items()))
                    if val_loss < min_val_loss:
                        min_val_loss = val_loss
                        best_ckpt_info = (epoch, min_val_loss, deepcopy(
                            policy.module.state_dict()))
                policy.train()
        if rank == 0:
            stage_pbar.close()
            # 每个stage结束后保存一次检查点、并画一次图（绘制至当前epoch）
            ckpt_path = os.path.join(
                ckpt_dir, f'policy_epoch_{end_epoch - 1}_seed_{seed}.ckpt')
            torch.save(policy.module.state_dict(), ckpt_path)
            plot_history(train_history, validation_history,
                         end_epoch - epoch_base, stats_dir, seed)

    if rank == 0:
        torch.save(policy.module.state_dict(),
                   os.path.join(stats_dir, 'policy_last.ckpt'))
        if best_ckpt_info:
            best_epoch, best_val_loss, best_state_dict = best_ckpt_info
            torch.save(best_state_dict, os.path.join(
                stats_dir, 'policy_best.ckpt'))
            description = f'Best ckpt: val loss {best_val_loss:.6f} @ epoch {best_epoch} with seed {seed}'
            tqdm.write(description)
            with open(os.path.join(stats_dir, 'description.txt'), 'w') as f:
                f.write(description)
        # 最后绘制全周期图
        plot_history(train_history, validation_history, stats_dir, seed)

    return best_ckpt_info


def plot_history(train_history, validation_history, stats_dir, seed):
    """
    参数说明：
      - train_history：list，每个元素为 (epoch, summary)，summary 是包含 loss 等指标的字典
      - validation_history：dict，键为 epoch，值为验证时的 summary
      - num_epochs：当前训练的总 epoch 数
    """
    if not train_history:
        return
    # 以第一个 epoch 的 summary 键来绘制所有指标曲线
    keys = list(train_history[0][1].keys())
    for key in keys:
        plot_path = os.path.join(stats_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        # 训练数据：横坐标取实际epoch数
        train_epochs = [epoch for (epoch, summary) in train_history]
        train_values = [summary[key].item() for (_, summary) in train_history]
        plt.plot(train_epochs, train_values, label='train')
        # 验证数据（可能不是每个epoch都有记录，因此仅取有记录的部分）
        val_epochs = sorted(validation_history.keys())
        val_values = [validation_history[e][key].item()
                      for e in val_epochs if key in validation_history[e]]
        if val_epochs:
            plt.plot(val_epochs, val_values, label='validation')
        plt.xlabel('Epoch')
        plt.ylabel(key)
        plt.title(key)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
    tqdm.write(f'Saved plots to {stats_dir}')


def parser_train(parser: argparse.ArgumentParser = None) -> argparse.ArgumentParser:
    if parser is None:
        parser = basic_parser()
    parser.add_argument('-bs', '--batch_size', type=int,
                        help='batch_size', required=False)
    parser.add_argument('-lr', '--learning_rate', type=float,
                        help='learning_rate', required=False)
    parser.add_argument('-ne', '--num_epochs', type=int,
                        help='num_epochs', required=False)
    parser.add_argument('-pcp', '--pretrain_ckpt_path', type=str,
                        help='pretrain_ckpt_path', required=False, default="")
    parser.add_argument('-peb', '--pretrain_epoch_base', type=str,
                        help='pretrain_epoch_base', required=False, default="0")
    parser.add_argument('-ve', '--validate_every', type=int,
                        help='validate_every', required=False, default=100)
    parser.add_argument('-se', '--save_every', type=int,
                        help='save_every', required=False, default=1000)
    parser.add_argument('-smd', '--skip_mirrored_data',
                        type=bool, help='skip_mirrored_data', required=False)
    parser.add_argument('-gth', '--gpu_threshold', type=float,
                        help='gpu_threshold', required=False)
    parser.add_argument("-ts", "--time_stamp", type=str,
                        help="time_stamp", required=False)
    parser.add_argument("-amp", "--use_amp", type=bool, default=True,
                        help="use_amp", required=False)
    return parser


if __name__ == '__main__':
    parser = basic_parser()
    parser_train(parser)
    args = parser.parse_args()
    main(vars(args))

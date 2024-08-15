import logging, os
import torch
from typing import List


def make_policy(
    config, stage=None
):  # TODO: remove this function and use the config file
    policy_maker = config["policy_maker"]
    policy = policy_maker(config, stage)
    assert policy is not None, "Please use the make_policy function in the config file"
    return policy


def post_init_policies(policies: List[torch.nn.Module], stage, ckpt_paths) -> None:
    """Load the checkpoint for the policies and move them to the GPU
    Args:
        policies (List[torch.nn.Module]): List of policies
        stage (str): "train" or "eval"
        ckpt_paths (List[str]): List of checkpoint paths
    """
    # https://pytorch.org/docs/stable/generated/torch.load.html
    weights_only = False
    for policy, ckpt_path in zip(policies, ckpt_paths):
        if ckpt_path not in [None, ""]:
            if not os.path.exists(ckpt_path):
                raise Exception(f"Checkpoint path does not exist: {ckpt_path}")
            if stage == "train":
                loading_status = policy.load_state_dict(torch.load(ckpt_path, weights_only=weights_only))
                logging.info(f'Resume policy from: {ckpt_path}, Status: {loading_status}')
            elif stage == "eval":
                loading_status = policy.load_state_dict(torch.load(ckpt_path, weights_only=weights_only))
                logging.info(loading_status)
                logging.info(f"Loaded: {ckpt_path}")
        policy.cuda()

        if stage == "eval":
            policy.eval()

def save_model(policy, path):
    torch.save(policy.state_dict(), path)
    logging.info(f"Saved policy to: {path}")
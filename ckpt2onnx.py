"""Export model to ONNX format.

ckpt file from: `act/myckpt/stack_cups/20240623-004841/policy_best.ckpt`

The .onnx file is exported to `act/onnx_output/act_policy.onnx`.

Please run this script with `python ckpt2onnx.py` in the root directory of the repository.
"""

import torch

from task_configs.example_task import TASK_CONFIG
from task_configs.example_task import policy_maker
from pathlib import Path


if __name__ == "__main__":

    # some configurations are missing or misplaced in the TASK_CONFIG dictionary
    TASK_CONFIG["common"]["policy_config"]["state_dim"] = 7
    TASK_CONFIG["common"]["policy_config"]["camera_names"] = ["0"]
    TASK_CONFIG["common"]["policy_config"]["enc_layers"] = 4
    TASK_CONFIG["common"]["policy_config"]["dec_layers"] = 7
    TASK_CONFIG["common"]["policy_config"]["nheads"] = 8

    print(TASK_CONFIG["common"]["policy_config"])
    """
    >>> {'policy_class': 'ACT',
        'policy_maker': <function task_configs.example_task.policy_maker(config: dict)>,
        'kl_weight': 10,
        'chunk_size': 25,
        'hidden_dim': 512,
        'dim_feedforward': 3200,
        'temporal_agg': False,
        'num_queries': 25,
        'state_dim': 7,
        'camera_names': ['0'],
        'enc_layers': 4,
        'dec_layers': 7,
        'nheads': 8}
    """

    # load policy
    act_policy = policy_maker(TASK_CONFIG["common"]["policy_config"])
    ckpt_root = Path("my_ckpt/stack_cups/20240623-004841")
    ckpt_path = ckpt_root / "policy_best.ckpt"

    loading_status = act_policy.to("cuda").load_state_dict(torch.load(ckpt_path))
    print(loading_status)
    qpos = torch.randn((1, 7), device="cuda")
    img = torch.randn([1, 3, 480, 640], device="cuda")
    # img = torch.randn([1, 1, 3, 480, 640], device="cuda")

    # export to onnx
    onnx_root = Path("./onnx_output")
    onnx_root.mkdir(parents=True, exist_ok=True)
    onnx_path = onnx_root / "act_policy.onnx"

    torch.onnx.export(act_policy, (qpos, img), onnx_path, opset_version=11)
    print("\nSuccessfullt exported to: ", onnx_path)

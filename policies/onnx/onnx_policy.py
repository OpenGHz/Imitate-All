import onnx
import onnxruntime as ort
import logging
import torch
import numpy as np
from policies.common.wrapper import TemporalEnsembling


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class ONNXPolicy(object):
    # define the observation space and action space
    # these are used to check the input and output data interaction
    # between the policy and the environment
    observation_space = {"state": 7, "image": [1, 3, 480, 640]}
    action_space = 7

    def __init__(self, args_override, config=None) -> None:
        path = args_override["ckpt_path"]
        logger.info(f"Loading ONNX model from {path}")
        self.path = path
        # 加载和检查模型
        onnx_model = onnx.load(path)
        onnx.checker.check_model(onnx_model)
        # 获取输出层，包含层名称、维度信息
        # output = onnx_model.graph.output
        # logging.info(output)
        try:
            if args_override["temporal_agg"]:
                self.temporal_ensembler = TemporalEnsembling(
                    args_override["chunk_size"],
                    args_override["action_dim"],
                    args_override["max_timesteps"],
                )
        except Exception as e:
            print(e)
            print(
                "The above Exception can be ignored when training instead of evaluating."
            )

    def reset(self):
        if self.temporal_ensembler is not None:
            self.temporal_ensembler.reset()

    def eval(self):
        # 创建推理会话
        self.input_names = []
        self.output_names = []
        self.ort_session = ort.InferenceSession(self.path)
        for i in self.ort_session.get_inputs():
            self.input_names.append(i.name)
        for i in self.ort_session.get_outputs():
            self.output_names.append(i.name)

    def __call__(
        self, qpos: torch.Tensor, image: torch.Tensor, actions=None, is_pad=None
    ) -> torch.Tensor:
        # TODO: change the input data to just one dictionary
        # if image.ndim == 4:
        #     image = image.unsqueeze(0)
        if image.ndim == 5:
            image = image.squeeze(0)
        # Convert PyTorch tensors to NumPy arrays
        qpos = qpos.cpu().numpy()
        qpos = torch.tensor(qpos, device="cuda").cpu().numpy().astype(np.float32)
        # qpos = qpos.astype("float32")
        image = image.cpu().numpy().astype(np.float32)
        input_feed = {}
        # for name in self.input_names:
        #     input_feed[name] = input_data
        # logging.debug(f"input_names: {self.input_names}")
        input_feed = {
            self.input_names[0]: qpos,
            self.input_names[1]: image,
        }
        output = self.ort_session.run(self.output_names, input_feed)
        output = torch.tensor(output[0], device="cuda")
        # logging.error(f"output shape: {output.shape}")
        if self.temporal_ensembler is not None:
            a_hat_one = self.temporal_ensembler.update(output)
            output[0][0] = a_hat_one
        return output


if __name__ == "__main__":

    import numpy as np

    onnx_policy = ONNXPolicy(
        "/home/ghz/Work/OpenGHz/Imitate-All/onnx_output/act_policy_4d.onnx"
    )
    onnx_policy.eval()

    qpos = np.array(
        [
            -0.000190738,
            -0.766194,
            0.702869,
            1.53601,
            -0.964942,
            -1.57607,
            1.01381,
        ]
    )
    img = torch.randn([1, 3, 480, 640], device="cuda")

    qpos_mean = [
        0.01466561,
        -1.1554501,
        1.1064852,
        1.5773835,
        -0.97277683,
        -1.5242718,
        0.48118713,
    ]
    qpos_std = [
        0.14543493,
        0.28088057,
        0.27077574,
        0.190825,
        0.20736837,
        0.22112855,
        0.36785737,
    ]

    action_mean = np.array(
        [
            0.01609754,
            -1.1573728,
            1.1107943,
            1.5749228,
            -0.97393966,
            -1.517563,
            0.45444244,
        ]
    )
    action_std = np.array(
        [
            0.14884493,
            0.28656977,
            0.27433002,
            0.19996108,
            0.22217035,
            0.22898215,
            0.4153349,
        ]
    )

    pre_process = lambda s_qpos: (s_qpos - qpos_mean) / qpos_std
    post_process = lambda a: a * action_std + action_mean

    logging.debug(f"raw qpos: {qpos}")
    qpos = pre_process(qpos)
    logging.debug(f"pre qpos: {qpos}")
    out_put = onnx_policy(qpos.reshape(1, -1), img)
    raw_action = out_put.cpu().numpy()[0][0]

    logging.debug(f"raw action: {raw_action}")
    action = post_process(raw_action)  # de-standardize action
    logging.debug(f"post action: {action}")

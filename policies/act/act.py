import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
from detr.main import build_ACT_model, build_optimizer
from policies.common.loss import kl_divergence
from policies.common.wrapper import TemporalEnsembling


class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, self._args = build_ACT_model(args_override)
        self.model = model  # CVAE decoder
        self.kl_weight = args_override["kl_weight"]
        print(f"KL Weight {self.kl_weight}")
        self.temporal_ensembler = None
        if args_override["temporal_agg"]:
            self.temporal_ensembler = TemporalEnsembling(
                args_override["chunk_size"],
                args_override["action_dim"],
                args_override["max_timesteps"],
            )

    # TODO: 使用装饰器在外部进行包装
    def reset(self):
        if self.temporal_ensembler is not None:
            self.temporal_ensembler.reset()

    # TODO: 使用装饰器在外部进行包装
    def __call__(self, qpos, image, actions=None, is_pad=None):
        if image.ndim == 4:
            image = image.unsqueeze(1)
        env_state = None
        # TODO: imagenet norm, move this outside
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        image = normalize(image)
        if actions is not None:  # training time
            actions = actions[:, : self.model.num_queries]
            assert is_pad is not None, "is_pad should not be None"
            is_pad = is_pad[:, : self.model.num_queries]

            a_hat, is_pad_hat, (mu, logvar) = self.model(
                qpos, image, env_state, actions, is_pad
            )
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction="none")
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict["l1"] = l1
            loss_dict["kl"] = total_kld[0]
            loss_dict["loss"] = loss_dict["l1"] + loss_dict["kl"] * self.kl_weight
            return loss_dict
        else:  # inference time
            a_hat, _, (_, _) = self.model(
                qpos, image, env_state
            )  # no action, sample from prior
            if self.temporal_ensembler is None:
                return a_hat
            else:
                a_hat_one = self.temporal_ensembler.update(a_hat)
                a_hat[0][0] = a_hat_one
            return a_hat

    def configure_optimizers(self):
        self.optimizer = build_optimizer(self.model, self._args)
        return self.optimizer

    def serialize(self):
        return self.state_dict()

    def deserialize(self, model_dict):
        return self.load_state_dict(model_dict)
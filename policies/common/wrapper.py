import numpy as np
import torch


class ActionChunkingExcutor(object):
    def __init__(self, query_period, chunk_size) -> None:
        self.query_period = query_period
        self.chunk_size = chunk_size
        self.all_actions = None

    def reset(self):
        self.t = 0

    def __call__(self, policy) -> None:
        # decorate reset
        if hasattr(policy, "reset"):
            raw_reset = policy.reset

            def reset(*args, **kwargs):
                raw_reset(*args, **kwargs)
                self.reset()

        else:

            def reset(*args, **kwargs):
                self.reset()

        policy.reset = reset

        # decorate call
        raw_call = policy.__class__.__call__

        def call(*args, **kwargs):
            if self.t % self.query_period == 0:
                self.all_actions = raw_call(*args, **kwargs)
            raw_action = self.all_actions[:, self.t % self.query_period]
            self.t += 1
            return raw_action

        policy.__class__.__call__ = call


# TODO: only change the method in the policy class
class TemporalEnsembling(object):
    """Temporal Ensembling to filter out the actions over time"""

    def __init__(self, chunk_size, action_dim, max_timesteps):
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.max_timesteps = max_timesteps
        self.reset()

    def reset(self):
        # use a reseter contains instances to reset before each evaluation
        self.t = 0
        self.all_time_actions = torch.zeros(
            [self.max_timesteps, self.max_timesteps + self.chunk_size, self.action_dim]
        ).cuda()

    def update(self, raw_actions: torch.Tensor) -> torch.Tensor:
        # raw_actions is a tensor of shape [chunk_size, action_dim]
        self.all_time_actions[[self.t], self.t : self.t + self.chunk_size] = raw_actions
        actions_for_curr_step = self.all_time_actions[:, self.t]
        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
        actions_for_curr_step = actions_for_curr_step[actions_populated]
        # TODO: configure the weight function when initiating the class
        k = 0.01
        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
        exp_weights = exp_weights / exp_weights.sum()
        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
        new_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
        self.t += 1
        return new_action

    # TODO: 将这个方法改成对输入的policy的reset和__call__进行装饰，增加新的功能
    def __call__(self, policy) -> None:
        # decorate reset
        if hasattr(policy, "reset"):
            raw_reset = policy.reset

            def reset(*args, **kwargs):
                out = raw_reset(*args, **kwargs)
                self.reset()
                return out

        else:

            def reset(*args, **kwargs):
                return self.reset()

        policy.reset = reset

        # decorate call
        raw_call = policy.__class__.__call__

        def call(*args, **kwargs):
            # TODO： change the dim of output？
            raw_actions: torch.Tensor = raw_call(*args, **kwargs)
            if len(raw_actions.shape) == 2:
                action = self.update(raw_actions.unsqueeze(0))
                raw_actions[0] = action
            else:
                action = self.update(raw_actions)
                raw_actions[0][0] = action
            self.t += 1
            return raw_actions

        policy.__class__.__call__ = call

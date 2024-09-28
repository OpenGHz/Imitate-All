import numpy as np
import torch
import logging
from typing import Union


logger = logging.getLogger(__name__)


class ActionChunkingExcutor(object):
    def __init__(self, query_period, chunk_size) -> None:
        self.query_period = query_period
        self.chunk_size = chunk_size
        self.all_actions = None

    def _reset(self):
        self.t = 0

    def __call__(self, policy) -> None:
        # decorate reset
        if hasattr(policy, "reset"):
            raw_reset = policy.reset

            def reset(*args, **kwargs):
                raw_reset(*args, **kwargs)
                self._reset()

        else:

            def reset(*args, **kwargs):
                self._reset()

        policy.reset = reset

        # decorate call
        raw_call = policy.__class__.__call__

        def call(*args, **kwargs):
            target_t = self.t % self.query_period
            if target_t == 0:
                # update all actions
                self.all_actions = raw_call(*args, **kwargs)
            raw_action = self.all_actions[:, target_t]
            self.t += 1
            return raw_action

        policy.__class__.__call__ = call


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
        # raw_actions is a tensor of shape [1, chunk_size, action_dim]
        # print(raw_actions.shape)
        self.all_time_actions[[self.t], self.t : self.t + self.chunk_size] = raw_actions
        bias = self.t - self.chunk_size + 1
        start = int(max(0, bias))
        end = int(start + self.chunk_size + min(0, bias))
        actions_for_curr_step = self.all_time_actions[start:end, self.t]
        # print(actions_for_curr_step.shape)
        # assert actions_for_curr_step.shape[0] <= self.chunk_size
        # assert actions_for_curr_step.shape[1] == self.action_dim
        # TODO: configure the weight function when initiating the class
        k = 0.01
        exp_weights = np.exp(-k * np.arange(actions_for_curr_step.shape[0]))
        exp_weights = exp_weights / exp_weights.sum()
        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
        new_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
        self.t += 1
        return new_action

    def __call__(self, policy) -> None:
        """Decorate the policy class's reset and __call__ method"""
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


class TemporalEnsemblingWithDroppedActions(object):
    """Temporal Ensembling With Dropped Actions to filter out the actions over time parrellelly with policy prediction"""

    def __init__(self, chunk_size, action_dim, max_timesteps, dropped_num):
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.max_timesteps = max_timesteps
        self.dead_num = dropped_num
        self.max_temporal_len = self.chunk_size // self.dead_num - 1
        self.max_temporal_len_res = self.chunk_size % self.dead_num
        logger.debug("max_temporal_len: %d", self.max_temporal_len)
        logger.debug("max_temporal_len_res: %d", self.max_temporal_len_res)
        self.reset()

        self._prediction_step_max = 1 + (max_timesteps - 1) // dropped_num + 1
        self.buffer_max_col = 1 + (self._prediction_step_max - 2) * dropped_num + chunk_size

    @staticmethod
    def get_dropped_action_num(action_freq, predict_freq) -> int:
        return np.ceil(action_freq / predict_freq).astype(int)

    def reset(self):
        self.t = 0
        self.infer_t = 0

    def update(self, all_time_actions: Union[torch.Tensor, np.ndarray], return_type:str="input") -> Union[torch.Tensor, np.ndarray]:
        # max_temporal_len + res + 初始行（可能）才为实际的
        ddl_num = self.infer_t - 2
        logger.debug("ddl_num: %d", ddl_num)
        dead_local_id = (self.t - 1) % self.dead_num
        if dead_local_id < self.max_temporal_len_res:
            temporal_res = 1
        else:
            temporal_res = 0
        logger.debug("dead_local_id: %d", dead_local_id)
        # logger.debug("temporal_res: %d", temporal_res)
        start = int(ddl_num - self.max_temporal_len + 1 - temporal_res)
        logger.debug(f"raw start: {start}")
        if start <= 1:
            if self.t < self.chunk_size:
                start = 0
            else:
                start = 1

        end = int(max(start + 1, ddl_num + 1))
        logger.debug(f"post start: {start}, end: {end}")
        actions_for_curr_step = all_time_actions[start:end, self.t]
        logger.debug(f"actions_for_curr_step:{actions_for_curr_step}")
        # logger.warning(f"actions_for_curr_step.shape: {actions_for_curr_step.shape}")
        # assert actions_for_curr_step.shape[0] <= self.chunk_size
        # assert actions_for_curr_step.shape[1] == self.action_dim
        # TODO: configure the weight function when initiating the class
        k = 0.01
        exp_weights = np.exp(-k * np.arange(actions_for_curr_step.shape[0]))
        exp_weights = exp_weights / exp_weights.sum()
        logger.debug(f"exp_weights: {exp_weights}")
        use_numpy = isinstance(actions_for_curr_step, np.ndarray)
        if use_numpy:
            actions_for_curr_step = torch.from_numpy(actions_for_curr_step).cuda()
        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
        new_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
        self.t += 1
        if (use_numpy and return_type == "input") or return_type == "numpy":
            new_action = new_action.cpu().numpy()
        return new_action

    def need_infer(self):
        if (self.t % self.dead_num == 1) or self.t == 0:
            self.infer_t += 1
            return True

    def __call__(self, policy) -> None:
        """Decorate the policy class's reset and __call__ method"""
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

    def get_action_buffer(self):
        return torch.zeros(
            [self._prediction_step_max, self.buffer_max_col, self.action_dim]
        ).cuda()

    @property
    def prediction_step_max(self):
        return self._prediction_step_max
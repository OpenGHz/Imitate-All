import numpy as np
import torch


# TODO: only change the method in the policy class
class TemporalEnsembling(object):
    """Temporal Ensembling to filter out the actions over time"""

    def __init__(self, policy, chunk_size, action_dim, max_timesteps):
        self.policy = policy
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.max_timesteps = max_timesteps

    def reset(self):
        self.t = 0
        # TODO: change to circular queue with lenth of chunk size to reduce memory usage
        self.all_time_actions = torch.zeros(
            [self.max_timesteps, self.max_timesteps + self.chunk_size, self.action_dim]
        ).cuda()

    def __call__(self, *args, **kwargs):
        # raw_actions is a tensor of shape [chunk_size, action_dim]
        raw_actions = self.policy(*args, **kwargs)
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

    def save(self, filename):
        return self.policy.save(filename)

    def load(self, filename):
        return self.policy.load(filename)

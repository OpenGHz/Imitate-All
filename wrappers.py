from typing import Callable, Tuple, List, Any, Union, Literal, final
from abc import ABC, abstractmethod
from pydantic import BaseModel, PositiveInt, NonNegativeInt
from logging import getLogger
import numpy as np


class WrapperBasis(ABC):
    caller: Callable
    # the chained outputs of the wrappers, cleared on reset
    output_chain: list = []

    @final
    def wrap(self, caller: Callable):
        """Wrap the callable with additional functionality."""
        caller_type = self.__annotations__["caller"]
        if caller_type == Callable:
            if not callable(caller):
                raise TypeError("caller must be callable")
        elif not isinstance(caller, caller_type):
            raise TypeError(
                f"Expected callable of type {self.caller}, got {type(caller)}"
            )
        self.caller = caller
        self._warmed_up = False
        return self

    @final
    def warm_up(self, *args, **kwds):
        """Warm up the wrapper"""
        if not hasattr(self, "caller"):
            raise RuntimeError("No caller, please wrap first")
        output = self.caller(*args, **kwds)
        if hasattr(output, "shape"):
            if len(output.shape) in (3, 5):
                self.output_type = type(output)
                self.output_type_name = type(output).__name__
                self.output_dtype = output.dtype
                if self.output_type_name == "ndarray":
                    backend = np
                    self.output_device = "cpu"
                elif self.output_type_name == "Tensor":
                    import torch as backend

                    self.output_device = output.device
                else:
                    raise TypeError(f"Unsupported output type: {self.output_type_name}")
                self.output_backend = backend
                self._warmed_up = True
                return output
            raise ValueError(
                "The shape of the caller output must be (B, T, C, H, W) or (B, T, D), "
                "i.e. (batch, time, channel, height, width) or (batch, time, dimension)"
            )
        raise TypeError("The caller output must have a shape")

    @final
    def reset(self):
        """Reset the internal state of the wrapper, if any."""
        if not self._warmed_up:
            raise RuntimeError("Please warm up first")
        self.output_chain.clear()
        return self.on_reset()

    @abstractmethod
    def on_reset(self):
        """Hook when reset is called."""

    @abstractmethod
    def call(self, *args, **kwds):
        """Call the wrapped callable."""

    @final
    def __call__(self, *args, **kwds):
        """Call the wrapped callable."""
        output = self.call(*args, **kwds)
        self.output_chain.append(output)
        return output

    @classmethod
    def get_logger(cls):
        return getLogger(cls.__name__)

    @staticmethod
    def calculate_size(data) -> float:
        type_name = type(data).__name__
        if type_name == "Tensor":
            size_mb = data.numel() * data.element_size() / (1024**2)
        elif type_name == "ndarray":
            size_mb = data.nbytes / (1024**2)
        else:
            raise TypeError(f"Unsupported type name: {type_name}")
        return size_mb


class FrequencyReductionCallConfig(BaseModel):
    # 1 means call at each step
    period: PositiveInt = 1


class FrequencyReductionCall(WrapperBasis):
    def __init__(self, config: FrequencyReductionCallConfig) -> None:
        self.config = config

    def warm_up(self, *args, **kwds):
        output = super().warm_up(*args, **kwds)
        horizon = output.shape[1]
        period = self.config.period
        if horizon < period:
            raise ValueError(
                f"output horizon: {horizon} can not be shorter than period: {period}"
            )
        return output

    def on_reset(self):
        self._t = 0

    def call(self, *args, **kwds):
        target_t = self._t % self.config.period
        if target_t == 0:
            self._outputs = self.caller(*args, **kwds)
        self._t += 1
        return self._outputs[:, target_t]


class TemporalEnsemblingWithDroppingConfig(BaseModel):
    drop_num: NonNegativeInt = 0
    # empty list means average, float means exp decay
    weights: Union[List[float], float] = []
    max_timesteps: NonNegativeInt = 0


class TemporalEnsemblingWithDropping(WrapperBasis):
    """Temporal Ensembling for output time serial data with dropping method"""

    caller: Callable

    def __init__(self, config: TemporalEnsemblingWithDroppingConfig):
        self.config = config
        # TODO: use a dynamic method to adjust the
        # buffer size instead of allocating a very
        # large memory
        if self.config.max_timesteps == 0:
            self.config.max_timesteps = 2048

    def warm_up(self, *args, **kwds):
        output = super().warm_up(*args, **kwds)
        self._horizon = output.shape[1]
        if self.config.drop_num >= self._horizon:
            raise ValueError(
                f"drop_num {self.config.drop_num} must be less than the output horizon {self._horizon}"
            )
        self._ele_shape = output.shape[2:]
        weights = self.config.weights
        if isinstance(weights, float):
            weights = np.exp(-weights * np.arange(self._horizon))
        else:
            if not weights:
                weights = np.ones(self._horizon)
            elif len(weights) != self._horizon:
                raise ValueError(
                    f"The length of the weights {weights} must be equal to {self._horizon}"
                )
            weights = np.array(weights)
        if self.output_type_name == "Tensor":
            weights = (
                self.output_backend.from_numpy(weights)
                .to(device=self.output_device)
                .unsqueeze(dim=1)  # add a batch dim
            )
        self._weights = weights / weights.sum()
        # print(self._weights.shape)
        return output

    def on_reset(self):
        self._t = 0
        self._call_t = 0
        kwds = {}
        if self.output_type == "Tensor":
            kwds = {"device": self.output_device}
        self._all_time_outputs = self.output_backend.zeros(
            (self.config.max_timesteps, self.config.max_timesteps + self._horizon)
            + self._ele_shape,
            **kwds,
            dtype=self.output_dtype,
        )
        self.get_logger().info(
            f"output buffer shape: {self._all_time_outputs.shape}, size: {self.calculate_size(self._all_time_outputs):.2f} MB"
        )

    def call(self, *args, **kwds):
        t = self._t
        # need call
        if t == 0 or self.config.drop_num == 0 or ((t - 1) % self.config.drop_num == 0):
            outputs = self.caller(*args, **kwds)
            # x axis uses t, y axis uses call_t
            self._all_time_outputs[[self._call_t], t : t + self._horizon] = outputs
            self._call_t += 1
        drop_num = self.config.drop_num
        hori = self._horizon
        eq_drop_num = 1 if drop_num == 0 else drop_num
        start = max(0, (t - (hori - eq_drop_num) - 1)) // eq_drop_num + (t > hori - 1)
        end = max(0, (t - 1) // drop_num if drop_num > 0 else t) + 1
        self._t += 1
        used_outputs = self._all_time_outputs[start:end, t]
        used_weights = self._weights[: end - start]
        # print(used_outputs)
        return (used_outputs * used_weights).sum(dim=0, keepdim=True)


class HorizonConfig(BaseModel):
    input: int = 1
    output: int = 1


class MockCallerConfig(BaseModel):
    output_type: Literal["ndarray", "Tensor", "list"] = "list"
    horizon: HorizonConfig = HorizonConfig()


class MockCaller:
    def __init__(self, config: MockCallerConfig):
        self.config = config

    def reset(self):
        """Do nothing"""

    def __call__(self, data: Tuple[float]):
        lis = [[[data[0] + i] for i in range(self.config.horizon.output)]]
        if self.config.output_type == "list":
            return lis
        elif self.config.output_type == "ndarray":
            return np.array(lis)
        elif self.config.output_type == "Tensor":
            return torch.tensor(lis)


class EnvironmentOutput(BaseModel):
    # The current observation from the environment
    observation: Any
    # The reward obtained from the last action
    reward: float = 0.0
    # Whether the agent reaches the terminal state (as defined under the MDP of the task) which can be positive or negative. If true, the user needs to call reset().
    terminated: bool = False
    # Whether the truncation condition outside the scope of the MDP is satisfied. Typically, this is a timelimit, but could also be used to indicate an agent physically going out of bounds. Can be used to end the episode prematurely before a terminal state is reached. If true, the user needs to call reset().
    truncated: bool = False
    # Contains auxiliary diagnostic information (helpful for debugging, learning, and logging). This might, for instance, contain: metrics that describe the agent’s performance state, variables that are hidden from observations, or individual reward terms that are combined to produce the total reward.
    info: dict = {}


class EnvironmentBasis(ABC):
    @abstractmethod
    def reset(self) -> None:
        """Reset the environment"""

    @abstractmethod
    def input(self, input: Any) -> None:
        """Take an action in the environment"""

    @abstractmethod
    def output(self) -> EnvironmentOutput:
        """Get the current output from the environment"""


class MockEnvironmentConfig(BaseModel):
    max_steps: int = 10
    output: Any = None


class MockEnvironment(EnvironmentBasis):
    def __init__(self, config: MockEnvironmentConfig):
        self._t = 0
        self.config = config

    def reset(self) -> None:
        self._t = 0

    def input(self, input: Any) -> None:
        self._t += 1

    def output(self) -> EnvironmentOutput:
        terminated = self._t >= 10
        return EnvironmentOutput(
            observation=np.array([self._t])
            if self.config.output is None
            else self.config.output,
            terminated=terminated,
        )


if __name__ == "__main__":
    import torch
    import logging
    from itertools import count

    logging.basicConfig(level=logging.INFO)

    caller = MockCaller(
        MockCallerConfig(output_type="Tensor", horizon=HorizonConfig(output=3))
    )

    wrapped = caller

    ted = TemporalEnsemblingWithDropping(
        # TemporalEnsemblingWithDroppingConfig(weights=0.01)
        TemporalEnsemblingWithDroppingConfig(weights=[], drop_num=1)
    )

    env = MockEnvironment(MockEnvironmentConfig(output=(0.0,), max_steps=8))
    env.reset()

    init_input = env.output().observation
    wrappers = [ted]
    # wrap and warm up all the wrappers once
    for wrapper in wrappers:
        wrapped = wrapper.wrap(wrapped)
        wrapped.warm_up(init_input)

    rollouts = 1
    for r in range(rollouts):
        # reset the caller, environment, and all the wrappers
        caller.reset()
        for wrapper in wrappers:
            wrapper.reset()
        # step through the environment
        for step in count():
            env_output = env.output()
            if env_output.terminated:
                print(f"rollout {r} terminated at step {step}")
                break
            wrapped_output = ted(env_output.observation)
            print(f"rollout: {r}, step: {step}, output: {wrapped_output}")
            env.input(wrapped_output)

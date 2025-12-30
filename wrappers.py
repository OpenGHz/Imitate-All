import numpy as np
from typing import Callable, Tuple, List, Any, Union, Literal, final
from typing_extensions import Self
from abc import ABC, abstractmethod
from pydantic import BaseModel, PositiveInt, NonNegativeInt
from logging import getLogger
from airdc.common.utils.event_rpc import (
    EventRpcManager,
    ConcurrentMode,
    EventRpcServer,
)


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
        """Get the current output from the environment.
        This method should not change the state of the environment.
        """

    @abstractmethod
    def shutdown(self) -> bool:
        """Shutdown the environment"""

    @classmethod
    def get_logger(cls):
        return getLogger(cls.__name__)


class MockEnvironmentConfig(BaseModel):
    max_steps: int = 0
    output: Any = None


class MockEnvironment(EnvironmentBasis):
    def __init__(self, config: MockEnvironmentConfig):
        self.config = config
        self._output_t = 0

    def reset(self) -> None:
        # self.get_logger().info("Environment reset")
        self._t = 0

    def input(self, input: Any) -> None:
        # self.get_logger().info(f"Environment input: {input}")
        self._t += 1

    def output(self) -> EnvironmentOutput:
        # self._output_t += 1
        # if self._t != self._output_t:
        #     raise RuntimeError(
        #         "The environment output is out of sync with the input. Please make sure to call input() before output()"
        #     )
        terminated = (
            (self._t >= self.config.max_steps) if self.config.max_steps > 0 else False
        )
        # self.get_logger().info(f"Environment step: {self._t}, terminated: {terminated}")
        return EnvironmentOutput(
            observation=np.array([self._t])
            if self.config.output is None
            else self.config.output,
            terminated=terminated,
        )

    def shutdown(self) -> bool:
        return True


class WrapperBasis(ABC):
    caller: Callable
    # the chained outputs of the wrappers, cleared on reset
    output_chain: list = []
    should_take_over_env: bool = False

    @final
    def wrap(self, caller: Callable) -> Self:
        """Wrap the callable with additional functionality."""
        caller_type = self.__annotations__["caller"]
        if caller_type == Callable:
            if not callable(caller):
                raise TypeError("The caller must be callable")
        elif not isinstance(caller, caller_type):
            raise TypeError(
                f"Expected callable of type {self.caller}, got {type(caller)}"
            )
        self._warmed_up = False
        self.caller = caller
        self.env = None
        return self

    @final
    def take_over_env(self, env: EnvironmentBasis):
        """Set the environment for the topmost wrapper.
        If the subclass needs to take over env, please
        set `self.should_take_over_env = True` in any of __init__ or warm_up.
        """
        if not self._warmed_up:
            raise RuntimeError("Please warm up first")
        if not self.should_take_over_env:
            raise RuntimeError("This wrapper has no need to take over the environment")
        if not isinstance(env, EnvironmentBasis):
            raise TypeError("The `env` must be an instance of EnvironmentBasis")
        self.env = env

    @final
    def warm_up(self, *args, **kwds) -> Any:
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
                return self.on_warm_up(output)
            raise ValueError(
                "The shape of the caller output must be (B, T, C, H, W) or (B, T, D), "
                "i.e. (batch, time, channel, height, width) or (batch, time, dimension)."
                f"But got shape: {output.shape}"
            )
        raise TypeError("The caller output must have a shape")

    @final
    def reset(self):
        """Reset the internal state of the wrapper, if any."""
        if not self._warmed_up:
            raise RuntimeError("Please warm up first")
        if self.taken_over_env:
            self.env.reset()
            self.last_env_output = self.env.output()
        return self.on_reset()

    @abstractmethod
    def on_reset(self):
        """Hook when `reset` is called."""

    @abstractmethod
    def on_warm_up(self, output: Any):
        """Hook when `warm_up` is called."""

    @abstractmethod
    def call(self, *args, **kwds) -> Any:
        """Call the wrapped callable and return the output."""

    @final
    def __call__(self, *args, **kwds):
        """Call the wrapped callable and update the output chain.
        If the env is not taken over, this method will return
        the output of the wrapped caller, and return the last output of env
        otherwise.
        """
        output = self.call(*args, **kwds)
        self.output_chain.append(output)
        if self.taken_over_env:
            self.env.input(output)
            return self.last_env_output
        return output

    @final
    def shutdown(self) -> bool:
        """Shutdown the wrapper and the environment."""
        if self.on_shutdown():
            if self.taken_over_env:
                return self.env.shutdown()
            return True
        return False

    def on_shutdown(self) -> bool:
        """Shutdown the wrapper, if any."""
        # NOTE: since most wrappers do not need to shutdown,
        # we provide a default implementation that does nothing
        return True

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

    @property
    def taken_over_env(self) -> bool:
        """Whether the wrapper takes over the environment"""
        return self.env is not None


class ForwardingWrapper(WrapperBasis):
    """A wrapper that just forwards the call to the caller"""

    def on_warm_up(self, output: Any):
        return output

    def on_reset(self):
        """Do nothing"""

    def call(self, *args, **kwds) -> Any:
        return self.caller(*args, **kwds)


class TakeOverEnvWrapper(WrapperBasis):
    """A wrapper that just takes over the environment"""

    def __init__(self):
        self.should_take_over_env = True

    def on_warm_up(self, output: Any):
        return output

    def on_reset(self):
        """Do nothing"""

    def call(self) -> EnvironmentOutput:
        """Call the environment, using the output
        observation as the caller input and returning
        the environment output
        """
        env_output = self.env.output()
        call_output = self.caller(env_output.observation)
        self.env.input(call_output)
        self.last_env_output = env_output
        return env_output

    def __call__(self, *args, **kwds) -> EnvironmentOutput:
        return super().__call__(*args, **kwds)


class NormalizationConfig(BaseModel):
    mean: float = 0.0
    std: float = 0.0
    min_std: float = 1e-8

    def model_post_init(self, context):
        self.std = self.std or self.min_std


class NormalizerConfig(BaseModel):
    """Configuration to normalize the input and denormalize the output"""

    input: NormalizationConfig = NormalizationConfig()
    output: NormalizationConfig = NormalizationConfig()
    # if input data is a dict and input_key is not empty, normalize the data[input_key]
    input_key: str = ""


class Normalizer(WrapperBasis):
    def __init__(self, config: NormalizerConfig):
        self.config = config

    def normalize_input(self, data: Any) -> Any:
        if isinstance(data, dict):
            key = self.config.input_key
            if not key:
                raise ValueError("data is a dict, but key is empty")
            data[key] = self._normalize(data[key])
            return data
        return self._normalize(data)

    def _normalize(self, data: Any) -> Any:
        return (data - self.config.input.mean) / self.config.input.std

    def denormalize_output(self, data: Any) -> Any:
        return data * self.config.output.std + self.config.output.mean

    def on_warm_up(self, output: Any):
        """Do nothing"""

    def on_reset(self):
        """Do nothing"""

    def call(self, *args, **kwds):
        return self.denormalize_output(self.caller(self.normalize_input(*args, **kwds)))


class FrequencyReductionCallConfig(BaseModel):
    # 1 means call at each step
    period: PositiveInt = 1


class FrequencyReductionCall(WrapperBasis):
    def __init__(self, config: FrequencyReductionCallConfig) -> None:
        self.config = config

    def on_warm_up(self, output: Any):
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
        if self.config.drop_num > 0:
            self.should_take_over_env = True
            mode = ConcurrentMode.thread
            self._rpc_manager = EventRpcManager(EventRpcManager.get_args(mode))
            self._call_thread = self._rpc_manager.get_concurrent_cls(mode)(
                target=self._async_call, args=(self._rpc_manager.server,), daemon=True
            )
            self._call_thread.start()

    def on_warm_up(self, output: Any):
        self._horizon = output.shape[1]
        if self.config.drop_num >= self._horizon:
            raise ValueError(
                f"The `drop_num`: {self.config.drop_num} must be less than the output horizon {self._horizon}"
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
        self._async_t = 0
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
            f"Output buffer shape: {self._all_time_outputs.shape}, size: {self.calculate_size(self._all_time_outputs):.2f} MB"
        )

    def _async_call(self, rpc_server: EventRpcServer):
        self.get_logger().info("Call thread started")
        while rpc_server.wait():
            output = self.env.output()
            self.last_env_output = output
            self._sync_call(output.observation)
            rpc_server.respond()
        self.get_logger().info("Call thread exited")

    def _sync_call(self, *args, **kwds):
        t = self._async_t
        outputs = self.caller(*args, **kwds)
        # x axis uses t, y axis uses call_t
        self._all_time_outputs[[self._call_t], t : t + self._horizon] = outputs
        self._call_t += 1

    def call(self, *args, **kwds):
        t = self._t
        # need call
        if t == 0 or self.config.drop_num == 0 or ((t - 1) % self.config.drop_num == 0):
            self._async_t = t
            if self.taken_over_env:
                client = self._rpc_manager.client
                if not client.is_responded():
                    self.get_logger().info("Waiting for call thread to respond...")
                if client.wait(5.0):
                    if not client.request(0):
                        raise RuntimeError("Failed to request")
                    if t == 0:
                        client.wait(5.0)
                else:
                    raise TimeoutError("Timeout waiting for call thread to be ready")
            else:
                self._sync_call(*args, **kwds)

        drop_num = self.config.drop_num
        hori = self._horizon
        eq_drop_num = 1 if drop_num == 0 else drop_num
        start = max(0, (t - (hori - eq_drop_num) - 1)) // eq_drop_num + (t > hori - 1)
        end = max(0, (t - 1) // drop_num if drop_num > 0 else t) + 1
        used_outputs = self._all_time_outputs[start:end, t]
        used_weights = self._weights[: end - start]
        # print(used_outputs)
        self._t += 1
        return (used_outputs * used_weights).sum(dim=0, keepdim=True)

    def on_shutdown(self):
        if self.taken_over_env:
            if self._rpc_manager.shutdown():
                self._call_thread.join(5.0)
                return not self._call_thread.is_alive()
        return True


class CallerBasis(ABC):
    @abstractmethod
    def reset(self):
        """Reset the internal state of the caller, if any."""

    @abstractmethod
    def __call__(self, *args, **kwds) -> Any:
        """Call the caller with the given inputs."""


class HorizonConfig(BaseModel):
    input: int = 1
    output: int = 1


class MockCallerConfig(BaseModel):
    output_type: Literal["ndarray", "Tensor", "list"] = "list"
    horizon: HorizonConfig = HorizonConfig()


class MockCaller(CallerBasis):
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


class PolicyEvaluationCallerConfig(BaseModel):
    """Configuration for policy evaluation."""

    # max number of evaluation steps per rollout, 0 means no limit
    max_steps: NonNegativeInt = 0
    # number of rollouts to evaluate, 0 means infinite
    num_rollouts: NonNegativeInt = 0
    # checkpoint path to load the policy from
    checkpoint_path: str = ""


class PolicyEvaluationCaller(CallerBasis):
    def __init__(self, config: PolicyEvaluationCallerConfig):
        self.config = config

    def reset(self):
        """Reset the internal state of the caller, if any."""

    def __call__(self, *args, **kwds) -> Any:
        """Call the caller with the given inputs."""


if __name__ == "__main__":
    import torch
    import logging
    from itertools import count
    from pprint import pformat

    logging.basicConfig(level=logging.INFO)

    logger = getLogger("Main")

    caller = MockCaller(
        MockCallerConfig(output_type="Tensor", horizon=HorizonConfig(output=3))
    )
    if not isinstance(caller, CallerBasis):
        logger.warning("Caller does not inherit from CallerBasis")
        if not hasattr(caller, "reset"):
            logger.warning("Caller does not have reset method")
            caller.reset = lambda: None
    caller.reset()

    ted = TemporalEnsemblingWithDropping(
        # TemporalEnsemblingWithDroppingConfig(weights=0.01)
        TemporalEnsemblingWithDroppingConfig(drop_num=1, weights=[])
    )

    # the rightmost will be the topmost
    wrappers = [ted]

    # create the environment and reset it
    env = MockEnvironment(MockEnvironmentConfig(max_steps=8, output=(0.0,)))
    env.reset()

    # wrap, warm up and reset all the wrappers once
    # using the initial observation from the environment
    init_input = env.output().observation
    # wrap by a forwarding wrapper to make a complete output chain
    wrapped = ForwardingWrapper().wrap(caller)
    for i, wrapper in enumerate(wrappers):
        wrapped = wrapper.wrap(wrapped)
        wrapped.warm_up(init_input)
        # reset all the wrapped wrappers since the topper
        # wrapper will call it when warming up
        for wp in wrappers[: i + 1]:
            wp.reset()
        WrapperBasis.output_chain = []
    # check that only the topmost wrapper can take over the environment
    # and reset all the other wrappers since they have been called
    # once during warming up the topmost wrapper
    for wrapper in wrappers[1:]:
        if wrapper.should_take_over_env:
            raise RuntimeError("Only the topmost wrapper can take over the environment")

    # take over the environment
    if not wrapped.should_take_over_env:
        wrapped.get_logger().info("Do not take over the environment")
        wrapped = TakeOverEnvWrapper().wrap(wrapped)
        # wrapped.warm_up()  # actually no need to warm up
        # WrapperBasis.output_chain = []
    wrapped.get_logger().info("Taking over the environment")
    wrapped.take_over_env(env)

    rollouts = 2
    for r in range(rollouts):
        if rollouts > 1 and input(
            "Press `Enter` to start a new rollout or 'q'/`z` to quit..."
        ) in ("q", "z"):
            break
        # reset the caller
        caller.reset()
        # reset all the wrappers
        for wrapper in wrappers:
            wrapper.reset()
        # step through the environment
        for step in count():
            env_output = wrapped()
            logger.info(f"rollout: {r}, step: {step}, env_output: {env_output}")
            logger.info(f"output_chain: \n{pformat(wrapped.output_chain)}")
            if env_output.terminated:
                break
            # clear is O(n) but assigning a empty list is O(1)
            WrapperBasis.output_chain = []
            # input("Press Enter to take the next step...")
    # shutdown all the wrappers in reversed order
    # i.e. from the topmost to the bottommost, so
    # that the environment is shutdown last
    for wrapper in reversed(wrappers):
        if not wrapper.shutdown():
            logger.error("Failed to shutdown the wrapper")
    logger.info("Done.")

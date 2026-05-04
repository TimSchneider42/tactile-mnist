from __future__ import annotations

from typing import Any, Generic, TypeVar, Callable

import gymnasium as gym
import numpy as np

from .active_perception_env import BaseActivePerceptionEnv
from .active_perception_vector_env import BaseActivePerceptionVectorEnv

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
PredType = TypeVar("PredType")
PredTargetType = TypeVar("PredTargetType")
VectorEnvType = TypeVar("VectorEnvType", bound=gym.vector.VectorEnv)


class VectorToSingleWrapper(
    gym.Env[ObsType, ActType], Generic[ObsType, ActType, VectorEnvType]
):
    def __init__(self, vector_env: VectorEnvType):
        self.__vector_env = vector_env

    @classmethod
    def _tree_map(cls, f: Callable[[Any], Any], data: Any) -> Any:
        if isinstance(data, dict):
            return {k: cls._tree_map(f, v) for k, v in data.items()}
        elif isinstance(data, tuple):
            return tuple(cls._tree_map(f, v) for v in data)
        else:
            return f(data)

    @staticmethod
    def _vectorize(value: Any):
        if isinstance(value, np.ndarray):
            return value[np.newaxis]
        else:
            return np.array([value])

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any | None] = None
    ) -> tuple["ObsType", dict[str, Any]]:
        options = (
            self._tree_map(self._vectorize, options) if options is not None else None
        )
        obs, info = self.__vector_env.reset(seed=seed, options=options)
        obs = self._tree_map(lambda x: np.asarray(x[0]), obs)
        info = self._tree_map(lambda x: np.asarray(x[0]), info)
        return obs, info

    def step(
        self, action: "ActType"
    ) -> tuple["ObsType", float, bool, bool, dict[str, Any]]:
        action = self._tree_map(self._vectorize, action)
        obs, reward, terminated, truncated, info = self.__vector_env.step(action)
        obs = self._tree_map(lambda x: np.asarray(x[0]), obs)
        info = self._tree_map(lambda x: np.asarray(x[0]), info)
        return obs, reward[0], terminated[0], truncated[0], info

    def render(self) -> np.ndarray | None:
        return self.__vector_env.render()[0]

    def close(self):
        self.__vector_env.close()

    def __getattr__(self, item):
        return getattr(self.__vector_env, item)

    @property
    def action_space(self):
        return self.__vector_env.single_action_space

    @property
    def observation_space(self):
        return self.__vector_env.single_observation_space

    @property
    def metadata(self):
        return self.__vector_env.metadata

    @property
    def render_mode(self):
        return self.__vector_env.render_mode

    @property
    def spec(self):
        return self.__vector_env.spec

    @spec.setter
    def spec(self, value):
        self.__vector_env.spec = value

    @property
    def _np_random(self):
        return self.__vector_env._np_random

    @property
    def _np_random_seed(self):
        return self.__vector_env._np_random_seed

    @property
    def vector_env(self):
        return self.__vector_env


class ActivePerceptionVectorToSingleWrapper(
    VectorToSingleWrapper[
        ObsType,
        ActType,
        BaseActivePerceptionVectorEnv[ObsType, ActType, PredType, PredTargetType, Any],
    ],
    BaseActivePerceptionEnv[ObsType, ActType, PredType, PredTargetType],
    Generic[ObsType, ActType, PredType, PredTargetType],
):
    def __init__(
        self,
        vector_env: BaseActivePerceptionVectorEnv[
            ObsType, ActType, PredType, PredTargetType, Any
        ],
    ):
        super().__init__(vector_env)

    @property
    def prediction_target_space(self):
        return self.vector_env.single_prediction_target_space

    @property
    def loss_fn(self):
        return self.vector_env.loss_fn

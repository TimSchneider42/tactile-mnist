from copy import deepcopy
from typing import Generic, Any

import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import parse_env_id, get_env_id
from gymnasium.vector.utils import batch_space

from .active_perception_env import (
    ActivePerceptionWrapper,
    BaseActivePerceptionEnv,
    ensure_active_perception_env,
)
from .active_perception_vector_env import (
    ActivePerceptionVectorWrapper,
    BaseActivePerceptionVectorEnv,
    ensure_active_perception_vector_env,
)
from .loss_fn import WeightedLossFn
from .types import ObsType, ActType, PredType, PredTargetType, ArrayType


class SparsifyVectorWrapper(
    ActivePerceptionVectorWrapper[
        ObsType,
        ActType,
        PredType,
        PredTargetType,
        ArrayType,
        ObsType,
        ActType,
        PredType,
        PredTargetType,
        ArrayType,
    ],
    Generic[ObsType, ActType, PredType, PredTargetType, ArrayType],
):
    def __init__(
        self,
        env: (
            BaseActivePerceptionVectorEnv[
                ObsType, ActType, PredType, PredTargetType, ArrayType
            ]
            | gym.vector.VectorEnv
        ),
    ):
        env = ensure_active_perception_vector_env(env)
        super().__init__(env)
        self._single_prediction_target_space = gym.spaces.Dict(
            {
                "target": env.single_prediction_target_space,
                "weight": gym.spaces.Box(0, 1, shape=(), dtype=np.float32),
            }
        )
        self._prediction_target_space = batch_space(
            self._single_prediction_target_space,
            self.num_envs,
        )
        self._loss_fn = WeightedLossFn(env.loss_fn)

    def __info_add_weight(
        self, info: dict[str, Any], terminated: np.ndarray | None = None
    ) -> dict[str, Any]:
        info = deepcopy(info)
        if terminated is None:
            terminated = np.zeros(self.num_envs, dtype=bool)
        weight = terminated.astype(np.float32)
        info["prediction"]["target"] = {
            "target": info["prediction"]["target"],
            "weight": weight,
        }
        return info

    def step(
        self, actions: ActType
    ) -> tuple[ObsType, ArrayType, ArrayType, ArrayType, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(actions)
        info = self.__info_add_weight(info, terminated=terminated)
        reward = info["base_reward"] - self.loss_fn(
            actions["prediction"], info["prediction"]["target"], (self.num_envs,)
        )
        return (
            obs,
            reward,
            terminated,
            truncated,
            info,
        )

    @property
    def __idoc__(self) -> dict[str, Any]:
        return _get_sparse_idoc(self)


class SparsifyWrapper(
    ActivePerceptionWrapper[
        ObsType,
        ActType,
        PredType,
        PredTargetType,
        ObsType,
        ActType,
        PredType,
        PredTargetType,
    ],
    gym.utils.RecordConstructorArgs,
    Generic[ObsType, ActType, PredType, PredTargetType, ArrayType],
):
    def __init__(
        self,
        env: (
            BaseActivePerceptionEnv[ObsType, ActType, PredType, PredTargetType]
            | gym.Env
        ),
    ):
        env = ensure_active_perception_env(env)
        ActivePerceptionWrapper.__init__(self, env)
        gym.utils.RecordConstructorArgs.__init__(self)
        self._prediction_target_space = gym.spaces.Dict(
            {
                "target": env.prediction_target_space,
                "weight": gym.spaces.Box(0, 1, shape=(), dtype=np.float32),
            }
        )
        self._loss_fn = WeightedLossFn(env.loss_fn)

    def __info_add_weight(
        self, info: dict[str, Any], terminated: bool = False
    ) -> dict[str, Any]:
        info = deepcopy(info)
        info["prediction"]["target"] = {
            "target": info["prediction"]["target"],
            "weight": 1.0 if terminated else 0.0,
        }
        return info

    def step(
        self, actions: ActType
    ) -> tuple[ObsType, ArrayType, ArrayType, ArrayType, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(actions)
        info = self.__info_add_weight(info, terminated=terminated)
        reward = info["base_reward"] - self.loss_fn(
            actions["prediction"], info["prediction"]["target"]
        )
        return (
            obs,
            reward,
            terminated,
            truncated,
            info,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        return obs, self.__info_add_weight(info)

    @property
    def __idoc__(self) -> dict[str, Any]:
        return _get_sparse_idoc(self)


def _get_sparse_idoc(env: SparsifyVectorWrapper | SparsifyWrapper) -> dict[str, Any]:
    inner_env = env.env
    while hasattr(inner_env, "env") and not hasattr(inner_env, "__idoc__"):
        inner_env = inner_env.env
    if hasattr(inner_env, "__idoc__"):
        orig_idoc = inner_env.__idoc__
    else:
        orig_idoc = {}
    if env.spec.id is not None:
        ns, name, version = parse_env_id(env.spec.id)
        if name.endswith("-sparse"):
            name_orig = name[: -len("-sparse")]
            orig_id = get_env_id(ns, name_orig, version)
            return {
                **orig_idoc,
                "description": f"Variant of {orig_id} in which the loss is masked (set zero) except for the final "
                f"step of each episode.",
            }
    return orig_idoc

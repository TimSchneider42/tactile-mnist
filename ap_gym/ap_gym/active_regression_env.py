from __future__ import annotations

from abc import ABC
from collections import deque, defaultdict
from typing import Generic, Any, SupportsFloat

import gymnasium as gym
import numpy as np

from .active_perception_env import (
    ActivePerceptionEnv,
    ActivePerceptionActionSpace,
    ActivePerceptionWrapper,
)
from .active_perception_vector_env import (
    ActivePerceptionVectorEnv,
    ActivePerceptionVectorWrapper,
    FullActType,
    PredType,
)
from .loss_fn import MSELossFn
from .types import ObsType, ActType
from .util import update_info_metrics, update_info_metrics_vec
import logging

logger = logging.getLogger(__name__)


def _make_mse_loss_fn_and_target_space(
    target_dim: int,
    prediction_low: float | None = None,
    prediction_high: float | None = None,
    target_std: float | None = None,
) -> tuple[MSELossFn, gym.spaces.Box]:
    prediction_target_space = gym.spaces.Box(
        low=prediction_low, high=prediction_high, shape=(target_dim,)
    )
    if target_std is not None:
        target_std = target_std
    elif np.isfinite(prediction_low) and np.isfinite(prediction_high):
        # Assume uniform distribution over the prediction target space
        target_std = (prediction_high - prediction_low) / np.sqrt(12)
    else:
        target_std = None
    loss_fn = MSELossFn(target_std=target_std)
    if target_std is not None:
        loss_fn = loss_fn.normalized
    else:
        logger.warning(
            "Prediction target space is unbounded, and target_std is not provided. MSE loss will not be normalized."
        )
    return loss_fn, prediction_target_space


class ActiveRegressionEnv(
    ActivePerceptionEnv[ObsType, ActType, np.ndarray, np.ndarray],
    Generic[ObsType, ActType],
    ABC,
):
    def __init__(
        self,
        target_dim: int,
        inner_action_space: gym.Space[ActType],
        prediction_low: float | np.ndarray = -np.inf,
        prediction_high: float | np.ndarray = np.inf,
        target_std: float | None = None,
    ):
        prediction_space = gym.spaces.Box(
            low=prediction_low, high=prediction_high, shape=(target_dim,)
        )
        self.action_space = ActivePerceptionActionSpace(
            inner_action_space, prediction_space
        )
        self.loss_fn, self.prediction_target_space = _make_mse_loss_fn_and_target_space(
            target_dim, prediction_low, prediction_high, target_std
        )


class ActiveRegressionVectorEnv(
    ActivePerceptionVectorEnv[ObsType, ActType, np.ndarray, np.ndarray, np.ndarray],
    Generic[ObsType, ActType],
    ABC,
):
    def __init__(
        self,
        num_envs: int,
        target_dim: int,
        single_inner_action_space: gym.Space[ActType],
        prediction_low: float = -np.inf,
        prediction_high: float = np.inf,
        target_std: float | None = None,
    ):
        self.num_envs = num_envs
        single_prediction_space = gym.spaces.Box(
            low=prediction_low, high=prediction_high, shape=(target_dim,)
        )
        self.single_action_space = ActivePerceptionActionSpace(
            single_inner_action_space, single_prediction_space
        )
        self.action_space = gym.vector.utils.batch_space(
            self.single_action_space, num_envs
        )
        self.loss_fn, self.single_prediction_target_space = (
            _make_mse_loss_fn_and_target_space(
                target_dim, prediction_low, prediction_high, target_std
            )
        )
        self.prediction_target_space = gym.vector.utils.batch_space(
            self.single_prediction_target_space, num_envs
        )


class ActiveRegressionLogWrapper(
    ActivePerceptionWrapper[
        ObsType,
        ActType,
        np.ndarray,
        np.ndarray,
        ObsType,
        ActType,
        np.ndarray,
        np.ndarray,
    ],
    Generic[ObsType, ActType],
    ABC,
):
    def __init__(
        self, env: ActivePerceptionEnv[ObsType, ActType, np.ndarray, np.ndarray]
    ):
        super().__init__(env)
        self.__metrics: dict[str, deque[float]] | None = None

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any | None] = None
    ) -> tuple[ObsType, dict[str, Any]]:
        self.__metrics = defaultdict(deque)
        return super().reset(seed=seed, options=options)

    def step(
        self, action: FullActType[ActType, np.ndarray]
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)

        target = info["prediction"]["target"]
        prediction = action["prediction"]

        # Compute Euclidean distance and MSE
        euclidean_dist = np.linalg.norm(target - prediction)
        mse = np.mean((target - prediction) ** 2)

        self.__metrics["euclidean_distance"].append(euclidean_dist)
        self.__metrics["mse"].append(mse)

        if terminated or truncated:
            info = update_info_metrics(info, self.__metrics)

        return obs, reward, terminated, truncated, info


class ActiveRegressionVectorLogWrapper(
    ActivePerceptionVectorWrapper[
        ObsType,
        ActType,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        ObsType,
        ActType,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ],
    Generic[ObsType, ActType],
    ABC,
):
    def __init__(
        self,
        env: ActivePerceptionVectorWrapper[
            ObsType,
            ActType,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            ObsType,
            ActType,
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ],
    ):
        super().__init__(env)
        self.__prev_done = None
        self.__metrics: dict[str, tuple[deque[float], ...]] | None = None

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any | None] = None
    ) -> tuple[ObsType, dict[str, Any]]:
        self.__prev_done = np.zeros(self.num_envs, dtype=np.bool_)
        self.__metrics = defaultdict(
            lambda: tuple(deque() for _ in range(self.num_envs))
        )
        return super().reset(seed=seed, options=options)

    def step(
        self, action: FullActType[ActType, np.ndarray]
    ) -> tuple[ObsType, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)

        target = info["prediction"]["target"]
        prediction = action["prediction"]

        # Compute Euclidean distance and MSE
        euclidean_dist = np.linalg.norm(target - prediction, axis=-1)
        mse = np.mean((target - prediction) ** 2, axis=-1)

        for i in range(self.num_envs):
            if self.__prev_done[i]:
                self.__metrics["euclidean_distance"][i].clear()
                self.__metrics["mse"][i].clear()
            else:
                self.__metrics["euclidean_distance"][i].append(euclidean_dist[i])
                self.__metrics["mse"][i].append(mse[i])

        self.__prev_done = terminated | truncated
        if np.any(self.__prev_done):
            info = update_info_metrics_vec(info, self.__metrics, self.__prev_done)
        return obs, reward, terminated, truncated, info

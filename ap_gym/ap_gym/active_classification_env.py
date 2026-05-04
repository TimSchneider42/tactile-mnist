from __future__ import annotations

from abc import ABC
from collections import deque, defaultdict
from typing import Generic, Any, SupportsFloat

import gymnasium as gym
import numpy as np
import scipy

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
from .logit_space import LogitSpace
from .loss_fn import CrossEntropyLossFn
from .types import ObsType, ActType
from .util import update_info_metrics, update_info_metrics_vec


class ActiveClassificationEnv(
    ActivePerceptionEnv[ObsType, ActType, np.ndarray, int],
    Generic[ObsType, ActType],
    ABC,
):
    def __init__(self, num_classes: int, inner_action_space: gym.Space[ActType]):
        prediction_space = LogitSpace(-np.inf, np.inf, shape=(num_classes,))
        self.action_space = ActivePerceptionActionSpace(
            inner_action_space, prediction_space
        )
        self.prediction_target_space = gym.spaces.Discrete(num_classes)
        self.loss_fn = CrossEntropyLossFn(num_classes=num_classes).normalized


class ActiveClassificationVectorEnv(
    ActivePerceptionVectorEnv[ObsType, ActType, np.ndarray, np.ndarray, np.ndarray],
    Generic[ObsType, ActType],
    ABC,
):
    def __init__(
        self,
        num_envs: int,
        num_classes: int,
        single_inner_action_space: gym.Space[ActType],
    ):
        self.num_envs = num_envs
        single_prediction_space = LogitSpace(-np.inf, np.inf, shape=(num_classes,))
        self.single_action_space = ActivePerceptionActionSpace(
            single_inner_action_space, single_prediction_space
        )
        self.action_space = gym.vector.utils.batch_space(
            self.single_action_space, num_envs
        )
        self.single_prediction_target_space = gym.spaces.Discrete(num_classes)
        self.prediction_target_space = gym.spaces.MultiDiscrete(
            [num_classes] * num_envs
        )
        self.loss_fn = CrossEntropyLossFn(num_classes=num_classes).normalized


class ActiveClassificationLogWrapper(
    ActivePerceptionWrapper[
        ObsType, ActType, np.ndarray, int, ObsType, ActType, np.ndarray, int
    ],
    Generic[ObsType, ActType],
    ABC,
):
    def __init__(self, env: ActivePerceptionEnv[ObsType, ActType, np.ndarray, int]):
        super().__init__(env)
        self.__metrics: dict[str, deque[float] | np.ndarray] | None = None

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any | None] = None
    ) -> tuple[ObsType, dict[str, Any]]:
        self.__metrics = defaultdict(deque)
        return super().reset(seed=seed, options=options)

    def step(
        self, action: FullActType[ActType, PredType]
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)
        self.__metrics["correct_label_prob"].append(
            float(
                scipy.special.softmax(action["prediction"])[
                    info["prediction"]["target"]
                ]
            )
        )
        done = terminated or truncated
        if done:
            num_classes = self.prediction_target_space.n
            correct_label_prob = np.array(
                self.__metrics["correct_label_prob"], dtype=np.float32
            )
            is_correct = correct_label_prob > 1 / num_classes
            self.__metrics["accuracy"] = is_correct.astype(np.float32)
            info = update_info_metrics(info, self.__metrics)
            first_correct_candidates = np.where(is_correct)[0]
            if len(first_correct_candidates) > 0:
                info["stats"]["scalar"]["first_correct"] = first_correct_candidates[0]
            last_incorrect_candidates = np.where(~is_correct)[-1]
            if len(last_incorrect_candidates) > 0:
                info["stats"]["scalar"]["last_incorrect"] = last_incorrect_candidates[
                    -1
                ]
        return obs, reward, terminated, truncated, info


class ActiveClassificationVectorLogWrapper(
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
        env: ActivePerceptionVectorEnv[ObsType, ActType, np.ndarray, np.ndarray],
    ):
        super().__init__(env)
        self.__prev_done = None
        self.__metrics: dict[str, tuple[deque[float] | np.ndarray, ...]] | None = None

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any | None] = None
    ) -> tuple[ObsType, dict[str, Any]]:
        self.__prev_done = np.zeros(self.num_envs, dtype=np.bool_)
        self.__metrics = defaultdict(
            lambda: tuple(deque() for _ in range(self.num_envs))
        )
        return super().reset(seed=seed, options=options)

    def step(
        self, action: FullActType[ActType, PredType]
    ) -> tuple[ObsType, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)
        for i in range(self.num_envs):
            if self.__prev_done[i]:
                self.__metrics["correct_label_prob"][i].clear()
            else:
                self.__metrics["correct_label_prob"][i].append(
                    scipy.special.softmax(action["prediction"][i])[
                        info["prediction"]["target"][i]
                    ]
                )

        self.__prev_done = terminated | truncated
        if np.any(self.__prev_done):
            num_classes = self.single_prediction_target_space.n
            correct_label_prob = [
                np.array(e, dtype=np.float32)
                for e in self.__metrics["correct_label_prob"]
            ]
            is_correct = [e > 1 / num_classes for e in correct_label_prob]
            self.__metrics["accuracy"] = tuple(e.astype(np.float32) for e in is_correct)

            info = update_info_metrics_vec(info, self.__metrics, self.__prev_done)

            del self.__metrics["accuracy"]
            first_correct = np.full(self.num_envs, -1, dtype=np.int32)
            first_correct_valid = np.zeros(self.num_envs, dtype=np.bool_)
            last_incorrect = np.full(self.num_envs, -1, dtype=np.int32)
            last_incorrect_valid = np.zeros(self.num_envs, dtype=np.bool_)
            for i in range(self.num_envs):
                first_correct_candidates = np.where(is_correct[i])[0]
                if len(first_correct_candidates) > 0:
                    first_correct[i] = first_correct_candidates[0]
                    first_correct_valid[i] = True
                last_incorrect_candidates = np.where(~is_correct[i])[0]
                if len(last_incorrect_candidates) > 0:
                    last_incorrect[i] = last_incorrect_candidates[-1]
                    last_incorrect_valid[i] = True
            info["stats"]["scalar"].update(
                {
                    "first_correct": first_correct,
                    "_first_correct": first_correct_valid,
                    "last_incorrect": last_incorrect,
                    "_last_incorrect": last_incorrect_valid,
                }
            )
        return obs, reward, terminated, truncated, info

from copy import deepcopy
from typing import Any, Literal

import gymnasium as gym
import numpy as np
from gymnasium.vector.utils import batch_space

from ap_gym import (
    ActivePerceptionVectorWrapper,
    ZeroLossFn,
    ActivePerceptionActionSpace,
)
from ap_gym.envs.image import CircleSquareDataset
from ap_gym.envs.image_classification import ImageClassificationVectorEnv

ObsType = dict[Literal["glimpse", "glimpse_pos", "time_step"], np.ndarray]


class CircleSquareHideAndSeekVectorWrapper(
    ActivePerceptionVectorWrapper[
        ObsType,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        ObsType,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]
):
    def __init__(
        self, env: ImageClassificationVectorEnv, mask_prediction: bool = False
    ):
        assert isinstance(env.config.dataset, CircleSquareDataset)
        self.__dataset: CircleSquareDataset = env.config.dataset
        super().__init__(env)
        self.__mask_prediction = mask_prediction
        if self.__mask_prediction:
            self._loss_fn = ZeroLossFn()
            self._single_prediction_target_space = gym.spaces.Tuple(())
            self._prediction_target_space = batch_space(
                self._single_prediction_target_space, env.num_envs
            )
            self._single_action_space = ActivePerceptionActionSpace(
                inner_action_space=env.single_inner_action_space,
                prediction_space=gym.spaces.Tuple(()),
            )
            self._action_space = batch_space(self._single_action_space, env.num_envs)

    def reset(
        self,
        *,
        seed: int | list[int] | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        info = deepcopy(info)
        if self.__mask_prediction:
            info["prediction"]["target"] = ()
        return obs, info

    def step(self, actions: np.ndarray) -> tuple[
        ObsType,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        dict[str, Any],
    ]:
        if self.__mask_prediction:
            actions = {
                "action": actions["action"],
                "prediction": np.zeros(self.env.prediction_space.shape),
            }
        obs, reward, terminated, truncated, info = self.env.step(actions)
        info = deepcopy(info)
        indices = info["index"]
        positions, labels = self.__dataset.get_object_position_and_label(indices)
        sign = labels * 2 - 1
        positions_norm = (
            self.env.image_perception_module.normalize_coords(
                np.flip(positions, axis=-1)
            )
            - 1
        )
        distances = np.linalg.norm(obs["glimpse_pos"] - positions_norm, axis=-1)
        additional_reward = sign * distances
        info["base_reward"] += additional_reward
        if self.__mask_prediction:
            info["prediction"]["target"] = ()
            reward = info["base_reward"]
        else:
            reward += additional_reward
        return obs, reward, terminated, truncated, info

    @property
    def config(self):
        return self.env.config

    @property
    def spec(self):
        return self.env.spec

    @spec.setter
    def spec(self, value):
        self.env.spec = value

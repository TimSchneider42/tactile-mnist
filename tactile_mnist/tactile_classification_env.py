from __future__ import annotations

from itertools import chain
from typing import (
    Literal,
    TYPE_CHECKING,
)

import gymnasium as gym
import numpy as np

from ap_gym import (
    ActivePerceptionVectorToSingleWrapper,
    CrossEntropyLossFn,
)
from tactile_mnist import (
    MeshDataset,
)
from .tactile_perception_vector_env import (
    TactilePerceptionVectorEnv,
    TactilePerceptionConfig,
)

if TYPE_CHECKING:
    from .tactile_perception_vector_env import ObsType, ActType


class TactileClassificationVectorEnv(
    TactilePerceptionVectorEnv[np.ndarray, np.ndarray],
):
    def __init__(
        self,
        config: TactilePerceptionConfig,
        num_envs: int,
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
    ):
        if isinstance(config.dataset, MeshDataset):
            datasets = [config.dataset] * num_envs
        else:
            assert len(config.dataset) == num_envs
            datasets = config.dataset
        self.__label_map = {
            i: l
            for i, l in enumerate(sorted(set(chain(*(ds.labels for ds in datasets)))))
        }
        self.__inverse_label_map = {l: i for i, l in self.__label_map.items()}
        all_labels = {l for ds in datasets for l in ds.labels}

        super().__init__(
            config,
            num_envs,
            single_prediction_space=gym.spaces.Box(
                -np.inf, np.inf, shape=(len(all_labels),)
            ),
            single_prediction_target_space=gym.spaces.Discrete(len(all_labels)),
            loss_fn=CrossEntropyLossFn(),
            render_mode=render_mode,
        )

    def _get_prediction_targets(self) -> np.ndarray:
        return np.array(
            [
                self.__inverse_label_map[dp.metadata.label]
                for dp in self.current_data_points
            ]
        )


def TactileClassificationEnv(
    config: TactilePerceptionConfig,
    render_mode: Literal["rgb_array", "human"] = "rgb_array",
) -> ActivePerceptionVectorToSingleWrapper[
    "ObsType", "ActType", np.ndarray, np.ndarray
]:
    return ActivePerceptionVectorToSingleWrapper(
        TactileClassificationVectorEnv(
            config,
            1,
            render_mode=render_mode,
        )
    )

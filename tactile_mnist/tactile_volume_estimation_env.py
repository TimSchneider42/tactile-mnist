from __future__ import annotations

import functools
from collections import deque, defaultdict
from functools import partial
from typing import (
    Literal,
    TYPE_CHECKING,
    Any,
)

import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation
from transformation import Transformation

from ap_gym import (
    ActivePerceptionVectorToSingleWrapper,
    MSELossFn,
)
from ap_gym.util import update_info_metrics_vec
from tactile_mnist import MeshDataPoint
from .tactile_perception_vector_env import (
    TactilePerceptionVectorEnv,
    TactilePerceptionConfig,
    ActType,
)

if TYPE_CHECKING:
    from .tactile_perception_vector_env import ObsType


class TactileVolumeEstimationVectorEnv(
    TactilePerceptionVectorEnv[np.ndarray, np.ndarray],
):
    def __init__(
        self,
        config: TactilePerceptionConfig,
        num_envs: int,
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        renderer_show_shadow_objects: bool = True,
    ):
        self.__compute_object_volume_cached = functools.lru_cache(maxsize=num_envs)(
            self.__compute_object_volume
        )

        super().__init__(
            config,
            num_envs,
            single_prediction_space=gym.spaces.Box(-np.inf, np.inf, shape=(1,)),
            single_prediction_target_space=gym.spaces.Box(-np.inf, np.inf, shape=(1,)),
            loss_fn=MSELossFn(),
            render_mode=render_mode,
        )
        self.__renderer_show_shadow_objects = renderer_show_shadow_objects

    def _step(
        self,
        action: dict[str, np.ndarray],
        prediction: np.ndarray,
    ):
        obs, action_reward, terminated, truncated, info, labels = super()._step(
            action, prediction
        )

        if self.__renderer_show_shadow_objects:
            # Do that after the step as new objects might be loaded
            self._renderer.update_shadow_objects(
                self.current_object_poses_platform_frame,
                new_shadow_object_scales=np.maximum(prediction, 0)
                / np.maximum(labels, 1e-4),
                shadow_object_visible=~np.array(self._prev_done),
            )

        return obs, action_reward, terminated, truncated, info, labels

    @staticmethod
    def __compute_object_volume(
        dp: MeshDataPoint,
    ) -> float:
        return dp.mesh.volume * 100**3  # Convert m^3 to cm^3

    def _get_prediction_targets(self) -> np.ndarray:
        return np.array(
            [
                self.__compute_object_volume_cached(dp)
                for dp in self.current_data_points
            ],
            dtype=np.float32,
        )[..., None]


def TactileVolumeEstimationEnv(
    config: TactilePerceptionConfig,
    render_mode: Literal["rgb_array", "human"] = "rgb_array",
    renderer_show_shadow_objects: bool = True,
) -> ActivePerceptionVectorToSingleWrapper["ObsType", ActType, np.ndarray, np.ndarray]:
    return ActivePerceptionVectorToSingleWrapper(
        TactileVolumeEstimationVectorEnv(
            config,
            1,
            render_mode=render_mode,
            renderer_show_shadow_objects=renderer_show_shadow_objects,
        )
    )

from __future__ import annotations

import functools
import logging
from collections import deque, defaultdict
from typing import (
    Literal,
    TYPE_CHECKING,
    Any,
)

import gymnasium as gym
import numpy as np

from ap_gym import (
    ActivePerceptionVectorToSingleWrapper,
    MSELossFn,
)
from ap_gym.util import update_info_metrics_vec
from .mesh_dataset import MeshDataPoint, MeshDataset
from .simple_mesh_dataset import SimpleMeshDataset
from .tactile_perception_vector_env import (
    TactilePerceptionVectorEnv,
    TactilePerceptionConfig,
    ActType,
)
from .util import get_dataset_stats

if TYPE_CHECKING:
    from .tactile_perception_vector_env import ObsType

logger = logging.getLogger(__name__)


def _compute_object_volume_idx(idx: int, ds: MeshDataset):
    return {"volume": SimpleMeshDataset(ds)[idx].mesh.volume}


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

        statistics = get_dataset_stats(
            config.dataset, "volume", _compute_object_volume_idx
        )
        self.__mean_volume = statistics["volume"]["mean"]
        self.__std_volume = statistics["volume"]["std"]
        self.__min_volume = statistics["volume"]["min"]
        self.__max_volume = statistics["volume"]["max"]

        pred_space = gym.spaces.Box(
            self.__normalize_volume(self.__min_volume),
            self.__normalize_volume(self.__max_volume),
            shape=(1,),
        )

        super().__init__(
            config,
            num_envs,
            single_prediction_space=pred_space,
            single_prediction_target_space=pred_space,
            loss_fn=MSELossFn(target_std=1.0).normalized,
            render_mode=render_mode,
        )

        self.__renderer_show_shadow_objects = renderer_show_shadow_objects
        self.__metrics: dict[str, tuple[deque[float], ...]] | None = None

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any | None] = None
    ) -> tuple[ObsType, dict[str, Any]]:
        self.__metrics = defaultdict(
            lambda: tuple(deque() for _ in range(self.num_envs))
        )
        return super().reset(seed=seed, options=options)

    def _step(
        self,
        action: dict[str, np.ndarray],
        prediction: np.ndarray,
    ):
        target_volume = self.__get_object_volumes()
        predicted_volume = prediction * self.__std_volume + self.__mean_volume
        relative_error = np.maximum(predicted_volume, 0) / np.maximum(
            target_volume, 1e-10
        )
        abs_error = np.abs(predicted_volume - target_volume)

        for i in range(self.num_envs):
            if self._prev_done[i]:
                self.__metrics["abs_error_cm3"][i].clear()
                self.__metrics["rel_error"][i].clear()
            else:
                self.__metrics["abs_error_cm3"][i].append(abs_error[i] * 100**3)
                self.__metrics["rel_error"][i].append(relative_error[i])

        obs, action_reward, terminated, truncated, info, labels = super()._step(
            action, prediction
        )

        if self.__renderer_show_shadow_objects:
            # Do that after the step as new objects might be loaded
            self._renderer.update_shadow_objects(
                self.current_object_poses_platform_frame,
                new_shadow_object_scales=relative_error,
                shadow_object_visible=~np.array(self._prev_done),
            )

        if np.any(terminated | truncated):
            info = update_info_metrics_vec(info, self.__metrics, terminated | truncated)

        return obs, action_reward, terminated, truncated, info, labels

    @staticmethod
    def __compute_object_volume(dp: MeshDataPoint) -> float:
        return dp.mesh.volume

    def __normalize_volume(self, volume: np.ndarray | float) -> np.ndarray | float:
        return (volume - self.__mean_volume) / self.__std_volume

    def _get_prediction_targets(self) -> np.ndarray:
        return self.__normalize_volume(self.__get_object_volumes())

    def __get_object_volumes(self) -> np.ndarray:
        return np.array(
            [self.__compute_object_volume(dp) for dp in self.current_data_points],
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

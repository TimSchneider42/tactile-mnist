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
    HuberLossFn,
    LossFn,
)
from ap_gym.util import update_info_metrics_vec
from .mesh_dataset import MeshDataPoint, MeshDataset
from .tactile_perception_vector_env import (
    TactilePerceptionVectorEnv,
    TactilePerceptionConfig,
    ActType,
)
from .util import get_dataset_stats

if TYPE_CHECKING:
    from .tactile_perception_vector_env import ObsType

logger = logging.getLogger(__name__)


# Error (in standard deviations of the volume) beyond which the Huber loss of the volume prediction grows linearly
DEFAULT_HUBER_DELTA = 1.0


def _compute_object_volume_idx(idx: int, ds: MeshDataset):
    return {"volume": ds[idx].volume}


def get_object_volumes(dataset: MeshDataset) -> np.ndarray:
    """Volumes of all objects of the dataset (in dataset order), cached."""
    return get_dataset_stats(dataset, "volume", _compute_object_volume_idx)["volume"]


def get_volume_statistics(dataset: MeshDataset) -> dict[str, float]:
    volumes = get_object_volumes(dataset)
    return {
        "mean": float(np.mean(volumes)),
        "std": float(np.std(volumes)),
        "min": float(np.min(volumes)),
        "max": float(np.max(volumes)),
    }


def mk_volume_loss_fn(
    statistics_dataset: MeshDataset, statistics: dict[str, float], delta: float
) -> LossFn:
    """
    Huber loss of the standardized volume prediction, normalized such that the best blind guess on statistics_dataset
    scores 1. The Huber loss bounds the gradient of every sample, which keeps the objects in the heavy right tail of the
    volume distribution from dominating the loss.
    """
    targets = (
        get_object_volumes(statistics_dataset) - statistics["mean"]
    ) / statistics["std"]
    return HuberLossFn.from_data(delta, targets).normalized


class TactileVolumeEstimationVectorEnv(
    TactilePerceptionVectorEnv[np.ndarray, np.ndarray],
):
    def __init__(
        self,
        config: TactilePerceptionConfig,
        num_envs: int,
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        renderer_show_shadow_objects: bool = True,
        huber_delta: float = DEFAULT_HUBER_DELTA,
        volume_statistics_dataset: MeshDataset | None = None,
    ):
        """
        :param huber_delta: Error (in standard deviations of the volume) beyond which the Huber loss of the volume
            prediction grows linearly (see mk_volume_loss_fn).
        :param volume_statistics_dataset: Dataset whose volume statistics standardize the prediction targets and
            normalize the loss. Defaults to config.dataset. The registered evaluation splits pass the training split
            here, so that a policy sees the same target scale in training and evaluation.
        """
        self.__compute_object_volume_cached = functools.lru_cache(maxsize=num_envs)(
            self.__compute_object_volume
        )

        if volume_statistics_dataset is None:
            volume_statistics_dataset = config.dataset
        statistics = get_volume_statistics(volume_statistics_dataset)
        self.__mean_volume = statistics["mean"]
        self.__std_volume = statistics["std"]
        self.__min_volume = statistics["min"]
        self.__max_volume = statistics["max"]
        loss_fn = mk_volume_loss_fn(volume_statistics_dataset, statistics, huber_delta)

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
            loss_fn=loss_fn,
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
        return dp.volume

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
    huber_delta: float = DEFAULT_HUBER_DELTA,
    volume_statistics_dataset: MeshDataset | None = None,
) -> ActivePerceptionVectorToSingleWrapper["ObsType", ActType, np.ndarray, np.ndarray]:
    return ActivePerceptionVectorToSingleWrapper(
        TactileVolumeEstimationVectorEnv(
            config,
            1,
            render_mode=render_mode,
            renderer_show_shadow_objects=renderer_show_shadow_objects,
            huber_delta=huber_delta,
            volume_statistics_dataset=volume_statistics_dataset,
        )
    )

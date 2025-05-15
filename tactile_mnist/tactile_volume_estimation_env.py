from __future__ import annotations

import functools
import json
from collections import deque, defaultdict
from typing import (
    Literal,
    TYPE_CHECKING,
    Any,
)

import filelock
import gymnasium as gym
import numpy as np
import tqdm

from ap_gym import (
    ActivePerceptionVectorToSingleWrapper,
    MSELossFn,
)
from ap_gym.util import update_info_metrics_vec
from tactile_mnist import MeshDataPoint, CACHE_BASE_DIR
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

        cache_dir = CACHE_BASE_DIR / "volume_estimation"
        cache_dir.mkdir(parents=True, exist_ok=True)
        ds_fingerprint = config.dataset.huggingface_dataset._fingerprint
        cache_file = cache_dir / f"{ds_fingerprint}.json"
        with filelock.FileLock(cache_dir / f"{ds_fingerprint}.lock"):
            if cache_file.exists():
                with cache_file.open() as f:
                    data = json.load(f)
                self.__mean_volume = data["mean_volume"]
                self.__std_volume = data["std_volume"]
            else:
                volumes = np.zeros(len(config.dataset))
                print(
                    "Computing object volumes for normalization (the results will be cached)..."
                )
                for i, dp in tqdm.tqdm(
                    enumerate(config.dataset), total=len(config.dataset)
                ):
                    volumes[i] = dp.mesh.volume
                self.__mean_volume = np.mean(volumes)
                self.__std_volume = np.std(volumes)
                with cache_file.open("w") as f:
                    json.dump(
                        {
                            "mean_volume": self.__mean_volume,
                            "std_volume": self.__std_volume,
                        },
                        f,
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

    def _get_prediction_targets(self) -> np.ndarray:
        return (self.__get_object_volumes() - self.__mean_volume) / self.__std_volume

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

from __future__ import annotations

import functools
from typing import (
    Literal,
)

import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation
from transformation import Transformation

from ap_gym import (
    ActivePerceptionVectorToSingleWrapper,
    MSELossFn,
)
from .tactile_perception_vector_env import (
    TactilePerceptionVectorEnv,
    TactilePerceptionConfig,
)


class TactilePoseEstimationVectorEnv(
    TactilePerceptionVectorEnv[np.ndarray, np.ndarray],
):
    def __init__(
        self,
        config: TactilePerceptionConfig,
        num_envs: int,
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
    ):
        self.__compute_object_frame_cached = functools.lru_cache(maxsize=num_envs)(
            self.__compute_object_frame
        )

        super().__init__(
            config,
            num_envs,
            single_prediction_space=gym.spaces.Box(-np.inf, np.inf, shape=(4,)),
            single_prediction_target_space=gym.spaces.Box(-np.inf, np.inf, shape=(4,)),
            loss_fn=MSELossFn(),
            render_mode=render_mode,
        )

    def _step(
        self,
        action: dict[str, np.ndarray],
        prediction: np.ndarray,
    ):
        prev_done = np.array(self._prev_done)

        object_frame_pos_2d = prediction[..., :2] * np.array(self.config.cell_size) / 2
        object_frames = Transformation.batch_concatenate(
            [self.__compute_object_frame_cached(dp) for dp in self.current_data_points]
        )
        object_frames_world = self.current_object_poses_platform_frame * object_frames
        object_frame_pos = np.concatenate(
            [object_frame_pos_2d, object_frames_world.translation[..., 2:3]], axis=-1
        )
        x = prediction[..., 2].copy()
        y = prediction[..., 3].copy()
        x[np.abs(x) < 1e-5] = 1e-5
        y[np.abs(y) < 1e-5] = 1e-5
        object_frame_rot_angle = np.arctan2(y, x)
        object_frame_rot = Rotation.from_euler(
            "xyz",
            np.concatenate(
                [
                    np.zeros((prediction.shape[0], 2)),
                    object_frame_rot_angle[..., np.newaxis],
                ],
                axis=-1,
            ),
        )

        object_frame_pose = Transformation(object_frame_pos, object_frame_rot)

        res = super()._step(action, prediction)

        # Do that after the step as new objects might be loaded
        self._renderer.update_shadow_objects(
            object_frame_pose,
            shadow_object_visible=~prev_done,
        )

        return res

    @staticmethod
    def __compute_object_frame(dp) -> Transformation:
        # TODO
        return Transformation()

    def _get_prediction_targets(self) -> np.ndarray:
        object_frames = Transformation.batch_concatenate(
            [self.__compute_object_frame_cached(dp) for dp in self.current_data_points]
        )
        object_frames_world = self.current_object_poses_platform_frame * object_frames
        object_position_2d = object_frames_world.translation[..., :2]
        object_x_axis_2d = object_frames_world.rotation.as_matrix()[..., :2, 0]
        object_x_axis_2d = object_x_axis_2d / np.linalg.norm(
            object_x_axis_2d, axis=-1, keepdims=True
        )
        object_position_2d_norm = object_position_2d / (
            np.array(self.config.cell_size, dtype=np.float32) / 2
        )
        return np.concatenate([object_position_2d_norm, object_x_axis_2d], axis=-1)


def TactilePoseEstimationEnv(
    config: TactilePerceptionConfig,
    render_mode: Literal["rgb_array", "human"] = "rgb_array",
) -> ActivePerceptionVectorToSingleWrapper[
    "ObsType", "ActType", np.ndarray, np.ndarray
]:
    return ActivePerceptionVectorToSingleWrapper(
        TactilePoseEstimationVectorEnv(
            config,
            1,
            render_mode=render_mode,
        )
    )

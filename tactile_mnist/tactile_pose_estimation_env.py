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


class TactilePoseEstimationVectorEnv(
    TactilePerceptionVectorEnv[np.ndarray, np.ndarray],
):
    def __init__(
        self,
        config: TactilePerceptionConfig,
        num_envs: int,
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        frame_position_mode: Literal["model", "inertia_frame"] = "model",
        frame_rotation_mode: Literal["model", "inertia_frame"] = "model",
        renderer_show_shadow_objects: bool = True,
    ):
        self.__compute_object_frame_cached = functools.lru_cache(maxsize=num_envs)(
            partial(
                self.__compute_object_frame,
                position_mode=frame_position_mode,
                rotation_mode=frame_rotation_mode,
            )
        )

        super().__init__(
            config,
            num_envs,
            single_prediction_space=gym.spaces.Box(-np.inf, np.inf, shape=(4,)),
            single_prediction_target_space=gym.spaces.Box(-np.inf, np.inf, shape=(4,)),
            loss_fn=MSELossFn(),
            render_mode=render_mode,
        )
        self.__metrics: dict[str, tuple[deque[float], ...]] | None = None
        self.__renderer_show_shadow_objects = renderer_show_shadow_objects

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
        prev_done = np.array(self._prev_done)

        object_frames = Transformation.batch_concatenate(
            [self.__compute_object_frame_cached(dp) for dp in self.current_data_points]
        )
        pred_object_frame_pos_2d = (
            prediction[..., :2] * np.array(self.config.cell_size) / 2
        )
        actual_object_frames_world = (
            self.current_object_poses_platform_frame * object_frames
        )
        pred_object_frame_pos = np.concatenate(
            [
                pred_object_frame_pos_2d,
                actual_object_frames_world.translation[..., 2:3],
            ],
            axis=-1,
        )
        x = prediction[..., 2].copy()
        y = prediction[..., 3].copy()
        x[np.abs(x) < 1e-5] = 1e-5
        y[np.abs(y) < 1e-5] = 1e-5
        pred_object_frame_rot_angle = np.arctan2(y, x)
        pred_object_frame_rot = Rotation.from_euler(
            "xyz",
            np.concatenate(
                [
                    np.zeros((prediction.shape[0], 2)),
                    pred_object_frame_rot_angle[..., np.newaxis],
                ],
                axis=-1,
            ),
        )

        # Compute actual object frame pose
        actual_object_frame_pos_2d = actual_object_frames_world.translation[..., :2]
        actual_object_x_axis_2d = actual_object_frames_world.rotation.as_matrix()[
            ..., :2, 0
        ]
        pred_object_x_axis_2d = np.stack([x, y], axis=-1)

        linear_error = np.linalg.norm(
            actual_object_frame_pos_2d - pred_object_frame_pos_2d, axis=-1
        )
        angular_error = np.arccos(
            np.clip(
                (actual_object_x_axis_2d * pred_object_x_axis_2d).sum(-1)
                / (
                    np.linalg.norm(actual_object_x_axis_2d, axis=-1)
                    * np.linalg.norm(pred_object_x_axis_2d, axis=-1)
                ),
                -1,
                1,
            )
        )

        for i in range(self.num_envs):
            if prev_done[i]:
                self.__metrics["linear_error"][i].clear()
                self.__metrics["angular_error"][i].clear()
            else:
                self.__metrics["linear_error"][i].append(linear_error[i])
                self.__metrics["angular_error"][i].append(angular_error[i])

        obs, action_reward, terminated, truncated, info, labels = super()._step(
            action, prediction
        )

        if self.__renderer_show_shadow_objects:
            # Do that after the step as new objects might be loaded
            self._renderer.update_shadow_objects(
                Transformation(pred_object_frame_pos, pred_object_frame_rot),
                shadow_object_visible=~prev_done,
            )

        if np.any(terminated | truncated):
            info = update_info_metrics_vec(info, self.__metrics, terminated | truncated)

        return obs, action_reward, terminated, truncated, info, labels

    @staticmethod
    def __compute_object_frame(
        dp: MeshDataPoint,
        position_mode: Literal["model", "center_of_mass"],
        rotation_mode: Literal["model"],
    ) -> Transformation:
        if position_mode == "model":
            position = np.zeros(3, dtype=np.float32)
        elif position_mode == "inertia_frame":
            position = dp.mesh.center_mass
        else:
            raise ValueError(f"Unknown frame position mode: {position_mode}")
        if rotation_mode == "model":
            rotation = Rotation.identity()
        elif rotation_mode == "inertia_frame":
            inertia_tensor = dp.mesh.moment_inertia
            # Determine eigenvalues and eigenvectors of the X and Y components of the inertia tensor
            inertia_components2d, inertia_vectors2d = np.linalg.eigh(
                inertia_tensor[:2, :2]
            )
            # Axis in the XY plane along which the mass is most closely distributed
            x_axis_idx = np.argmin(inertia_components2d)
            x_axis_2d = inertia_vectors2d[x_axis_idx]

            # The inertia frame is not unique in orientation, as it might be flipped 180 degrees. We always let the
            # X-axis point towards the direction in which the object is the longest.
            y_axis_2d = np.array([-x_axis_2d[1], x_axis_2d[0]], dtype=np.float32)
            vertices_proj = dp.mesh.vertices[:, :2] @ y_axis_2d
            max_val = np.max(vertices_proj)
            min_val = np.min(vertices_proj)
            if np.abs(min_val) > np.abs(max_val):
                x_axis_2d = -x_axis_2d
                y_axis_2d = -y_axis_2d

            x_axis = np.concatenate([x_axis_2d, [0.0]], dtype=np.float32)
            y_axis = np.concatenate([y_axis_2d, [0.0]], dtype=np.float32)
            z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            rotation = Rotation.from_matrix(np.stack([x_axis, y_axis, z_axis], axis=1))
        else:
            raise ValueError(f"Unknown frame rotation mode: {rotation_mode}")
        return Transformation(position, rotation)

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
    frame_position_mode: Literal["model", "inertia_frame"] = "model",
    frame_rotation_mode: Literal["model", "inertia_frame"] = "model",
    renderer_show_shadow_objects: bool = True,
) -> ActivePerceptionVectorToSingleWrapper["ObsType", ActType, np.ndarray, np.ndarray]:
    return ActivePerceptionVectorToSingleWrapper(
        TactilePoseEstimationVectorEnv(
            config,
            1,
            render_mode=render_mode,
            frame_position_mode=frame_position_mode,
            frame_rotation_mode=frame_rotation_mode,
            renderer_show_shadow_objects=renderer_show_shadow_objects,
        )
    )

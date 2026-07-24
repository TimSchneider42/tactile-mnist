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
from transformation import Transformation

from ap_gym import (
    ActivePerceptionVectorToSingleWrapper,
    MSELossFn,
)
from ap_gym.util import update_info_metrics_vec
from .mesh_dataset import MeshDataset
from .mesh_laplacian import (
    SpectralShapeRepresentation,
    compute_spectral_representation,
)
from .simple_mesh_dataset import SimpleMeshDataset
from .tactile_perception_vector_env import (
    TactilePerceptionVectorEnv,
    TactilePerceptionConfig,
    ActType,
    GenericMeshDataPoint,
)
from .util import get_dataset_stats

if TYPE_CHECKING:
    from .tactile_perception_vector_env import ObsType

logger = logging.getLogger(__name__)


def _compute_spectral_target_stats_idx(
    idx: int,
    ds: MeshDataset,
    num_coefficients: int,
    num_pose_samples: int,
    randomize_initial_object_pose: bool,
    max_initial_angle_perturbation: float,
    cell_size: tuple[float, float],
    smallest_dimension_up: bool,
) -> dict[str, float]:
    dp = TactilePerceptionVectorEnv._pre_process_dp(
        SimpleMeshDataset(ds)[idx], smallest_dimension_up=smallest_dimension_up
    )
    representation = compute_spectral_representation(dp.mesh, num_coefficients)
    rng = np.random.default_rng(idx)
    poses = TactilePerceptionVectorEnv._get_random_object_pose_batch(
        dp,
        randomize_initial_object_pose,
        max_initial_angle_perturbation,
        cell_size,
        rotation_perturbation_norm=rng.uniform(size=num_pose_samples),
        translation_perturbation_norm=rng.uniform(size=(num_pose_samples, 2)),
    )
    targets = np.stack(
        [representation.transform_coefficients(pose).reshape(-1) for pose in poses]
    )
    output = {}
    for d in range(targets.shape[1]):
        output[f"mean_{d}"] = float(np.mean(targets[:, d]))
        output[f"var_{d}"] = float(np.var(targets[:, d]))
        output[f"min_{d}"] = float(np.min(targets[:, d]))
        output[f"max_{d}"] = float(np.max(targets[:, d]))
    return output


class TactileShapeReconstructionVectorEnv(
    TactilePerceptionVectorEnv[np.ndarray, np.ndarray],
):
    """
    Active tactile shape reconstruction environment.

    The agent has to reconstruct the geometry of the touched object by regressing to its truncated Laplace-Beltrami
    spectral representation: the coefficients of the object's vertex positions w.r.t. the first num_coefficients
    eigenvectors of the cotangent Laplace-Beltrami operator of its mesh. The vertex positions are expressed in the
    platform frame, so the target jointly encodes the object's shape and its current pose. This joint representation
    remains well-defined even for (rotationally) symmetric objects, for which pose alone would be ambiguous.
    Predictions can be decoded back into a mesh via the eigenbasis, which is used to render the current prediction as
    a shadow object.
    """

    def __init__(
        self,
        config: TactilePerceptionConfig,
        num_envs: int,
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        num_coefficients: int = 64,
        renderer_show_shadow_objects: bool = True,
    ):
        self.__num_coefficients = num_coefficients
        self.__compute_spectral_representation_cached = functools.lru_cache(
            maxsize=num_envs
        )(self.__compute_spectral_representation)

        kwargs = {
            "num_coefficients": num_coefficients,
            "num_pose_samples": 32,
            "randomize_initial_object_pose": config.randomize_initial_object_pose,
            "max_initial_angle_perturbation": float(
                config.max_initial_angle_perturbation
            ),
            "cell_size": tuple(map(float, config.cell_size)),
            "smallest_dimension_up": config.smallest_dimension_up,
        }
        statistics = get_dataset_stats(
            config.dataset,
            "laplacian_spectral_coefficients",
            _compute_spectral_target_stats_idx,
            kwargs,
        )

        dims = 3 * num_coefficients
        self.__target_mean = np.array(
            [statistics[f"mean_{d}"]["mean"] for d in range(dims)]
        )
        # Total variance = variance of the per-object means + mean of the per-object (pose-induced) variances
        target_var = np.array(
            [
                statistics[f"mean_{d}"]["std"] ** 2 + statistics[f"var_{d}"]["mean"]
                for d in range(dims)
            ]
        )
        self.__target_std = np.sqrt(
            np.maximum(target_var, max(np.max(target_var) * 1e-12, 1e-24))
        )
        target_min = np.array([statistics[f"min_{d}"]["min"] for d in range(dims)])
        target_max = np.array([statistics[f"max_{d}"]["max"] for d in range(dims)])

        low = self.__normalize_targets(target_min)
        high = self.__normalize_targets(target_max)
        # Loosen the bounds slightly to account for object pose perturbations during the episode
        margin = 0.05 * (high - low) + 1e-3
        pred_space = gym.spaces.Box(
            (low - margin).astype(np.float32),
            (high + margin).astype(np.float32),
            shape=(dims,),
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

    @staticmethod
    def _decompose_coefficient_error(
        representation: SpectralShapeRepresentation,
        predicted_coefficients: np.ndarray,
        target_coefficients: np.ndarray,
    ) -> dict[str, float]:
        """
        Decompose the error between predicted and target spectral coefficients into a position, a rotation, and a
        residual shape component.

        The position component is the distance between the centroids of the two reconstructions. The rotation
        component is the Z-rotation that optimally aligns the centered predicted reconstruction with the centered
        target reconstruction (2D orthogonal Procrustes). The shape component is the mass-weighted RMS error remaining
        after removing both.

        :param representation: Spectral representation of the object.
        :param predicted_coefficients: Predicted spectral coefficients of shape (num_coefficients, 3).
        :param target_coefficients: Target spectral coefficients of shape (num_coefficients, 3).
        :return: Dictionary containing the reconstruction, position, rotation, and shape errors.
        """
        total_mass = representation.total_mass
        constant_projection = representation.constant_projection

        reconstruction_error = np.linalg.norm(
            predicted_coefficients - target_coefficients
        ) / np.sqrt(total_mass)

        # The centroid of a reconstruction can be read off the coefficients directly
        predicted_centroid = constant_projection @ predicted_coefficients / total_mass
        target_centroid = constant_projection @ target_coefficients / total_mass
        position_error = np.linalg.norm(predicted_centroid - target_centroid)

        predicted_centered = predicted_coefficients - np.outer(
            constant_projection, predicted_centroid
        )
        target_centered = target_coefficients - np.outer(
            constant_projection, target_centroid
        )

        # Optimal Z-rotation aligning the centered predicted reconstruction with the centered target reconstruction
        a, b = predicted_centered[:, :2], target_centered[:, :2]
        rotation_angle = np.arctan2(
            np.sum(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]),
            np.sum(a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1]),
        )
        cos_a, sin_a = np.cos(rotation_angle), np.sin(rotation_angle)
        rotation_matrix_2d = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        a_aligned = a @ rotation_matrix_2d.T

        shape_error = np.sqrt(
            np.sum((a_aligned - b) ** 2)
            + np.sum((predicted_centered[:, 2] - target_centered[:, 2]) ** 2)
        ) / np.sqrt(total_mass)

        return {
            "reconstruction_error_mm": reconstruction_error * 1000,
            "position_error_mm": position_error * 1000,
            "rotation_error": np.abs(rotation_angle),
            "shape_error_mm": shape_error * 1000,
        }

    def _step(
        self,
        action: dict[str, np.ndarray],
        prediction: np.ndarray,
    ):
        prev_done = np.array(self._prev_done)

        representations = [
            self.__compute_spectral_representation_cached(dp)
            for dp in self.current_data_points
        ]
        target_coefficients = self.__get_target_coefficients(representations)
        predicted_coefficients = (
            prediction * self.__target_std + self.__target_mean
        ).reshape(self.num_envs, self.__num_coefficients, 3)

        for i in range(self.num_envs):
            metrics = self._decompose_coefficient_error(
                representations[i], predicted_coefficients[i], target_coefficients[i]
            )
            metrics["rel_error"] = metrics["reconstruction_error_mm"] / (
                max(representations[i].rms_radius, 1e-10) * 1000
            )
            for name, value in metrics.items():
                if prev_done[i]:
                    self.__metrics[name][i].clear()
                else:
                    self.__metrics[name][i].append(value)

        obs, action_reward, terminated, truncated, info, targets = super()._step(
            action, prediction
        )

        if self.__renderer_show_shadow_objects:
            # Do that after the step as new objects might be loaded. The predicted coefficients are expressed in the
            # platform frame already, hence the shadow objects are placed at the platform frame's origin.
            prediction_clipped = np.clip(
                prediction,
                self.single_prediction_space.low,
                self.single_prediction_space.high,
            )
            reconstructed_coefficients = (
                prediction_clipped * self.__target_std + self.__target_mean
            ).reshape(self.num_envs, self.__num_coefficients, 3)
            self._renderer.update_shadow_objects(
                Transformation.batch_concatenate(
                    [Transformation()] * self.num_envs,
                ),
                shadow_object_visible=~prev_done,
                new_shadow_object_vertices=[
                    (
                        None
                        if prev_done[i]
                        else r.reconstruct(reconstructed_coefficients[i])
                    )
                    for i, r in enumerate(representations)
                ],
            )

        if np.any(terminated | truncated):
            info = update_info_metrics_vec(info, self.__metrics, terminated | truncated)

        return obs, action_reward, terminated, truncated, info, targets

    def __compute_spectral_representation(
        self, dp: GenericMeshDataPoint
    ) -> SpectralShapeRepresentation:
        return compute_spectral_representation(dp.mesh, self.__num_coefficients)

    def __normalize_targets(self, targets: np.ndarray) -> np.ndarray:
        return (targets - self.__target_mean) / self.__target_std

    def __get_target_coefficients(
        self, representations: list[SpectralShapeRepresentation]
    ) -> np.ndarray:
        return np.stack(
            [
                representation.transform_coefficients(pose)
                for representation, pose in zip(
                    representations, self.current_object_poses_platform_frame
                )
            ]
        )

    def _get_prediction_targets(self) -> np.ndarray:
        representations = [
            self.__compute_spectral_representation_cached(dp)
            for dp in self.current_data_points
        ]
        target_coefficients = self.__get_target_coefficients(representations)
        return self.__normalize_targets(
            target_coefficients.reshape(self.num_envs, -1)
        ).astype(np.float32)

    @property
    def num_coefficients(self) -> int:
        return self.__num_coefficients


def TactileShapeReconstructionEnv(
    config: TactilePerceptionConfig,
    render_mode: Literal["rgb_array", "human"] = "rgb_array",
    num_coefficients: int = 64,
    renderer_show_shadow_objects: bool = True,
) -> ActivePerceptionVectorToSingleWrapper["ObsType", ActType, np.ndarray, np.ndarray]:
    return ActivePerceptionVectorToSingleWrapper(
        TactileShapeReconstructionVectorEnv(
            config,
            1,
            render_mode=render_mode,
            num_coefficients=num_coefficients,
            renderer_show_shadow_objects=renderer_show_shadow_objects,
        )
    )

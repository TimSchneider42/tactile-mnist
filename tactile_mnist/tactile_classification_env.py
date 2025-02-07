from typing import Dict, Union, Optional, Literal, Sequence, TYPE_CHECKING

import numpy as np

from ap_gym import ActivePerceptionVectorToSingleWrapper
from .mesh_dataset import MeshDataset
from .tactile_classification_vector_env import TactileClassificationVectorEnv

if TYPE_CHECKING:
    import torch
    import jax

    ObsType = Dict[str, Union[np.ndarray, torch.Tensor, jax.Array]]
    ActType = Dict[str, np.ndarray]


def TactileClassificationEnv(
    dataset: MeshDataset,
    max_episode_steps: int = 16,
    render_mode: Literal["rgb_array", "human"] = "rgb_array",
    taxim_device: Optional[str] = None,
    convert_image_to_numpy: bool = True,
    show_sensor_target_pos: bool = False,
    perturb_object_pose: bool = True,
    randomize_initial_object_pose: bool = True,
    randomize_initial_sensor_pose: bool = True,
    sensor_output_size: Optional[Sequence[int]] = None,
    depth_only: bool = False,
    allow_sensor_rotation: bool = True,
    sensor_backend: Literal["torch", "jax", "auto"] = "auto",
    linear_velocity: float = 0.1,
    angular_velocity: float = np.pi / 2,
    linear_acceleration: float = 2.0,
    angular_acceleration: float = 10 * np.pi,
    transfer_timedelta_s: float = 0.2,
    action_regularization: float = 1e-3,
    max_tilt_angle: float = np.pi / 4,
    render_transparent_background: bool = False,
) -> ActivePerceptionVectorToSingleWrapper[
    "ObsType", "ActType", np.ndarray, np.ndarray
]:
    return ActivePerceptionVectorToSingleWrapper(
        TactileClassificationVectorEnv(
            dataset,
            1,
            max_episode_steps=max_episode_steps,
            render_mode=render_mode,
            taxim_device=taxim_device,
            convert_image_to_numpy=convert_image_to_numpy,
            show_sensor_target_pos=show_sensor_target_pos,
            perturb_object_pose=perturb_object_pose,
            randomize_initial_object_pose=randomize_initial_object_pose,
            randomize_initial_sensor_pose=randomize_initial_sensor_pose,
            sensor_output_size=sensor_output_size,
            depth_only=depth_only,
            allow_sensor_rotation=allow_sensor_rotation,
            sensor_backend=sensor_backend,
            linear_velocity=linear_velocity,
            angular_velocity=angular_velocity,
            linear_acceleration=linear_acceleration,
            angular_acceleration=angular_acceleration,
            transfer_timedelta_s=transfer_timedelta_s,
            action_regularization=action_regularization,
            max_tilt_angle=max_tilt_angle,
            render_transparent_background=render_transparent_background,
        )
    )

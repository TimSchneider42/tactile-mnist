from __future__ import annotations

import copy
from abc import abstractmethod, ABC
from functools import partial
from itertools import chain
from typing import (
    Any,
    Literal,
    Sequence,
    TYPE_CHECKING,
    Generic,
    TypeVar,
    List,
)

import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import EnvSpec
from gymnasium.vector.utils import batch_space
from scipy.spatial.transform import Rotation
from taxim import Taxim, CALIB_GELSIGHT_MINI
from transformation import Transformation

from ap_gym import (
    ImageSpace,
    ActiveClassificationVectorEnv,
    ActivePerceptionVectorToSingleWrapper,
)
from tactile_mnist import (
    CELL_MARGIN,
    CELL_SIZE,
    GELSIGHT_IMAGE_SIZE_PX,
    MeshDataPoint,
    MeshDataset,
    GELSIGHT_GEL_THICKNESS_MM,
)
from .tactile_classification_renderer import TactileClassificationRenderer
from .util import OverridableStaticField, transformation_where

try:
    import torch
    import torchvision
except ImportError:
    torch = torchvision = None

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    jax = jnp = None

if TYPE_CHECKING:
    ObsType = dict[str, np.ndarray | torch.Tensor | jax.Array]
    ActType = dict[str, np.ndarray]

ArrayType = TypeVar("ArrayType")


class Backend(ABC, Generic[ArrayType]):
    @abstractmethod
    def depth_map_to_numpy(self, depth_map: ArrayType) -> np.ndarray:
        pass

    @abstractmethod
    def depth_map_to_img(
        self, depth_map: ArrayType, min_depth: float, max_depth: float
    ) -> ArrayType:
        pass

    @abstractmethod
    def scale_img(self, img: ArrayType, height: int, width: int) -> ArrayType:
        pass


if torch is not None:

    class TorchBackend(Backend[torch.Tensor]):
        def depth_map_to_numpy(self, depth_map: torch.Tensor) -> np.ndarray:
            return depth_map.cpu().numpy()

        def depth_map_to_img(
            self, depth_map: ArrayType, min_depth: float, max_depth: float
        ) -> torch.Tensor:
            depth_img = (torch.clip(depth_map, min_depth, max_depth) - min_depth) / (
                max_depth - min_depth
            )
            return torch.broadcast_to(depth_img[None], (3, *depth_img.shape))

        def scale_img(self, img: torch.Tensor, height: int, width: int) -> torch.Tensor:
            return torchvision.transforms.functional.resize(
                img,
                (height, width),
                torchvision.transforms.InterpolationMode.BICUBIC,
                antialias=True,
            ).clip(0, 1)

else:
    TorchBackend = None

if jax is not None:

    class JaxBackend(Backend[jax.Array]):
        def depth_map_to_numpy(self, depth_map: jax.Array) -> np.ndarray:
            return np.array(depth_map)

        # Make min_depth and max_depth static as they do not change between calls
        @partial(jax.jit, static_argnames=("self", "min_depth", "max_depth"))
        def depth_map_to_img(
            self, depth_map: ArrayType, min_depth: float, max_depth: float
        ) -> jax.Array:
            depth_img = (jnp.clip(depth_map, min_depth, max_depth) - min_depth) / (
                max_depth - min_depth
            )
            return jnp.broadcast_to(depth_img[None], (*depth_img.shape, 3))

        # Make height and width static as they do not change between calls
        @partial(jax.jit, static_argnames=("self", "height", "width"))
        def scale_img(self, img: jax.Array, height: int, width: int) -> jax.Array:
            shape = img.shape[:-3] + (height, width, 3)
            return jax.image.resize(img, shape, method="bicubic", antialias=True).clip(
                0, 1
            )

else:
    JaxBackend = None


class TactileClassificationVectorEnv(
    ActiveClassificationVectorEnv["ObsType", "ActType"]
):
    metadata: dict[str, Any] = OverridableStaticField(
        {
            "render_fps": 5,
            "render_modes": ["rgb_array", "human"],
            "autoreset_mode": gym.vector.AutoresetMode.NEXT_STEP,
        }
    )

    @metadata.dynamic_update
    def metadata(self):
        return {**type(self).metadata, "render_fps": 1 / self._transfer_timedelta_s}

    def __init__(
        self,
        dataset: MeshDataset | Sequence[MeshDataset],
        num_envs: int,
        step_limit: int = 16,
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        taxim_device: str | None = None,
        convert_image_to_numpy: bool = True,
        show_sensor_target_pos: bool = False,
        perturb_object_pose: bool = True,
        randomize_initial_object_pose: bool = True,
        sensor_output_size: Sequence[int] | None = None,
        randomize_initial_sensor_pose: bool = True,
        depth_only: bool = False,
        allow_sensor_rotation: bool = True,
        sensor_backend: Literal["torch", "jax", "auto"] = "auto",
        linear_velocity: float = 0.2,
        angular_velocity: float = np.pi / 2,
        linear_acceleration: float = 4.0,
        angular_acceleration: float = 10 * np.pi,
        transfer_timedelta_s: float = 0.2,
        action_regularization: float = 1e-3,
        max_tilt_angle: float = np.pi / 4,
        render_transparent_background: bool = False,
    ):
        sensor_output_size = tuple(
            map(
                int,
                GELSIGHT_IMAGE_SIZE_PX
                if sensor_output_size is None
                else sensor_output_size,
            )
        )
        self.num_envs = num_envs
        self._sensor_output_hw = tuple(reversed(sensor_output_size))
        self._render_mode = render_mode
        self._depth_only = depth_only
        self._transfer_timedelta_s = transfer_timedelta_s
        if isinstance(dataset, MeshDataset):
            dataset = [dataset] * num_envs
        else:
            assert len(dataset) == num_envs
        self._datasets = dataset
        self._label_map = {
            i: l
            for i, l in enumerate(
                sorted(set(chain(*(ds.labels for ds in self._datasets))))
            )
        }
        self._inverse_label_map = {l: i for i, l in self._label_map.items()}
        self._current_data_points: tuple[MeshDataPoint] | None = None
        self._sensor_device = taxim_device
        self._sensor = Taxim(
            calib_folder=CALIB_GELSIGHT_MINI,
            device=taxim_device,
            params={"simulator": {"contact_scale": 0.6}},
            backend=sensor_backend,
        )
        if self._sensor.backend_name == "torch":
            self._backend = TorchBackend()
        else:
            self._backend = JaxBackend()
            # For some reason, JITing Taxim inside a host callback deadlocks, so we have to make sure it happens before
            output = self._sensor.render_direct(
                jnp.zeros(
                    (self.num_envs, self._sensor.height, self._sensor.width),
                    dtype=jnp.float32,
                )
            )
            self._backend.scale_img(output, *self._sensor_output_hw)
        self._gel_thickness_mm = GELSIGHT_GEL_THICKNESS_MM
        self._gel_penetration_depth_mm = self._gel_thickness_mm / 2
        self._show_sensor_target_pos = show_sensor_target_pos
        self._allow_sensor_rotation = allow_sensor_rotation
        self._randomize_initial_sensor_pose = randomize_initial_sensor_pose
        dt = np.float32
        all_labels = {l for ds in dataset for l in ds.labels}
        single_action_space = {
            # Target position of the sensor
            "sensor_target_pos_rel": gym.spaces.Box(
                -np.ones(3, dtype=dt), np.ones(3, dtype=dt)
            )
        }
        single_observation_space = {
            "sensor_pos": gym.spaces.Box(-np.ones(3, dtype=dt), np.ones(3, dtype=dt)),
            "sensor_img": ImageSpace(
                sensor_output_size[0], sensor_output_size[1], 3, dtype=dt
            ),
            "time_step": gym.spaces.Box(-np.ones((), dtype=dt), np.ones((), dtype=dt)),
        }

        if allow_sensor_rotation:
            single_action_space["sensor_target_rot_rel"] = gym.spaces.Box(
                -np.ones(3, dtype=dt), np.ones(3, dtype=dt)
            )
            single_observation_space["sensor_rot"] = gym.spaces.Box(
                -np.ones(6, dtype=dt), np.ones(6, dtype=dt)
            )
        single_action_space = gym.spaces.Dict(single_action_space)
        super().__init__(
            num_envs,
            len(all_labels),
            single_action_space,
        )
        self.single_observation_space = gym.spaces.Dict(single_observation_space)
        self.observation_space = batch_space(self.single_observation_space, num_envs)

        self._convert_image_to_numpy = convert_image_to_numpy
        self._current_sensor_pose_platform_frame = Transformation.from_pos_euler(
            [[0.0, 0.0, 0.1]] * num_envs, [[0.0, np.pi, 0.0]] * num_envs
        )
        self._current_sensor_target_pose_platform_frame = (
            self._current_sensor_pose_platform_frame
        )
        self._perturb_object_pose = perturb_object_pose
        self._randomize_initial_object_pose = randomize_initial_object_pose

        self._sensor_pos_limits = (
            np.concatenate([-CELL_SIZE / 2 + CELL_MARGIN, [0.0]]),
            np.concatenate([CELL_SIZE / 2 - CELL_MARGIN, [0.02]]),
        )

        self._object_poses_platform_frame: Transformation | None = None
        self._current_step: np.ndarray | None = None
        self._last_sensor_output: np.ndarray | None = None

        self._renderer = TactileClassificationRenderer(
            self.num_envs,
            (self._sensor.width, self._sensor.height),
            self._sensor.sensor_params.pixmm,
            show_viewer=render_mode == "human",
            show_sensor_target_pos=show_sensor_target_pos,
            transparent_background=render_transparent_background,
        )

        self._step_limit = step_limit
        self._linear_velocity = linear_velocity
        self._angular_velocity = angular_velocity
        self._linear_acceleration = linear_acceleration
        self._angular_acceleration = angular_acceleration
        self._action_regularization = action_regularization
        self._max_tilt_angle = max_tilt_angle

        # Calculate the maximum distance the sensor can travel in one step
        self._max_distance_linear = self._calculate_max_distance_scalar(
            transfer_timedelta_s, linear_acceleration, linear_velocity
        )
        self._max_distance_angular = self._calculate_max_distance_scalar(
            transfer_timedelta_s, angular_acceleration, angular_velocity
        )
        self._prev_done = None
        self._spec: EnvSpec | None = None

    def _sample_sensor_target_poses(self, count: int) -> List[Transformation]:
        sensor_poses = []
        for i in range(count):
            position = np.zeros(3, dtype=np.float32)
            rotation = Rotation.identity()
            if self._randomize_initial_sensor_pose:
                position[:2] = self.np_random.uniform(
                    low=self._sensor_pos_limits[0][:2],
                    high=self._sensor_pos_limits[1][:2],
                    size=(2,),
                )
                if self._allow_sensor_rotation:
                    polar_angle = self.np_random.uniform(
                        low=0, high=self._max_tilt_angle
                    )
                    azimuthal_angle = self.np_random.uniform(low=-np.pi, high=np.pi)
                    z_angle = self.np_random.uniform(low=-np.pi, high=np.pi)
                    rotation = Rotation.from_euler(
                        "zyz", [z_angle, polar_angle, azimuthal_angle]
                    )
                else:
                    rotation = Rotation.identity()

            sensor_poses.append(Transformation(position, rotation))
        return sensor_poses

    def _reset_partial(
        self, mask: Sequence[bool], options: dict[str, Any] | None = None
    ) -> np.ndarray:
        if np.any(mask):
            if options is None:
                options = {}

            datapoint_idx = list(options.get("datapoint_idx", [None] * self.num_envs))
            current_datapoints_lst = list(self._current_data_points)
            object_poses_lst = [Transformation() for _ in range(self.num_envs)]
            for i in np.where(mask)[0]:
                idx = (
                    self.np_random.integers(0, len(self._datasets[i]))
                    if datapoint_idx[i] is None
                    else datapoint_idx[i]
                )
                current_datapoints_lst[i] = self._datasets[i][idx]
                initial_pose = Transformation(
                    [
                        0,
                        0,
                        np.quantile(
                            -current_datapoints_lst[i].mesh.vertices[:, 2], 0.9
                        ),
                    ]
                )
                if self._randomize_initial_object_pose:
                    xy_min = np.min(
                        current_datapoints_lst[i].mesh.vertices[:, :2], axis=0
                    )
                    xy_max = np.max(
                        current_datapoints_lst[i].mesh.vertices[:, :2], axis=0
                    )
                    margin = 0.01
                    low = -CELL_SIZE / 2 + margin - xy_min
                    high = CELL_SIZE / 2 - margin - xy_max
                    conflict = low > high
                    low[conflict] = high[conflict] = ((low + high) / 2)[conflict]
                    translation_perturbation = self.np_random.uniform(
                        low=low, high=high
                    )
                    rotation_perturbation = self.np_random.uniform(
                        low=-np.pi / 8, high=np.pi / 8, size=(1,)
                    )
                    perturbation = Transformation.from_pos_euler(
                        np.concatenate(
                            [translation_perturbation, np.zeros((1,), dtype=np.float32)]
                        ),
                        np.concatenate(
                            [np.zeros((2,), dtype=np.float32), rotation_perturbation]
                        ),
                    )
                    initial_pose *= perturbation
                object_poses_lst[i] = initial_pose

            self._current_data_points = tuple(current_datapoints_lst)
            assert all(dp is not None for dp in self._current_data_points)
            self._renderer.objects = self._current_data_points
            self._object_poses_platform_frame = Transformation.batch_concatenate(
                object_poses_lst
            )
            self._renderer.set_object_poses(
                self._object_poses_platform_frame, mask=mask
            )
            self._current_step[mask] = np.zeros(np.sum(mask), dtype=np.float32)

        return np.array(
            [
                self._inverse_label_map[dp.metadata.label]
                for dp in self._current_data_points
            ]
        )

    def _get_obs_info(
        self, sensor_target_poses: Transformation
    ) -> tuple["ObsType", dict[str, Any]]:
        sensor_output, depth_output, sensor_pose = self.execute_step(
            sensor_target_poses
        )

        sensor_pos_min, sensor_pos_max = self._sensor_pos_limits
        sensor_pos_normalized = (sensor_pose.translation - sensor_pos_min) / (
            sensor_pos_max - sensor_pos_min
        ) * 2 - 1

        obs = {
            "sensor_pos": sensor_pos_normalized.astype(np.float32),
            "sensor_img": sensor_output,
            "time_step": (self._current_step / self._step_limit * 2 - 1).astype(
                np.float32
            ),
        }

        if self._allow_sensor_rotation:
            obs["sensor_rot"] = self.rotation_to_feature(sensor_pose.rotation)

        info = {"depth": depth_output, "sensor_pose": sensor_pose}
        return obs, info

    def _reset(self, *, options: dict[str, Any] | None = None):
        self._current_step = np.zeros(self.num_envs, dtype=np.int_)
        self._current_data_points = [None] * self.num_envs
        self._prev_done = np.zeros(self.num_envs, dtype=np.bool_)
        labels = self._reset_partial(
            np.ones(self.num_envs, dtype=np.bool_), options=options
        )
        sensor_target_poses = self._sample_sensor_target_poses(self.num_envs)
        obs, info = self._get_obs_info(
            Transformation.batch_concatenate(sensor_target_poses)
        )
        return obs, info, labels

    @staticmethod
    def rotation_to_feature(rot: Rotation) -> np.ndarray:
        """
        Extract rotation features according to this paper:
        https://openaccess.thecvf.com/content_CVPR_2019/html/Zhou_On_the_Continuity_of_Rotation_Representations_in_Neural_Networks_CVPR_2019_paper.html
        Unlike the paper though, we include the second and third column instead of the first and second, as it helps
        us to ensure that the sensor only receives downwards pointing target orientations.
        :param rot: Rotation to compute features representation for.
        :return: 6D feature representation of the given rotation.
        """
        matrix = rot.inv().as_matrix()
        return matrix.reshape((*matrix.shape[:-2], -1))[..., 3:]

    @staticmethod
    def feature_to_rotation(feature: np.ndarray) -> Rotation:
        z_axis_unnorm = feature[..., 3:]
        z_norm = np.linalg.norm(z_axis_unnorm, axis=-1, keepdims=True)
        assert np.all(z_norm > 0)
        z_axis = z_axis_unnorm / z_norm
        y_axis_unnorm = (
            feature[..., :3]
            - (z_axis * feature[..., :3]).sum(-1, keepdims=True) * z_axis
        )
        y_norm = np.linalg.norm(y_axis_unnorm, axis=-1, keepdims=True)
        assert np.all(y_norm > 0)
        y_axis = y_axis_unnorm / y_norm
        x_axis = np.cross(y_axis, z_axis)
        return Rotation.from_matrix(np.stack([x_axis, y_axis, z_axis], axis=-1))

    @staticmethod
    def _calculate_transfer_time_scalar(
        distance: float | np.ndarray, acceleration: float, max_velocity: float
    ) -> float | np.ndarray:
        half_distance = distance / 2
        unconstrained_acceleration_time = np.sqrt(2 * half_distance / acceleration)
        acceleration_time = max_velocity / acceleration
        remaining_distance = half_distance - 0.5 * acceleration * acceleration_time**2
        max_velocity_time = remaining_distance / max_velocity
        return 2 * np.where(
            remaining_distance > 0,
            acceleration_time + max_velocity_time,
            unconstrained_acceleration_time,
        )

    @staticmethod
    def _calculate_max_distance_scalar(
        transfer_time: float | np.ndarray,
        acceleration: float,
        max_velocity: float,
    ) -> float | np.ndarray:
        half_transfer_time = transfer_time / 2
        acceleration_time = np.minimum(max_velocity / acceleration, half_transfer_time)
        max_velocity_time = half_transfer_time - acceleration_time
        return (
            max_velocity_time * max_velocity
            + 0.5 * acceleration * acceleration_time**2
        )

    def _calculate_transfer_time(self, relative_pose: Transformation):
        linear_distance = np.linalg.norm(relative_pose.translation, axis=-1)
        linear_transfer_time = self._calculate_transfer_time_scalar(
            linear_distance, self._linear_acceleration, self._linear_velocity
        )
        angular_distance = relative_pose.angle
        angular_transfer_time = self._calculate_transfer_time_scalar(
            angular_distance, self._angular_acceleration, self._angular_velocity
        )
        return np.maximum(linear_transfer_time, angular_transfer_time)

    @staticmethod
    def _project_sphere(x: np.ndarray, radius: float = 1.0) -> np.ndarray:
        # We mask here to avoid the special case of 0 (or very low) magnitude
        magnitude = np.linalg.norm(x, axis=-1)
        mask = magnitude > radius
        direction = x[mask] / magnitude[mask, None]
        output = x.copy()
        output[mask] = direction * radius
        return output

    def _step(
        self,
        action: "ActType",
        prediction: np.ndarray,
    ):
        labels = self._reset_partial(self._prev_done)

        sensor_pos_min, sensor_pos_max = self._sensor_pos_limits
        sensor_target_pos_rel = action["sensor_target_pos_rel"]
        if np.any(np.isnan(sensor_target_pos_rel)):
            raise ValueError("NaN values detected in sensor target position.")
        action_reward = np.mean(
            -(sensor_target_pos_rel**2) * self._action_regularization, axis=-1
        )

        # Project everything back into unit sphere
        sensor_target_pos_rel_clipped = self._project_sphere(sensor_target_pos_rel)
        sensor_target_pos_rel_scaled = (
            sensor_target_pos_rel_clipped * self._max_distance_linear
        )
        sensor_target_pos_unconstrained = (
            self._current_sensor_pose_platform_frame.translation
            + sensor_target_pos_rel_scaled
        )
        sensor_target_pos = np.clip(
            sensor_target_pos_unconstrained, sensor_pos_min, sensor_pos_max
        )

        if self._allow_sensor_rotation:
            sensor_target_rot_rel = action["sensor_target_rot_rel"]
            if np.any(np.isnan(sensor_target_rot_rel)):
                raise ValueError("NaN values detected in sensor target rotation.")
            action_reward += np.mean(
                -(sensor_target_rot_rel**2) * self._action_regularization, axis=-1
            )
            sensor_target_rot_rel_clipped = self._project_sphere(sensor_target_rot_rel)
            sensor_target_rot_rel_scaled = (
                sensor_target_rot_rel_clipped * self._max_distance_angular
            )

            new_sensor_target_rot = (
                Rotation.from_rotvec(sensor_target_rot_rel_scaled)
                * self._current_sensor_pose_platform_frame.rotation
            )

            # Ensure that sensor does not violate the maximum tilt angle
            # We do this by restricting the angle between the Z-axis of the rotated sensor and the world Z-axis
            sensor_target_rot_feat = self.rotation_to_feature(new_sensor_target_rot)
            max_radius_xy = np.sin(self._max_tilt_angle)
            xy = sensor_target_rot_feat[..., 3:5]
            xy_clipped = self._project_sphere(xy, max_radius_xy)
            z_component = np.sqrt(1 - np.sum(xy_clipped**2, axis=-1))
            sensor_target_rot_feat[..., 3:5] = xy_clipped
            sensor_target_rot_feat[..., 5] = z_component
            sensor_target_rot = self.feature_to_rotation(sensor_target_rot_feat)
        else:
            sensor_target_rot = Rotation.from_matrix(np.eye(3))
        sensor_target_rot_mat = sensor_target_rot.as_matrix()
        assert np.all(sensor_target_rot_mat[..., 2, 2] >= 0)

        sensor_target_pose = Transformation(sensor_target_pos, sensor_target_rot)
        if np.any(self._prev_done):
            sensor_target_pose = transformation_where(
                self._prev_done,
                Transformation.batch_concatenate(
                    self._sample_sensor_target_poses(self.num_envs)
                ),
                sensor_target_pose,
            )

        # relative_sensor_pose = self._current_sensor_pose_platform_frame.inv * sensor_pose
        # transfer_time = self._calculate_transfer_time(relative_sensor_pose)

        self._current_step[~self._prev_done] += 1
        terminated = self._current_step >= self._step_limit
        truncated = np.zeros(self.num_envs, dtype=np.bool_)

        obs, info = self._get_obs_info(sensor_target_pose)

        action_reward = np.where(self._prev_done, 0, action_reward)
        self._prev_done = terminated | truncated
        return obs, action_reward, terminated, truncated, info, labels

    def execute_step(
        self, sensor_target_pose: Transformation, mask: Sequence[bool] | None = None
    ):
        if mask is None:
            mask = np.ones(self.num_envs, dtype=np.bool_)
        self._current_sensor_target_pose_platform_frame = transformation_where(
            mask, sensor_target_pose, self._current_sensor_target_pose_platform_frame
        )
        sensor_output, depth_output, current_sensor_pose_platform_frame = self.touch(
            sensor_target_pose
        )
        self._current_sensor_pose_platform_frame = transformation_where(
            mask,
            current_sensor_pose_platform_frame,
            self._current_sensor_pose_platform_frame,
        )
        self._renderer.sensor_poses = self._current_sensor_pose_platform_frame
        self._renderer.sensor_shadow_poses = (
            self._current_sensor_target_pose_platform_frame
        )
        if self._perturb_object_pose:
            translation_perturbation = self.np_random.normal(scale=1e-3, size=2)
            rotation_perturbation = self.np_random.normal(scale=5e-2)
            perturbation = Transformation.from_pos_euler(
                np.concatenate([translation_perturbation, [0]]),
                [0, 0, rotation_perturbation],
            )
            self._renderer.set_object_poses(
                self._object_poses_platform_frame * perturbation, mask=mask
            )
        self._last_sensor_output = sensor_output
        return sensor_output, depth_output, self._current_sensor_pose_platform_frame

    def touch(self, sensor_target_poses: Transformation):
        depth_gel_frame_shifted = self._renderer.render_sensor_depths(
            sensor_target_poses
        )
        offset = self._gel_penetration_depth_mm / 1000 - np.min(
            depth_gel_frame_shifted, axis=(-1, -2)
        )
        depth_gel_frame = depth_gel_frame_shifted + offset[:, None, None]

        sensor_pose_target_frame = Transformation(
            np.concatenate(
                [np.zeros((offset.shape[0], 2), dtype=np.float32), offset[:, None]],
                axis=-1,
            )
        )
        sensor_poses = sensor_target_poses * sensor_pose_target_frame

        depth_mm = depth_gel_frame * 1000
        depth_conv = self._sensor.convert_height_map(depth_mm)
        if self._depth_only:
            sensor_output = self._backend.depth_map_to_img(
                depth_conv, self._gel_penetration_depth_mm, self._gel_thickness_mm
            )
        else:
            # Taxim expects the depth relative to the highest point of the gel
            sensor_output = self._sensor.render_direct(
                depth_conv - self._gel_thickness_mm
            )
        sensor_output_scaled = self._backend.scale_img(
            sensor_output, *self._sensor_output_hw
        )

        if self._convert_image_to_numpy:
            sensor_output = self._sensor.img_to_numpy(sensor_output_scaled)
            depth_output = self._backend.depth_map_to_numpy(depth_conv)
        else:
            sensor_output = sensor_output_scaled
            depth_output = depth_conv

        return sensor_output, depth_output, sensor_poses

    def render(self) -> np.ndarray | None:
        return self._renderer.render_external_cameras(self._last_sensor_output)

    @property
    def render_mode(self):
        return self._render_mode

    @property
    def sensor_pos_limits(self) -> tuple[np.ndarray, np.ndarray]:
        return self._sensor_pos_limits

    @property
    def current_data_points(self) -> tuple[MeshDataPoint]:
        return self._current_data_points

    @property
    def current_object_poses_platform_frame(self) -> Transformation:
        return self._object_poses_platform_frame

    @property
    def step_limit(self) -> int:
        return self._step_limit

    @property
    def spec(self) -> EnvSpec | None:
        return self._spec

    @spec.setter
    def spec(self, spec: EnvSpec):
        spec = copy.copy(spec)
        spec.max_episode_steps = self._step_limit
        self._spec = spec


def TactileClassificationEnv(
    dataset: MeshDataset,
    step_limit: int = 16,
    render_mode: Literal["rgb_array", "human"] = "rgb_array",
    taxim_device: str | None = None,
    convert_image_to_numpy: bool = True,
    show_sensor_target_pos: bool = False,
    perturb_object_pose: bool = True,
    randomize_initial_object_pose: bool = True,
    randomize_initial_sensor_pose: bool = True,
    sensor_output_size: Sequence[int] | None = None,
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
            step_limit=step_limit,
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

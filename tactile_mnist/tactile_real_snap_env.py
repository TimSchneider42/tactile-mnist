from __future__ import annotations

import copy
import logging
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import (
    Any,
    Generic,
    Literal,
    Sequence,
)

import PIL.Image
import ap_gym
import gymnasium as gym
import numpy as np
import scipy.special
from ap_gym import (
    ActivePerceptionActionSpace,
    ActivePerceptionVectorEnv,
    ActivePerceptionVectorToSingleWrapper,
    CrossEntropyLossFn,
    ImageSpace,
    LogitSpace,
    MSELossFn,
)
from ap_gym.types import PredType, PredTargetType
from ap_gym.util import update_info_metrics_vec
from gymnasium.envs.registration import EnvSpec
from gymnasium.vector.utils import batch_space
from transformation import Transformation

from .constants import (
    CELL_PADDING,
    CELL_SIZE,
    GEL_PENETRATION_DEPTH_MM,
    GELSIGHT_MINI_SENSOR_SURFACE_SIZE,
)
from .mesh_dataset import MeshDataPoint, MeshDataset
from .tactile_perception_renderer import TactilePerceptionRenderer
from .tactile_renderer import mk_tactile_renderer
from .touch_data import TouchSingle, TouchSingleDataset
from .util import get_dataset_stats, int_binary_search_right, compute_touch_window_size

logger = logging.getLogger(__file__)

ObsType = dict[str, np.ndarray]
ActType = dict[str, np.ndarray]


@dataclass(frozen=True)
class TactileRealSnapConfig:
    dataset: TouchSingleDataset | Sequence[TouchSingleDataset]
    step_limit: int = 16
    sensor_output_size: Sequence[int] | None = None
    randomize_initial_sensor_pose: bool = True
    linear_velocity: float = 0.2
    linear_acceleration: float = 4.0
    transfer_timedelta_s: float = 0.2
    action_regularization: float = 1e-3
    timeout_behavior: Literal["terminate", "truncate"] = "terminate"
    cell_size: tuple[float, float] = tuple(CELL_SIZE)
    cell_padding: tuple[float, float] = tuple(CELL_PADDING)
    # The mesh of each round is looked up by matching the round's object_id against the mesh datapoint ids. The mesh
    # dataset is required for regression tasks (e.g. volume estimation) and is otherwise used for visualization only:
    # if it is provided and enable_rendering is set, env.render() shows the object (in a default position, as the
    # actual object pose is not known), the effective sensor pose of the selected touch, and the requested target
    # sensor position.
    mesh_dataset: MeshDataset | Sequence[MeshDataset] | None = None
    # The zero of the recorded gel z-positions drifts between rounds (by up to 3mm), as it depends on the state of the
    # gel and the calibration of the robot at the time of recording. If recalibrate_sensor_z is set, the recorded
    # z-positions of each round are re-zeroed on the deepest touch of that round, which is the platform surface (each
    # round contains touches that miss the object), and shifted such that touching the platform yields the same
    # sensor z-position as it does in the simulated environments (GEL_PENETRATION_DEPTH_MM above the platform).
    # Without this correction, the z-component of the sensor_pos observation is offset by an unknown per-round
    # constant and does not match the simulated environments.
    recalibrate_sensor_z: bool = True
    enable_rendering: bool = True
    show_sensor_target_pos: bool = True
    renderer_show_tactile_image: bool = True
    renderer_show_class_weights: bool = False
    render_transparent_background: bool = False
    renderer_external_camera_resolution: tuple[int, int] = (640, 480)
    # The touch window size will be chosen such that the probability of running into an early truncation is
    # approximately this value. There is one caveat: the model used to compute the window size assumes that samples are
    # chosen uniformly at random from the window. However, this assumption neglects the fact that the distribution is
    # skewed. Due to survivor bias, the distribution favors later samples: if a sample was already in the previous
    # window and not chosen, it is likely far away and won't be chosen again. So the actual truncation probability will
    # be slightly above this value.
    approx_truncation_probability: float = 0.1


class TactileRealSnapVectorEnv(
    ActivePerceptionVectorEnv[ObsType, ActType, PredType, PredTargetType, np.ndarray],
    Generic[PredType, PredTargetType],
    ABC,
):
    metadata: dict[str, Any] = {
        "render_fps": 5,
        "render_modes": ["rgb_array", "human"],
        "autoreset_mode": gym.vector.AutoresetMode.NEXT_STEP,
    }

    def __init__(
        self,
        config: TactileRealSnapConfig,
        num_envs: int,
        single_prediction_space: gym.Space[PredType],
        single_prediction_target_space: gym.Space[PredTargetType],
        loss_fn: ap_gym.LossFn,
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
    ):
        self.__config = config
        self.num_envs = num_envs
        self.__render_mode = render_mode
        self.metadata = {
            **type(self).metadata,
            "render_fps": 1 / config.transfer_timedelta_s,
        }
        if isinstance(config.dataset, TouchSingleDataset):
            self.__datasets = [config.dataset] * num_envs
        else:
            assert len(config.dataset) == num_envs
            self.__datasets = list(config.dataset)

        self.__available_touches_per_sequence = (
            config.dataset.huggingface_dataset.features["pos_in_cell"].length
        )

        self.__touch_window_size = compute_touch_window_size(
            self.__available_touches_per_sequence,
            config.step_limit,
            config.approx_truncation_probability,
        )
        logger.info(
            f"Determined touch sequence window size to be {self.__touch_window_size}."
        )

        if config.sensor_output_size is None:
            first_image = np.asarray(self.__datasets[0][0].sensor_image[0])
            sensor_output_size = (first_image.shape[1], first_image.shape[0])
        else:
            sensor_output_size = tuple(map(int, config.sensor_output_size))
        self.__sensor_output_size = sensor_output_size

        dt = np.float32
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
        }
        if config.timeout_behavior == "terminate":
            single_observation_space["time_step"] = gym.spaces.Box(
                -np.ones((), dtype=dt), np.ones((), dtype=dt)
            )

        self.single_prediction_target_space = single_prediction_target_space
        self.prediction_target_space = batch_space(
            self.single_prediction_target_space, num_envs
        )
        self.single_action_space = ActivePerceptionActionSpace(
            gym.spaces.Dict(single_action_space), single_prediction_space
        )
        self.action_space = batch_space(self.single_action_space, num_envs)
        self.single_observation_space = gym.spaces.Dict(single_observation_space)
        self.observation_space = batch_space(self.single_observation_space, num_envs)
        self.loss_fn = loss_fn

        self.__sensor_pos_limits = (
            np.concatenate(
                [
                    -np.array(config.cell_size) / 2 + np.array(config.cell_padding),
                    [0.0],
                ]
            ),
            np.concatenate(
                [
                    np.array(config.cell_size) / 2 - np.array(config.cell_padding),
                    [0.02],
                ]
            ),
        )

        # Calculate the maximum distance the sensor can travel in one step
        half_transfer_time = config.transfer_timedelta_s / 2
        acceleration_time = min(
            config.linear_velocity / config.linear_acceleration, half_transfer_time
        )
        max_velocity_time = half_transfer_time - acceleration_time
        self.__max_distance_linear = (
            max_velocity_time * config.linear_velocity
            + 0.5 * config.linear_acceleration * acceleration_time**2
        )

        self.__current_data_points: list[TouchSingle | None] = [None] * num_envs
        self.__current_mesh_data_points: list[MeshDataPoint | None] = [None] * num_envs
        self.__current_object_poses = Transformation.batch_concatenate(
            [Transformation()] * num_envs
        )
        self.__current_touch_idx = np.zeros(num_envs, dtype=np.int_)
        self.__current_sensor_pos = np.zeros((num_envs, 3), dtype=np.float64)
        self.__sensor_z_offset = np.zeros(num_envs, dtype=np.float64)
        self.__current_sensor_target_pos = np.zeros((num_envs, 3), dtype=np.float64)
        self.__current_step = np.zeros(num_envs, dtype=np.int_)
        self.__prev_done = np.zeros(num_envs, dtype=np.bool_)
        self.__last_sensor_images = np.zeros(
            (num_envs, sensor_output_size[1], sensor_output_size[0], 3), dtype=np.uint8
        )
        self.__spec: EnvSpec | None = None

        if config.mesh_dataset is None:
            self.__mesh_datasets = None
            self.__mesh_id_maps = None
        else:
            if isinstance(config.mesh_dataset, MeshDataset):
                self.__mesh_datasets = [config.mesh_dataset] * num_envs
            else:
                assert len(config.mesh_dataset) == num_envs
                self.__mesh_datasets = list(config.mesh_dataset)
            # Map object ids to mesh dataset indices (shared between identical dataset instances)
            id_maps_by_dataset: dict[int, dict[Any, int]] = {}
            self.__mesh_id_maps = []
            for ds in self.__mesh_datasets:
                if id(ds) not in id_maps_by_dataset:
                    id_maps_by_dataset[id(ds)] = {
                        dp_id: i for i, dp_id in enumerate(ds.huggingface_dataset["id"])
                    }
                self.__mesh_id_maps.append(id_maps_by_dataset[id(ds)])

        if self.__mesh_datasets is None or not config.enable_rendering:
            self.__renderer: TactilePerceptionRenderer | None = None
        else:
            # The tactile renderer is only used by TactilePerceptionRenderer to lay out its (disabled) simulated
            # tactile image display, as the tactile images of this environment come from the dataset
            display_sensor = mk_tactile_renderer(renderer_type="depth", backend="numpy")
            depth_map_size = display_sensor.get_desired_depth_map_size(
                sensor_output_size
            )
            mm_per_pixel = tuple(
                GELSIGHT_MINI_SENSOR_SURFACE_SIZE / np.array(depth_map_size) * 1000
            )
            self.__renderer = TactilePerceptionRenderer(
                num_envs,
                display_sensor,
                depth_map_size,
                mm_per_pixel,
                show_viewer=render_mode == "human",
                show_sensor_target_pos=config.show_sensor_target_pos,
                show_tactile_image=False,
                show_class_weights=config.renderer_show_class_weights,
                transparent_background=config.render_transparent_background,
                cell_size=config.cell_size,
                external_camera_resolution=config.renderer_external_camera_resolution,
            )

    @staticmethod
    def __project_sphere(x: np.ndarray, radius: float = 1.0) -> np.ndarray:
        magnitude = np.linalg.norm(x, axis=-1, keepdims=True)
        direction = x / np.maximum(magnitude, radius)
        return np.where(magnitude > radius, direction * radius, x)

    def __select_touch(
        self, i: int, target_pos_xy: np.ndarray, first: bool = False
    ) -> None:
        window_start = 0 if first else int(self.__current_touch_idx[i]) + 1
        window_end = window_start + self.__touch_window_size
        if window_end > self.__available_touches_per_sequence:
            raise ValueError("Window does not fit in remaining data.")
        window = np.arange(window_start, window_end)
        distances = np.linalg.norm(
            self.__current_data_points[i].pos_in_cell[window] - target_pos_xy, axis=-1
        )
        touch_idx = int(window[np.argmin(distances)])
        self.__current_touch_idx[i] = touch_idx
        self.__current_sensor_pos[i] = self.__get_sensor_pose(i, touch_idx).translation

    def __get_sensor_pose(self, i: int, touch_idx: int) -> Transformation:
        pose = self.__current_data_points[i].gel_pose_cell_frame[touch_idx]
        if self.__sensor_z_offset[i] == 0.0:
            return pose
        return Transformation(
            pose.translation + np.array([0.0, 0.0, self.__sensor_z_offset[i]]),
            pose.rotation,
        )

    def __reset_partial(
        self, mask: Sequence[bool], options: dict[str, Any] | None = None
    ):
        if np.any(mask):
            if options is None:
                options = {}
            datapoint_idx = list(options.get("datapoint_idx", [None] * self.num_envs))
            initial_sensor_target_pos = list(
                options.get("initial_sensor_target_pos", [None] * self.num_envs)
            )
            for i in np.where(mask)[0]:
                idx = (
                    self.np_random.integers(0, len(self.__datasets[i]))
                    if datapoint_idx[i] is None
                    else datapoint_idx[i]
                )
                self.__current_data_points[i] = self.__datasets[i][idx]
                if self.__config.recalibrate_sensor_z:
                    # The deepest touch of the round is the one that went down to the platform surface, which
                    # corresponds to a sensor z-position of GEL_PENETRATION_DEPTH_MM in the simulated environments
                    recorded_z = self.__current_data_points[
                        i
                    ].gel_pose_cell_frame.translation[:, 2]
                    self.__sensor_z_offset[i] = (
                        GEL_PENETRATION_DEPTH_MM / 1000 - np.min(recorded_z)
                    )
                if self.__mesh_datasets is not None:
                    object_id = self.__current_data_points[i].object_id
                    if object_id not in self.__mesh_id_maps[i]:
                        raise KeyError(
                            f"Object id {object_id} of touch datapoint "
                            f"{self.__current_data_points[i].id} was not found in the mesh dataset."
                        )
                    self.__current_mesh_data_points[i] = self.__mesh_datasets[i][
                        self.__mesh_id_maps[i][object_id]
                    ]
                if initial_sensor_target_pos[i] is None:
                    if self.__config.randomize_initial_sensor_pose:
                        target_pos_xy = self.np_random.uniform(
                            low=self.__sensor_pos_limits[0][:2],
                            high=self.__sensor_pos_limits[1][:2],
                        )
                    else:
                        target_pos_xy = np.zeros(2)
                else:
                    target_pos_xy = np.asarray(initial_sensor_target_pos[i])[:2]
                self.__current_sensor_target_pos[i] = np.concatenate(
                    [target_pos_xy, [0.0]]
                )
                self.__select_touch(i, target_pos_xy, first=True)
            self.__current_step[mask] = 0

            if self.__mesh_datasets is not None:
                # The actual object poses are unknown, so the objects are shown in a default position: centered in
                # the cell and resting on the platform
                self.__current_object_poses = Transformation.batch_concatenate(
                    [
                        Transformation([0, 0, -np.min(dp.mesh.vertices[:, 2])])
                        for dp in self.__current_mesh_data_points
                    ]
                )
                if self.__renderer is not None:
                    self.__renderer.objects = tuple(self.__current_mesh_data_points)
                    self.__renderer.set_object_poses(self.__current_object_poses)
                    self.__renderer.update_shadow_objects(
                        self.__current_object_poses,
                        shadow_object_visible=np.zeros(self.num_envs, dtype=np.bool_),
                    )

    @abstractmethod
    def _get_prediction_targets(self) -> np.ndarray:
        pass

    def __get_sensor_image(self, i: int) -> np.ndarray:
        dp = self.__current_data_points[i]
        img = np.asarray(dp.sensor_image[int(self.__current_touch_idx[i])])
        w, h = self.__sensor_output_size
        if img.shape[:2] != (h, w):
            img = np.asarray(
                PIL.Image.fromarray(img).resize((w, h), PIL.Image.Resampling.BILINEAR)
            )
        return img

    def __get_obs_info(self) -> tuple[ObsType, dict[str, Any]]:
        for i in range(self.num_envs):
            self.__last_sensor_images[i] = self.__get_sensor_image(i)

        sensor_pos_min, sensor_pos_max = self.__sensor_pos_limits
        sensor_pos_normalized = np.clip(
            (self.__current_sensor_pos - sensor_pos_min)
            / (sensor_pos_max - sensor_pos_min)
            * 2
            - 1,
            -1.0,
            1.0,
        )

        obs = {
            "sensor_pos": sensor_pos_normalized.astype(np.float32),
            "sensor_img": self.__last_sensor_images.astype(np.float32) / 255,
        }

        if self.__config.timeout_behavior == "terminate":
            obs["time_step"] = (
                self.__current_step / self.__config.step_limit * 2 - 1
            ).astype(np.float32)

        sensor_poses = Transformation.batch_concatenate(
            [
                self.__get_sensor_pose(i, int(self.__current_touch_idx[i]))
                for i in range(self.num_envs)
            ]
        )

        if self.__renderer is not None:
            self.__renderer.sensor_poses = sensor_poses
            self.__renderer.sensor_shadow_poses = Transformation(
                self.__current_sensor_target_pos.copy()
            )

        info = {
            "sensor_pose": sensor_poses,
            "touch_idx": self.__current_touch_idx.copy(),
        }
        return obs, info

    def reset(self, *, seed: int | None = None, options: dict[str, Any | None] = None):
        super().reset(seed=seed, options=options)
        self.__current_step = np.zeros(self.num_envs, dtype=np.int_)
        self.__prev_done = np.zeros(self.num_envs, dtype=np.bool_)
        self.__current_data_points = [None] * self.num_envs
        self.__current_mesh_data_points = [None] * self.num_envs
        self.__reset_partial(np.ones(self.num_envs, dtype=np.bool_), options=options)
        return self.__get_obs_info()

    def _step(
        self,
        action: ActType,
        prediction: np.ndarray,
    ):
        targets = self._get_prediction_targets()
        self.__reset_partial(self.__prev_done)

        sensor_target_pos_rel = action["sensor_target_pos_rel"]
        if np.any(np.isnan(sensor_target_pos_rel)):
            raise ValueError("NaN values detected in sensor target position.")
        action_reward = np.mean(
            -(sensor_target_pos_rel**2) * self.__config.action_regularization, axis=-1
        )

        # Project everything back into unit sphere
        sensor_target_pos_rel_clipped = self.__project_sphere(sensor_target_pos_rel)
        sensor_target_pos = np.clip(
            self.__current_sensor_pos
            + sensor_target_pos_rel_clipped * self.__max_distance_linear,
            *self.__sensor_pos_limits,
        )

        active = ~self.__prev_done
        self.__current_sensor_target_pos[active] = sensor_target_pos[active]
        for i in np.where(active)[0]:
            self.__select_touch(i, sensor_target_pos[i, :2])

        self.__current_step[active] += 1
        time_out = self.__current_step >= self.__config.step_limit
        terminated = np.zeros(self.num_envs, dtype=np.bool_)
        truncated = np.zeros(self.num_envs, dtype=np.bool_)
        if self.__config.timeout_behavior == "terminate":
            terminated = time_out
        else:
            truncated |= time_out

        # The episode ends once no full window of touches remains after the current touch
        exhausted = (
            self.__current_touch_idx + 1 + self.__touch_window_size
            > self.__available_touches_per_sequence
        )
        truncated |= exhausted & ~terminated

        obs, info = self.__get_obs_info()

        action_reward = np.where(self.__prev_done, 0, action_reward)
        self.__prev_done = terminated | truncated
        return obs, action_reward, terminated, truncated, info, targets

    def render(self) -> np.ndarray | None:
        if self.__renderer is None:
            return self.__last_sensor_images.copy()
        img = self.__renderer.render_external_cameras()
        if img is None:
            # Human render mode: the viewer displays the scene itself
            return None
        if self.__config.renderer_show_tactile_image:
            # Show the real tactile image of the current touch in the top right corner
            frame_height, frame_width = img.shape[1:3]
            inset_height = int(round(0.3 * frame_height))
            inset_width = int(
                round(
                    inset_height
                    * self.__sensor_output_size[0]
                    / self.__sensor_output_size[1]
                )
            )
            margin = int(round(0.02 * frame_height))
            pos_y = margin
            pos_x = frame_width - inset_width - margin
            for i in range(self.num_envs):
                inset = np.asarray(
                    PIL.Image.fromarray(self.__last_sensor_images[i]).resize(
                        (inset_width, inset_height), PIL.Image.Resampling.NEAREST
                    )
                )
                img[
                    i, pos_y : pos_y + inset_height, pos_x : pos_x + inset_width, :3
                ] = inset
                if img.shape[-1] == 4:
                    img[
                        i, pos_y : pos_y + inset_height, pos_x : pos_x + inset_width, 3
                    ] = 255
        return img

    def close(self):
        if self.__renderer is not None:
            self.__renderer.close()
        super().close()

    @property
    def render_mode(self):
        return self.__render_mode

    @property
    def sensor_pos_limits(self) -> tuple[np.ndarray, np.ndarray]:
        return self.__sensor_pos_limits

    @property
    def current_data_points(self) -> tuple[TouchSingle, ...]:
        return tuple(self.__current_data_points)

    @property
    def current_mesh_data_points(self) -> tuple[MeshDataPoint, ...] | None:
        if self.__mesh_datasets is None:
            return None
        return tuple(self.__current_mesh_data_points)

    @property
    def current_object_poses(self) -> Transformation:
        return self.__current_object_poses

    @property
    def current_touch_indices(self) -> np.ndarray:
        return self.__current_touch_idx.copy()

    @property
    def current_sensor_z_offsets(self) -> np.ndarray:
        """Offsets added to the recorded gel z-positions of the current rounds (see recalibrate_sensor_z)."""
        return self.__sensor_z_offset.copy()

    @property
    def spec(self) -> EnvSpec | None:
        return self.__spec

    @spec.setter
    def spec(self, spec: EnvSpec):
        spec = copy.copy(spec)
        spec.max_episode_steps = self.__config.step_limit
        self.__spec = spec

    @property
    def config(self) -> TactileRealSnapConfig:
        return self.__config

    @property
    def _prev_done(self) -> np.ndarray:
        return self.__prev_done.copy()

    @property
    def _renderer(self) -> TactilePerceptionRenderer | None:
        return self.__renderer


class TactileRealSnapClassificationVectorEnv(
    TactileRealSnapVectorEnv[np.ndarray, np.ndarray]
):
    def __init__(
        self,
        config: TactileRealSnapConfig,
        num_envs: int,
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
    ):
        if isinstance(config.dataset, TouchSingleDataset):
            datasets = [config.dataset] * num_envs
        else:
            assert len(config.dataset) == num_envs
            datasets = list(config.dataset)

        try:
            label_names = tuple(datasets[0].label_names)
            assert all(tuple(ds.label_names) == label_names for ds in datasets[1:])
        except AttributeError:
            # The label feature does not carry class names, so we derive them from the label values instead
            num_classes = (
                max(int(np.max(ds.huggingface_dataset["label"])) for ds in datasets) + 1
            )
            label_names = tuple(str(i) for i in range(num_classes))
        self.__label_names = label_names

        super().__init__(
            config,
            num_envs,
            single_prediction_space=LogitSpace(
                -np.inf, np.inf, shape=(len(label_names),)
            ),
            single_prediction_target_space=gym.spaces.Discrete(len(label_names)),
            loss_fn=CrossEntropyLossFn(num_classes=len(label_names)).normalized,
            render_mode=render_mode,
        )

    def reset(self, *, seed: int | None = None, options: dict[str, Any | None] = None):
        obs, info = super().reset(seed=seed, options=options)
        if self._renderer is not None:
            self._renderer.class_weights = np.zeros(
                (self.num_envs, self.single_prediction_space.shape[-1])
            )
            self._renderer.target_class_idx = self._get_prediction_targets()
        return obs, info

    def _step(
        self,
        action: ActType,
        prediction: np.ndarray,
    ):
        if self._renderer is not None:
            self._renderer.class_weights = scipy.special.softmax(prediction, axis=-1)
            self._renderer.class_weights[self._prev_done] = np.zeros(
                self.single_prediction_space.shape[-1]
            )
        obs, action_reward, terminated, truncated, info, labels = super()._step(
            action, prediction
        )
        if self._renderer is not None:
            self._renderer.target_class_idx = labels
        return obs, action_reward, terminated, truncated, info, labels

    def _get_prediction_targets(self) -> np.ndarray:
        return np.array([dp.label for dp in self.current_data_points])

    @property
    def label_names(self) -> tuple[str, ...]:
        return self.__label_names


class TactileRealSnapVolumeEstimationVectorEnv(
    TactileRealSnapVectorEnv[np.ndarray, np.ndarray]
):
    def __init__(
        self,
        config: TactileRealSnapConfig,
        num_envs: int,
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        renderer_show_shadow_objects: bool = True,
    ):
        if config.mesh_dataset is None:
            raise ValueError(
                "TactileRealSnapVolumeEstimationVectorEnv requires config.mesh_dataset to be set, as the "
                "prediction targets are computed from the object meshes."
            )
        # Avoid a circular import
        from .tactile_volume_estimation_env import _compute_object_volume_idx

        if isinstance(config.mesh_dataset, MeshDataset):
            mesh_dataset = config.mesh_dataset
        else:
            mesh_dataset = config.mesh_dataset[0]
        statistics = get_dataset_stats(
            mesh_dataset, "volume", _compute_object_volume_idx
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

    def reset(self, *, seed: int | None = None, options: dict[str, Any | None] = None):
        self.__metrics = defaultdict(
            lambda: tuple(deque() for _ in range(self.num_envs))
        )
        return super().reset(seed=seed, options=options)

    def _step(
        self,
        action: ActType,
        prediction: np.ndarray,
    ):
        target_volume = self.__get_object_volumes()
        predicted_volume = prediction * self.__std_volume + self.__mean_volume
        relative_error = np.maximum(predicted_volume, 0) / np.maximum(
            target_volume, 1e-10
        )
        abs_error = np.abs(predicted_volume - target_volume)

        prev_done = self._prev_done
        for i in range(self.num_envs):
            if prev_done[i]:
                self.__metrics["abs_error_cm3"][i].clear()
                self.__metrics["rel_error"][i].clear()
            else:
                # Append scalars, as environments may finish at different times and update_info_metrics_vec cannot
                # handle mixing arrays and NaNs
                self.__metrics["abs_error_cm3"][i].append(
                    float(abs_error[i, 0]) * 100**3
                )
                self.__metrics["rel_error"][i].append(float(relative_error[i, 0]))

        obs, action_reward, terminated, truncated, info, targets = super()._step(
            action, prediction
        )

        if self._renderer is not None and self.__renderer_show_shadow_objects:
            # Do that after the step as new objects might be loaded
            self._renderer.update_shadow_objects(
                self.current_object_poses,
                new_shadow_object_scales=relative_error,
                shadow_object_visible=~self._prev_done,
            )

        if np.any(terminated | truncated):
            info = update_info_metrics_vec(info, self.__metrics, terminated | truncated)

        return obs, action_reward, terminated, truncated, info, targets

    def __normalize_volume(self, volume: np.ndarray | float) -> np.ndarray | float:
        return (volume - self.__mean_volume) / self.__std_volume

    def _get_prediction_targets(self) -> np.ndarray:
        return self.__normalize_volume(self.__get_object_volumes())

    def __get_object_volumes(self) -> np.ndarray:
        return np.array(
            [dp.mesh.volume for dp in self.current_mesh_data_points],
            dtype=np.float32,
        )[..., None]


def TactileRealSnapClassificationEnv(
    config: TactileRealSnapConfig,
    render_mode: Literal["rgb_array", "human"] = "rgb_array",
) -> ActivePerceptionVectorToSingleWrapper[ObsType, ActType, np.ndarray, np.ndarray]:
    return ActivePerceptionVectorToSingleWrapper(
        TactileRealSnapClassificationVectorEnv(
            config,
            1,
            render_mode=render_mode,
        )
    )


def TactileRealSnapVolumeEstimationEnv(
    config: TactileRealSnapConfig,
    render_mode: Literal["rgb_array", "human"] = "rgb_array",
    renderer_show_shadow_objects: bool = True,
) -> ActivePerceptionVectorToSingleWrapper[ObsType, ActType, np.ndarray, np.ndarray]:
    return ActivePerceptionVectorToSingleWrapper(
        TactileRealSnapVolumeEstimationVectorEnv(
            config,
            1,
            render_mode=render_mode,
            renderer_show_shadow_objects=renderer_show_shadow_objects,
        )
    )

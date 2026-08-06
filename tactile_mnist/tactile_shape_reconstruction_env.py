from __future__ import annotations

import logging
import math
from collections import deque, defaultdict
from typing import (
    Literal,
    TYPE_CHECKING,
    Any,
)

import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import norm
from transformation import Transformation

from ap_gym import ActivePerceptionVectorToSingleWrapper
from ap_gym.util import update_info_metrics_vec
from cod_vae import CODVAE, pack_cube_transform, points_to_cube_transform
from .cod_vae_loss_fn import CODVAEReconstructionLossFn
from .mesh_dataset import MeshDataset
from .tactile_perception_renderer import MESH_INVISIBLE
from .tactile_perception_vector_env import (
    TactilePerceptionVectorEnv,
    TactilePerceptionConfig,
    TransformedDataPoint,
    ActType,
)

if TYPE_CHECKING:
    from .tactile_perception_vector_env import ObsType

logger = logging.getLogger(__name__)


class TactileShapeReconstructionVectorEnv(
    TactilePerceptionVectorEnv[np.ndarray, dict[str, np.ndarray]],
):
    """
    Active tactile shape reconstruction environment.

    The agent has to reconstruct the geometry of the touched object in the platform frame. Its prediction is a
    COD-VAE *full latent* (https://github.com/TimSchneider42/cod-vae, see cod_vae.CODVAEBase.pack_full_latent) in the
    platform frame normalized to [-1, 1] by frame_half_size = max(cell_size) / 2, i.e. everything needed to
    reconstruct a completely unknown object:

    - latent (num_latents * latent_dim entries): a COD-VAE latent describing the object's shape and orientation in
      the model's [-1, 1] cube.
    - bounding box center (3 entries): the center of the posed object's axis-aligned bounding box in the platform
      frame, divided by frame_half_size. As objects rest on the platform, the normalized z component is always
      positive.
    - bounding box size (1 entry): the maximum half-extent of the posed object's axis-aligned bounding box, divided
      by frame_half_size. A single scalar suffices because COD-VAE's cube normalization is isotropic; the aspect
      ratios of the bounding box are part of the shape encoded in the latent.

    Since the prediction is a regular COD-VAE full latent, it can be decoded back into a mesh in the platform frame
    with the same COD-VAE model (see cod_vae.CODVAEBase.decode_mesh_full with frame_half_size, or decode_full for
    occupancy queries).

    The prediction target does not prescribe a particular latent; it identifies the ground-truth geometry instead: a
    dict with the index of the object's mesh in the dataset ("mesh_index"), the pose mapping the raw dataset mesh
    into the platform frame ("position" and scalar-last "quaternion"; any mesh pre-processing rotation, e.g.
    smallest_dimension_up, is composed into the quaternion), and the posed object's ground-truth bounding box
    ("box": normalized center and maximum half-extent, computed with the exact functions COD-VAE's encoding uses and
    thus directly comparable to the prediction's last four entries). The loss is the loss COD-VAE is trained with
    (see CODVAEReconstructionLossFn): binary cross entropy of the occupancy decoded from the prediction against the
    ground-truth mesh occupancy, evaluated at volume query points re-sampled on every evaluation from the mesh's
    slice of a shared uniform point database plus a fixed per-mesh set of near-surface points, following the sdf_gen
    recipe COD-VAE is trained on. The query points are mapped through the object's pose and the predicted cube
    transform, so errors in the predicted bounding box center and size directly manifest as occupancy errors.
    Additionally, the loss penalizes the mean squared error between the predicted and the ground-truth bounding box
    parameters (weight box_coeff). This term is the bounding box's only gradient source — the decoder's own box
    gradient is always stopped, as it is noisy and vanishes when the predicted box does not overlap the object —
    and provides a smooth localization gradient at any distance.

    If renderer_show_shadow_objects is set, the current prediction (latent and cube transform) is decoded back into
    a mesh and rendered as a translucent shadow object. This is disabled by default, as it requires an additional
    COD-VAE decoder pass per step.

    If half_precision is set (the default), the COD-VAE model is loaded in float16, which roughly halves the memory
    footprint of a loss evaluation and doubles its throughput on a GPU, at a small gradient-fidelity cost. As the
    precision is a property of the model, it applies uniformly to the reconstruction loss and to the shadow-object
    reconstruction.
    """

    def __init__(
        self,
        config: TactilePerceptionConfig,
        num_envs: int,
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        model: str = "TimSchneider42/cod-vae-16x8",
        backend: Literal["auto", "torch", "jax"] = "auto",
        device: str | None = None,
        latent_bound: float = 8.0,
        renderer_show_shadow_objects: bool = True,
        shadow_object_resolution: int = 64,
        half_precision: bool = True,
        loss_fn_kwargs: dict[str, Any] | None = None,
    ):
        if not isinstance(config.dataset, MeshDataset):
            raise ValueError(
                "TactileShapeReconstructionVectorEnv requires a single dataset shared "
                "by all environments, as the prediction targets reference meshes by "
                "their index in it."
            )
        self.__vae = CODVAE.from_pretrained(
            model,
            backend=backend,
            device=device,
            dtype="float16" if half_precision else None,
        )
        dims = self.__vae.config.num_latents * self.__vae.config.latent_dim
        self.__latent_dims = dims

        # cell_size is 2D; the larger cell half-extent serves as the common length
        # scale for all three axes.
        self.__frame_half_size = float(np.max(config.cell_size) / 2)
        self.__object_scale = 0.9

        max_expected_translation_perturbation_norm = 0.0
        if config.perturb_object_pose:
            cumulative_std = config.translation_perturbation_scale * math.sqrt(
                config.step_limit
            )
            # For all objects starting directly at the edge of the platform, 99.99% of the time they will stay within
            # this bound.
            max_expected_translation_perturbation_norm = (
                cumulative_std * norm.ppf(0.9999) / self.__frame_half_size
            )

        # COD-VAE latents are KL-regularized towards a standard normal, so they are approximately unit-scale by
        # construction and are predicted without further normalization. The normalized bounding box center lies
        # within the normalized cell extents in x/y up to pose perturbation drift; its z component and the normalized
        # bounding box size are positive and bounded by 2 with generous headroom (rotated objects can have an
        # axis-aligned bounding box exceeding the cell half-extent).
        center_xy_bound = (
            np.array(config.cell_size, dtype=np.float32) / 2 / self.__frame_half_size
            + max_expected_translation_perturbation_norm
        )
        low = np.concatenate(
            [
                np.full(dims, -latent_bound, dtype=np.float32),
                np.array(
                    [-center_xy_bound[0], -center_xy_bound[1], 0.0], dtype=np.float32
                ),
                np.array([0.0], dtype=np.float32),
            ]
        )
        high = np.concatenate(
            [
                np.full(dims, latent_bound, dtype=np.float32),
                np.array(
                    [center_xy_bound[0], center_xy_bound[1], 2.0], dtype=np.float32
                ),
                np.array([2.0], dtype=np.float32),
            ]
        )
        pred_space = gym.spaces.Box(low, high, dtype=np.float32)
        target_space = gym.spaces.Dict(
            {
                "mesh_index": gym.spaces.Box(
                    0, len(config.dataset) - 1, shape=(), dtype=np.int64
                ),
                "position": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                "quaternion": gym.spaces.Box(-1.0, 1.0, shape=(4,)),
                "box": gym.spaces.Box(-np.inf, np.inf, shape=(4,)),
            }
        )

        super().__init__(
            config,
            num_envs,
            single_prediction_space=pred_space,
            single_prediction_target_space=target_space,
            loss_fn=CODVAEReconstructionLossFn(
                self.__vae,
                dataset=config.dataset,
                frame_half_size=self.__frame_half_size,
                object_scale=self.__object_scale,
                **(loss_fn_kwargs or {}),
            ).normalized,
            render_mode=render_mode,
        )

        self.__renderer_show_shadow_objects = renderer_show_shadow_objects
        self.__shadow_object_resolution = shadow_object_resolution
        self.__metrics: dict[str, tuple[deque[float], ...]] | None = None
        self.__hull_cache: dict[Any, np.ndarray] = {}

        if self.__vae.backend == "jax":
            # As with Taxim (see TactilePerceptionVectorEnv.__init__), JITing COD-VAE inside a host callback
            # deadlocks, so we trigger the compilation here. Since JAX recompiles for every new batch size, all
            # decoding uses batches of exactly num_envs elements with query batches padded to a fixed chunk size
            # (see cod_vae.CODVAEBase.decode_full), so a single warmup call per function covers all shapes the model
            # will ever see. The loss uses the full-latent decode path and decode_mesh_full additionally uses the
            # plain logits decode; the encoder is not used by this environment.
            dummy_full = np.zeros(
                (num_envs, self.__vae.full_latent_size), dtype=np.float32
            )
            dummy_queries = np.zeros((num_envs, 1, 3), dtype=np.float32)
            self.__vae.decode_full(dummy_full, dummy_queries)
            self.__vae.decode_mesh_full(
                dummy_full,
                resolution=self.__shadow_object_resolution,
                frame_half_size=self.__frame_half_size,
            )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any | None] = None
    ) -> tuple[ObsType, dict[str, Any]]:
        self.__metrics = defaultdict(
            lambda: tuple(deque() for _ in range(self.num_envs))
        )
        return super().reset(seed=seed, options=options)

    def __posed_boxes(self) -> np.ndarray:
        """
        The ground-truth normalized bounding boxes (B, 4) of the current objects in
        the platform frame: [center (3), size (1)] divided by frame_half_size,
        computed with the exact functions COD-VAE's encoding uses
        (cod_vae.points_to_cube_transform / pack_cube_transform) from the objects'
        (cached) convex hull vertices, which have the same bounding box as the meshes.
        """
        keys = [dp.id for dp in self.current_data_points]
        for key, dp in zip(keys, self.current_data_points):
            if key not in self.__hull_cache:
                self.__hull_cache[key] = np.asarray(
                    dp.mesh.convex_hull.vertices, dtype=np.float64
                )
        # Keep only the hulls of the current objects.
        self.__hull_cache = {key: self.__hull_cache[key] for key in keys}
        boxes = np.empty((self.num_envs, 4), dtype=np.float32)
        for i, (key, pose) in enumerate(
            zip(keys, self.current_object_poses_platform_frame)
        ):
            transform = points_to_cube_transform(
                pose.transform(self.__hull_cache[key]), self.__object_scale
            )
            boxes[i] = pack_cube_transform(
                transform,
                frame_half_size=self.__frame_half_size,
                object_scale=self.__object_scale,
            )
        return boxes

    def _step(
        self,
        action: dict[str, np.ndarray],
        prediction: np.ndarray,
    ):
        prev_done = np.array(self._prev_done)

        dims = self.__latent_dims
        # Bounding box center and size errors in meters.
        gt_boxes = self.__posed_boxes()
        center_error = (
            np.linalg.norm(prediction[:, dims : dims + 3] - gt_boxes[:, :3], axis=-1)
            * self.__frame_half_size
        )
        size_error = (
            np.abs(prediction[:, dims + 3] - gt_boxes[:, 3]) * self.__frame_half_size
        )
        step_metrics = {"center_error": center_error, "size_error": size_error}

        for i in range(self.num_envs):
            for name, values in step_metrics.items():
                if prev_done[i]:
                    self.__metrics[name][i].clear()
                else:
                    self.__metrics[name][i].append(values[i])

        obs, action_reward, terminated, truncated, info, targets = super()._step(
            action, prediction
        )

        if self.__renderer_show_shadow_objects:
            # Do that after the step as new objects might be loaded. The decoded meshes are mapped into the platform
            # frame via the bounding box center and size predicted by the agent, hence the shadow objects are placed
            # at the platform frame's origin.
            prediction_clipped = np.clip(
                prediction,
                self.single_prediction_space.low,
                self.single_prediction_space.high,
            )

            def reconstruct_meshes():
                reconstructions: list[Any] = [None] * self.num_envs
                active = np.where(~prev_done)[0]
                if len(active) > 0:
                    if self.__vae.backend == "jax":
                        # Decode the full batch so the jitted decoder only ever sees a single batch size and is never
                        # recompiled inside a host callback (see __init__). Inactive results are discarded below.
                        decode_idx = np.arange(self.num_envs)
                    else:
                        decode_idx = active
                    decoded = self.__vae.decode_mesh_full(
                        prediction_clipped[decode_idx],
                        resolution=self.__shadow_object_resolution,
                        frame_half_size=self.__frame_half_size,
                    )
                    active_set = set(active)
                    for i, mesh in zip(decode_idx, decoded):
                        if i in active_set:
                            reconstructions[i] = mesh if len(mesh.faces) > 0 else MESH_INVISIBLE
                return reconstructions
            self._renderer.update_shadow_objects(
                Transformation.batch_concatenate(
                    [Transformation()] * self.num_envs,
                ),
                new_shadow_object_meshes=reconstruct_meshes,
                shadow_object_visible=(True,) * self.num_envs
            )

        if np.any(terminated | truncated):
            info = update_info_metrics_vec(info, self.__metrics, terminated | truncated)

        return obs, action_reward, terminated, truncated, info, targets

    def _get_prediction_targets(self) -> dict[str, np.ndarray]:
        # The target pose maps the *raw* dataset mesh into the platform frame: the
        # loss' occupancy pools know nothing about the environment's mesh
        # pre-processing, so its rotation (see _pre_process_dp) is composed into the
        # pose here.
        poses = self.current_object_poses_platform_frame
        quaternions = np.stack(
            [
                (
                    Rotation.from_quat(quaternion) * dp.applied_rotation
                ).as_quat()
                if isinstance(dp, TransformedDataPoint)
                else quaternion
                for quaternion, dp in zip(poses.quaternion, self.current_data_points)
            ]
        )
        return {
            "mesh_index": np.asarray(self.current_data_point_indices, dtype=np.int64),
            "position": np.asarray(poses.translation, dtype=np.float32),
            "quaternion": np.asarray(quaternions, dtype=np.float32),
            "box": self.__posed_boxes(),
        }

    @property
    def vae(self):
        """The COD-VAE model used to decode predictions."""
        return self.__vae

    @property
    def frame_half_size(self) -> float:
        """Half-extent (in meters) of the [-1, 1] normalized platform frame of the predictions."""
        return self.__frame_half_size


def TactileShapeReconstructionEnv(
    config: TactilePerceptionConfig,
    render_mode: Literal["rgb_array", "human"] = "rgb_array",
    model: str = "TimSchneider42/cod-vae-16x8",
    backend: Literal["auto", "torch", "jax"] = "auto",
    device: str | None = None,
    latent_bound: float = 8.0,
    renderer_show_shadow_objects: bool = False,
    shadow_object_resolution: int = 64,
    half_precision: bool = True,
    loss_fn_kwargs: dict[str, Any] | None = None,
) -> ActivePerceptionVectorToSingleWrapper[
    "ObsType", ActType, np.ndarray, dict[str, np.ndarray]
]:
    return ActivePerceptionVectorToSingleWrapper(
        TactileShapeReconstructionVectorEnv(
            config,
            1,
            render_mode=render_mode,
            model=model,
            backend=backend,
            device=device,
            latent_bound=latent_bound,
            renderer_show_shadow_objects=renderer_show_shadow_objects,
            shadow_object_resolution=shadow_object_resolution,
            half_precision=half_precision,
            loss_fn_kwargs=loss_fn_kwargs,
        )
    )

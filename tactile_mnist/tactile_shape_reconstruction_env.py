from __future__ import annotations

import logging
import math
from collections import deque, defaultdict
from functools import partial
from typing import (
    Literal,
    TYPE_CHECKING,
    Any,
)

import filelock
import gymnasium as gym
import numpy as np
import tqdm
from scipy.spatial.transform import Rotation
from scipy.stats import norm
from transformation import Transformation

from ap_gym import ActivePerceptionVectorToSingleWrapper
from ap_gym.util import update_info_metrics_vec
from cod_vae import (
    CODVAE,
    CODVAEBase,
    CubeTransform,
    pack_cube_transform,
    points_to_cube_transform,
    sample_surface_points,
)
from .cod_vae_loss_fn import CODVAEReconstructionLossFn
from .constants import CACHE_BASE_DIR
from .mesh_dataset import MeshDataset
from .prefetched_dataset import PrefetchedDataset
from .tactile_perception_renderer import MESH_INVISIBLE
from .tactile_perception_vector_env import (
    TactilePerceptionVectorEnv,
    TactilePerceptionConfig,
    TransformedDataPoint,
    ActType,
)
from .util import get_cache_hash

if TYPE_CHECKING:
    from .tactile_perception_vector_env import ObsType

logger = logging.getLogger(__name__)


def _r2_sequence(count: int) -> np.ndarray:
    """
    The first ``count`` points of the R2 Kronecker sequence (Roberts, 2018): a
    deterministic low-discrepancy coverage of the unit square. Used instead of rng
    draws where uniform samples are needed, as it is immune to random stream changes
    across library versions (numpy does not guarantee Generator stream stability)
    and covers the square more evenly than iid draws.
    """
    plastic = 1.324717957244746  # The real solution of x**3 = x + 1.
    alpha = np.array([1 / plastic, 1 / plastic**2])
    return (0.5 + np.arange(1, count + 1)[:, None] * alpha) % 1.0


def _compute_prediction_target_stats(
    dataset: MeshDataset,
    vae: CODVAEBase,
    loss_fn: CODVAEReconstructionLossFn,
    config: TactilePerceptionConfig,
    frame_half_size: float,
    object_scale: float,
    num_rotation_samples: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """
    Per-dimension statistics (mean, std, min, max) of the full-latent prediction
    targets over the dataset and the initial object pose distribution, plus an
    empirical estimate of the occupancy loss terms' expected value under blind
    guessing.

    Analogously to the pose estimation environment's target statistics, the initial
    pose distribution of every object is covered by a linspace over the rotation
    perturbation, with each rotation evaluated at both translation extremes. COD-VAE
    latents are invariant to translation (the cube normalization removes it), so one
    encoder pass per distinct rotation suffices, batched over the rotations of one
    object at a time (which also keeps the encode batch size constant for jit-based
    backends). The mesh surface is sampled only once per object: uniform surface
    sampling commutes with the poses and cube normalizations (all similarity
    transforms), so each pose transforms the same seeded point cloud and normalizes
    it with the exact cube transform of its posed hull, reproducing what
    :meth:`cod_vae.CODVAEBase.encode_mesh_full` computes without re-sampling per
    pose. The bounding box center z and size are equally translation-invariant
    (rotations are about the z axis only), while the box center x/y at a fixed
    rotation is uniform between its values at the two translation extremes, so its
    moments follow analytically like the pose environment's translation targets.

    The blind-guessing estimate is a heuristic: no analytic bound is useful for the
    BCE of a decoded occupancy field (see the cod_vae_loss_fn module documentation),
    so the best uninformed static prediction we can construct — the mean latent and
    mean bounding box — is scored with the occupancy-only loss against every sampled
    rotation of every object. Unlike the moments of the box targets, the occupancy
    loss is not analytic in the translation, and evaluating it at the translation
    extremes would overstate it (the cell corners are the worst case for a
    mean-centered box prediction), so the blind poses instead take their translations
    from a deterministic low-discrepancy sequence covering the placement square (see
    _r2_sequence), matching the environment's uniform placement in distribution. The
    bounding box MSE term is excluded here; its blind-guessing expectation follows
    exactly from the box target variance.
    """
    dims = vae.config.num_latents * vae.config.latent_dim
    if config.randomize_initial_object_pose:
        rotation_norm = np.repeat(np.linspace(0.0, 1.0, num_rotation_samples), 2)
        translation_norm = np.tile(
            np.array([[0.0, 0.0], [1.0, 1.0]]), (num_rotation_samples, 1)
        )
    else:
        # _get_random_object_pose_batch returns the single deterministic pose.
        rotation_norm = translation_norm = None

    latent_sum = np.zeros(dims)
    latent_sq_sum = np.zeros(dims)
    latent_min = np.full(dims, np.inf)
    latent_max = np.full(dims, -np.inf)
    box_sum = np.zeros(4)
    box_sq_sum = np.zeros(4)
    box_min = np.full(4, np.inf)
    box_max = np.full(4, -np.inf)
    latent_count = box_count = 0
    blind_targets = []
    blind_translation_norm = _r2_sequence(2 * num_rotation_samples)

    # The prefetcher loads the upcoming meshes in a background thread while the current one is being encoded.
    with PrefetchedDataset(
        dataset, capacity=3, load_fn=lambda dp: dp.mesh
    ) as prefetched_dataset:
        prefetched_dataset.prefetch(range(len(dataset)))
        for mesh_index in tqdm.trange(len(dataset), desc="Encoding pose samples"):
            dp = TactilePerceptionVectorEnv._pre_process_dp(
                prefetched_dataset[mesh_index],
                smallest_dimension_up=config.smallest_dimension_up,
            )
            sample_poses = partial(
                TactilePerceptionVectorEnv._get_random_object_pose_batch,
                dp,
                config.randomize_initial_object_pose,
                config.max_initial_angle_perturbation,
                config.cell_size,
                object_placement_margin=config.object_placement_margin,
                rotation_perturbation_norm=rotation_norm,
            )
            hull = np.asarray(dp.mesh.convex_hull.vertices, dtype=np.float64)

            def cube_transforms(poses) -> list[CubeTransform]:
                return [
                    points_to_cube_transform(pose.transform(hull), object_scale)
                    for pose in poses
                ]

            def pack_boxes(transforms: list[CubeTransform]) -> np.ndarray:
                return np.stack(
                    [
                        pack_cube_transform(
                            transform,
                            frame_half_size=frame_half_size,
                            object_scale=object_scale,
                        )
                        for transform in transforms
                    ]
                ).astype(np.float64)

            poses = sample_poses(translation_perturbation_norm=translation_norm)
            transforms = cube_transforms(poses)
            boxes = pack_boxes(transforms)
            poses_list = list(poses)
            if config.randomize_initial_object_pose:
                center_low, center_high = boxes[0::2, :2], boxes[1::2, :2]
                unique_boxes = boxes[0::2]
                unique_poses = poses_list[0::2]
                unique_transforms = transforms[0::2]
            else:
                center_low = center_high = boxes[:, :2]
                unique_boxes = boxes
                unique_poses = poses_list
                unique_transforms = transforms
            # Box center xy is uniform between the two translation extremes; z and size
            # do not depend on the translation at all.
            center_mean = (center_low + center_high) / 2
            box_sum[:2] += center_mean.sum(0)
            box_sq_sum[:2] += (
                center_mean**2 + (center_high - center_low) ** 2 / 12
            ).sum(0)
            box_sum[2:] += unique_boxes[:, 2:].sum(0)
            box_sq_sum[2:] += (unique_boxes[:, 2:] ** 2).sum(0)
            box_min[:2] = np.minimum(box_min[:2], center_low.min(0))
            box_max[:2] = np.maximum(box_max[:2], center_high.max(0))
            box_min[2:] = np.minimum(box_min[2:], unique_boxes[:, 2:].min(0))
            box_max[2:] = np.maximum(box_max[2:], unique_boxes[:, 2:].max(0))
            box_count += len(unique_boxes)

            # 2048 points is encode_mesh's default surface sample size.
            surface_points = sample_surface_points(dp.mesh, 2048, seed=seed)
            clouds = np.stack(
                [
                    transform.apply(pose.transform(surface_points))
                    for pose, transform in zip(unique_poses, unique_transforms)
                ]
            ).astype(np.float32)
            latents = vae.encode(clouds).reshape(len(clouds), dims).astype(np.float64)
            latent_sum += latents.sum(0)
            latent_sq_sum += (latents**2).sum(0)
            latent_min = np.minimum(latent_min, latents.min(0))
            latent_max = np.maximum(latent_max, latents.max(0))
            latent_count += len(clouds)

            # The blind poses reuse the rotation linspace but take their translations
            # from the low-discrepancy sequence (see the docstring). The sequence is the
            # same for every object, but the actual translations are not, as the
            # placement bounds depend on the object's rotated silhouette.
            if config.randomize_initial_object_pose:
                blind_poses = sample_poses(
                    translation_perturbation_norm=blind_translation_norm
                )
                blind_boxes = pack_boxes(cube_transforms(blind_poses))
            else:
                blind_poses, blind_boxes = poses, boxes
            # The loss targets' pose refers to the raw dataset mesh (see
            # _get_prediction_targets), so the pre-processing rotation is composed in.
            quaternions = np.asarray(blind_poses.quaternion, dtype=np.float64)
            if isinstance(dp, TransformedDataPoint):
                quaternions = (
                    Rotation.from_quat(quaternions) * dp.applied_rotation
                ).as_quat()
            blind_targets.append(
                (
                    np.asarray(blind_poses.translation, dtype=np.float32),
                    quaternions.astype(np.float32),
                    blind_boxes.astype(np.float32),
                )
            )

    latent_mean = latent_sum / latent_count
    latent_std = np.sqrt(np.maximum(latent_sq_sum / latent_count - latent_mean**2, 0))
    box_mean = box_sum / box_count
    box_std = np.sqrt(np.maximum(box_sq_sum / box_count - box_mean**2, 0))

    mean_prediction = np.concatenate([latent_mean, box_mean]).astype(np.float32)
    rng = np.random.default_rng([seed, 2])
    occupancy_sum = 0.0
    for mesh_index, (position, quaternion, box) in enumerate(
        tqdm.tqdm(blind_targets, desc="Scoring the blind guess")
    ):
        batch_size = len(position)
        occupancy_sum += float(
            loss_fn.numpy(
                np.repeat(mean_prediction[None], batch_size, axis=0),
                {
                    "mesh_index": np.full(batch_size, mesh_index, dtype=np.int64),
                    "position": position,
                    "quaternion": quaternion,
                    "box": box,
                },
                (batch_size,),
                rng=rng,
                occupancy_only=True,
            ).mean()
        )

    return {
        "latent_mean": latent_mean.astype(np.float32),
        "latent_std": latent_std.astype(np.float32),
        "latent_min": latent_min.astype(np.float32),
        "latent_max": latent_max.astype(np.float32),
        "box_mean": box_mean.astype(np.float32),
        "box_std": box_std.astype(np.float32),
        "box_min": box_min.astype(np.float32),
        "box_max": box_max.astype(np.float32),
        "occupancy_blind_guessing_expected_value": np.float32(
            occupancy_sum / len(blind_targets)
        ),
    }


def _load_or_compute_prediction_target_stats(
    dataset: MeshDataset,
    vae: CODVAEBase,
    loss_fn: CODVAEReconstructionLossFn,
    model: str,
    config: TactilePerceptionConfig,
    frame_half_size: float,
    object_scale: float,
    loss_fn_kwargs: dict[str, Any] | None,
    num_rotation_samples: int = 16,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """
    Compute the prediction target statistics (see _compute_prediction_target_stats)
    or load them from the npz cache keyed by the dataset fingerprint and all
    content-determining parameters. Unlike the pose environment's statistics, they
    cannot be computed in worker processes, as they require the COD-VAE model (and
    typically its GPU); the encoder batching keeps the single-process pass fast.
    """
    # These loss parameters only affect where query data lives, not any results.
    content_loss_kwargs = {
        key: value
        for key, value in (loss_fn_kwargs or {}).items()
        if key not in ("max_pool_vram_fraction", "max_device_cached_pools")
    }
    kwargs = dict(
        model=model,
        dtype=str(vae.dtype),
        num_rotation_samples=num_rotation_samples,
        seed=seed,
        frame_half_size=frame_half_size,
        object_scale=object_scale,
        randomize_initial_object_pose=config.randomize_initial_object_pose,
        max_initial_angle_perturbation=config.max_initial_angle_perturbation,
        cell_size=tuple(config.cell_size),
        smallest_dimension_up=config.smallest_dimension_up,
        object_placement_margin=config.object_placement_margin,
        loss_fn_kwargs=tuple(sorted(content_loss_kwargs.items())),
    )
    cache_dir = CACHE_BASE_DIR / "cod_vae_prediction_target_stats"
    cache_dir.mkdir(parents=True, exist_ok=True)
    ds_fingerprint = dataset.huggingface_dataset._fingerprint
    cache_file = cache_dir / f"{get_cache_hash(ds_fingerprint, kwargs)}.npz"
    with filelock.FileLock(cache_dir / f"{ds_fingerprint}.lock"):
        if cache_file.exists():
            try:
                with np.load(cache_file) as data:
                    return dict(data)
            except Exception as ex:
                logger.warning(
                    f"Loading the prediction target statistics from cache failed "
                    f"with the following exception: {ex}"
                )
        print(
            "Computing COD-VAE prediction target statistics (the results will be "
            "cached)..."
        )
        stats = _compute_prediction_target_stats(
            dataset,
            vae,
            loss_fn,
            config,
            frame_half_size,
            object_scale,
            num_rotation_samples,
            seed,
        )
        np.savez(cache_file, **stats)
        return stats


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

    The prediction space bounds and the loss normalization are derived from per-dataset target statistics that are
    computed once on first use and cached on disk (analogously to the pose estimation environment): the initial pose
    distribution of every object is covered by a linspace over the rotation perturbation, each sampled pose's target
    (the COD-VAE embedding of the posed mesh and its normalized bounding box) is computed, and per-dimension mean,
    std, min, and max are accumulated (see prediction_target_stats). The prediction space is bounded by the observed
    per-dimension extremes, widened by a small margin plus the expected pose perturbation drift in the bounding box
    center xy. The reported loss is normalized by the expected loss of blind guessing, so an uninformed prediction
    scores around 1 and a perfect one 0: the bounding box
    MSE term's blind-guessing expectation follows analytically from the box target variance, while the occupancy
    terms' expectation is estimated empirically by scoring the best uninformed static prediction — the mean latent
    and mean bounding box — against every sampled pose of every object (a heuristic; no useful analytic bound exists
    for the BCE of a decoded occupancy field).

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

        loss_fn = CODVAEReconstructionLossFn(
            self.__vae,
            dataset=config.dataset,
            frame_half_size=self.__frame_half_size,
            object_scale=self.__object_scale,
            **(loss_fn_kwargs or {}),
        )
        stats = _load_or_compute_prediction_target_stats(
            config.dataset,
            self.__vae,
            loss_fn,
            model,
            config,
            self.__frame_half_size,
            self.__object_scale,
            loss_fn_kwargs,
        )
        loss_fn.set_blind_guessing_stats(
            float(stats["occupancy_blind_guessing_expected_value"]), stats["box_std"]
        )
        self.__prediction_target_stats = {
            name: np.concatenate([stats[f"latent_{name}"], stats[f"box_{name}"]])
            for name in ("mean", "std", "min", "max")
        }

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

        # The prediction bounds are the empirical target extremes over the dataset
        # and the initial pose distribution, widened by a small margin (the extremes
        # are estimates from finitely many pose samples, and a small floor keeps
        # dimensions the dataset barely exercises, e.g. KL-collapsed latent entries,
        # from degenerating to a point). The bounding box center xy is additionally
        # widened by the pose perturbation drift bound, and its z component and the
        # box size can never be negative.
        stats_min = self.__prediction_target_stats["min"]
        stats_max = self.__prediction_target_stats["max"]
        margin = np.maximum(0.1 * (stats_max - stats_min), 1e-3)
        low = stats_min - margin
        high = stats_max + margin
        low[dims : dims + 2] -= max_expected_translation_perturbation_norm
        high[dims : dims + 2] += max_expected_translation_perturbation_norm
        low[dims + 2 :] = np.maximum(low[dims + 2 :], 0.0)
        pred_space = gym.spaces.Box(
            low.astype(np.float32), high.astype(np.float32), dtype=np.float32
        )
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
            loss_fn=loss_fn.normalized,
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
            # plain logits decode; the encoder is only used during __init__ (by the target statistics computation,
            # which also compiles its own decode batch size when the statistics are not cached yet).
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
                reconstructions: list[Any] = [MESH_INVISIBLE] * self.num_envs
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

    @property
    def prediction_target_stats(self) -> dict[str, np.ndarray]:
        """
        Per-dimension statistics ("mean", "std", "min", "max"; each of shape
        (full_latent_size,)) of the prediction targets over the dataset and the
        initial object pose distribution (see _compute_prediction_target_stats).
        Agents may use them e.g. to standardize their prediction head's outputs.
        """
        return {
            name: value.copy()
            for name, value in self.__prediction_target_stats.items()
        }


def TactileShapeReconstructionEnv(
    config: TactilePerceptionConfig,
    render_mode: Literal["rgb_array", "human"] = "rgb_array",
    model: str = "TimSchneider42/cod-vae-16x8",
    backend: Literal["auto", "torch", "jax"] = "auto",
    device: str | None = None,
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
            renderer_show_shadow_objects=renderer_show_shadow_objects,
            shadow_object_resolution=shadow_object_resolution,
            half_precision=half_precision,
            loss_fn_kwargs=loss_fn_kwargs,
        )
    )

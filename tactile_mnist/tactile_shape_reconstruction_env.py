from __future__ import annotations

import logging
from collections import deque, defaultdict
from typing import (
    Literal,
    TYPE_CHECKING,
    Any,
)

import gymnasium as gym
import numpy as np
import trimesh
from transformation import Transformation

from ap_gym import (
    ActivePerceptionVectorToSingleWrapper,
    MSELossFn,
)
from ap_gym.util import update_info_metrics_vec
from cod_vae import CODVAE, CubeTransform
from .tactile_perception_renderer import MESH_INVISIBLE
from .tactile_perception_vector_env import (
    TactilePerceptionVectorEnv,
    TactilePerceptionConfig,
    ActType,
)

if TYPE_CHECKING:
    from .tactile_perception_vector_env import ObsType

logger = logging.getLogger(__name__)


class TactileShapeReconstructionVectorEnv(
    TactilePerceptionVectorEnv[np.ndarray, np.ndarray],
):
    """
    Active tactile shape reconstruction environment.

    The agent has to reconstruct the geometry of the touched object by regressing its flattened COD-VAE latent
    (https://github.com/TimSchneider42/cod-vae): the object's mesh, posed in the platform frame, is normalized into
    the model's [-1, 1] cube and encoded into num_latents x latent_dim latent vectors. Since the cube normalization
    removes the object's position and scale, the target encodes its shape and orientation. Predictions can be decoded
    back into a mesh with the same COD-VAE model (see cod_vae.CODVAEBase.decode_mesh).

    The latent is a deterministic function of the posed mesh: encoding uses the deterministic posterior mean, the
    surface sampling is seeded, and COD-VAE puts the latent tokens into a canonical order.

    If renderer_show_shadow_objects is set, the current prediction is decoded back into a mesh and rendered as a
    translucent shadow object. This is disabled by default, as it requires a COD-VAE decoder pass per step.
    """

    def __init__(
        self,
        config: TactilePerceptionConfig,
        num_envs: int,
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        model: str = "TimSchneider42/cod-vae-4x32",
        backend: Literal["auto", "torch", "jax"] = "auto",
        device: str | None = None,
        latent_bound: float = 8.0,
        renderer_show_shadow_objects: bool = True,
        shadow_object_resolution: int = 64,
    ):
        self.__vae = CODVAE.from_pretrained(model, backend=backend, device=device)
        dims = self.__vae.config.num_latents * self.__vae.config.latent_dim
        self.__latent_cache: dict[
            tuple[Any, bytes], tuple[np.ndarray, CubeTransform]
        ] = {}

        # COD-VAE latents are KL-regularized towards a standard normal, so they are approximately unit-scale by
        # construction and are regressed without further normalization.
        pred_space = gym.spaces.Box(-latent_bound, latent_bound, shape=(dims,))

        super().__init__(
            config,
            num_envs,
            single_prediction_space=pred_space,
            single_prediction_target_space=pred_space,
            loss_fn=MSELossFn(target_std=1.0).normalized,
            render_mode=render_mode,
        )

        self.__renderer_show_shadow_objects = renderer_show_shadow_objects
        self.__shadow_object_resolution = shadow_object_resolution
        self.__metrics: dict[str, tuple[deque[float], ...]] | None = None

        if self.__vae.backend == "jax":
            # As with Taxim (see TactilePerceptionVectorEnv.__init__), JITing COD-VAE inside a host callback
            # deadlocks, so we trigger the compilation here. Since JAX recompiles for every new batch size, encoding
            # and decoding always use batches of exactly num_envs elements (see __get_targets and _step), so a single
            # warmup call per direction covers all shapes the model will ever see.
            dummy_latents = self.__vae.encode_mesh(
                [trimesh.creation.icosphere(subdivisions=1)] * num_envs, seed=0
            )
            if self.__renderer_show_shadow_objects:
                self.__vae.decode_mesh(
                    dummy_latents, resolution=self.__shadow_object_resolution
                )

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

        target_latents, target_transforms = self.__get_targets()
        error = prediction - target_latents
        latent_rmse = np.sqrt(np.mean(error**2, axis=-1))
        rel_error = np.linalg.norm(error, axis=-1) / np.maximum(
            np.linalg.norm(target_latents, axis=-1), 1e-10
        )

        for i in range(self.num_envs):
            if prev_done[i]:
                self.__metrics["latent_rmse"][i].clear()
                self.__metrics["rel_error"][i].clear()
            else:
                self.__metrics["latent_rmse"][i].append(latent_rmse[i])
                self.__metrics["rel_error"][i].append(rel_error[i])

        obs, action_reward, terminated, truncated, info, targets = super()._step(
            action, prediction
        )

        if self.__renderer_show_shadow_objects:
            # Do that after the step as new objects might be loaded. The decoded meshes are mapped into the platform
            # frame via the cube transforms of the target meshes, hence the shadow objects are placed at the platform
            # frame's origin.
            prediction_clipped = np.clip(
                prediction,
                self.single_prediction_space.low,
                self.single_prediction_space.high,
            ).reshape(
                self.num_envs,
                self.__vae.config.num_latents,
                self.__vae.config.latent_dim,
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
                    decoded = self.__vae.decode_mesh(
                        prediction_clipped[decode_idx],
                        resolution=self.__shadow_object_resolution,
                        transform=[target_transforms[i] for i in decode_idx],
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

    def _get_prediction_targets(self) -> np.ndarray:
        return self.__get_targets()[0]

    def __get_targets(self) -> tuple[np.ndarray, list[CubeTransform]]:
        """
        Compute the flattened COD-VAE latents of the current objects posed in the platform frame, along with the cube
        transforms mapping decoded geometry back into the platform frame.

        Latents are cached per (object, pose), so the two target computations of a step (metrics and prediction
        targets) only encode once, and objects are only re-encoded when their pose changes. Missing latents are
        encoded in a single batch across environments.
        """
        keys = [
            (dp.id, np.asarray(pose.matrix).tobytes())
            for dp, pose in zip(
                self.current_data_points, self.current_object_poses_platform_frame
            )
        ]
        missing = {
            key: (dp, pose)
            for key, dp, pose in zip(
                keys, self.current_data_points, self.current_object_poses_platform_frame
            )
            if key not in self.__latent_cache
        }
        if missing:
            meshes = []
            for dp, pose in missing.values():
                mesh = dp.mesh.copy()
                mesh.apply_transform(pose.matrix)
                meshes.append(mesh)
            if self.__vae.backend == "jax":
                # Pad the batch to num_envs so the jitted encoder only ever sees a single batch size and is never
                # recompiled inside a host callback (see __init__). zip below drops the padding entries.
                meshes += [meshes[0]] * (self.num_envs - len(meshes))
            latents, transforms = self.__vae.encode_mesh(
                meshes, seed=0, return_transform=True
            )
            self.__latent_cache.update(
                {
                    key: (latent.reshape(-1).astype(np.float32), transform)
                    for key, latent, transform in zip(missing, latents, transforms)
                }
            )
        # Keep only the latents of the current objects/poses
        self.__latent_cache = {key: self.__latent_cache[key] for key in keys}
        return (
            np.stack([self.__latent_cache[key][0] for key in keys]),
            [self.__latent_cache[key][1] for key in keys],
        )

    @property
    def vae(self):
        """The COD-VAE model used to compute the latent targets."""
        return self.__vae


def TactileShapeReconstructionEnv(
    config: TactilePerceptionConfig,
    render_mode: Literal["rgb_array", "human"] = "rgb_array",
    model: str = "TimSchneider42/cod-vae-4x32",
    backend: Literal["auto", "torch", "jax"] = "auto",
    device: str | None = None,
    latent_bound: float = 8.0,
    renderer_show_shadow_objects: bool = False,
    shadow_object_resolution: int = 64,
) -> ActivePerceptionVectorToSingleWrapper["ObsType", ActType, np.ndarray, np.ndarray]:
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
        )
    )

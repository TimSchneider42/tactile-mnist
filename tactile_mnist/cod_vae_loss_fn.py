"""
COD-VAE reconstruction loss for the tactile shape reconstruction environments.

:class:`CODVAEReconstructionLossFn` scores a prediction that is a COD-VAE *full
latent* (flattened latent plus normalized bounding box center and size, see
:meth:`cod_vae.CODVAEBase.pack_full_latent`) in the platform frame normalized to
[-1, 1] by ``frame_half_size`` (half the platform cell size), using the loss COD-VAE
is trained with: binary cross entropy of the decoded occupancy against ground-truth
mesh occupancy at query points sampled uniformly in the volume (weight ``vol_coeff``)
and near the object surface (weight ``near_coeff``), plus the mean squared error
between the predicted and the ground-truth bounding box parameters (weight
``box_coeff``, see below).

The target identifies the ground-truth geometry instead of prescribing a particular
latent: it is a dict with the index of the mesh in the dataset ("mesh_index"), the
pose mapping the *raw* dataset mesh into the platform frame ("position" and
scalar-last "quaternion"), and the posed object's ground-truth bounding box ("box":
axis-aligned bounding box center and maximum half-extent in the platform frame,
normalized by ``frame_half_size`` exactly like the prediction's bounding box
entries); any pre-processing the environment applies to the mesh
(e.g. its smallest-dimension-up re-orientation) must be composed into that pose by
the target provider (see TactileShapeReconstructionVectorEnv._get_prediction_targets).
Query points and occupancy labels follow the sdf_gen recipe COD-VAE is trained on,
but in a compact per-mesh form: all meshes share one global database of uniform
volume points in the [-1, 1] cube, of which each mesh labels a contiguous slice (in
the raw dataset mesh's sdf_gen cube normalization; the dataset meshes are assumed to
be watertight), and each mesh additionally stores a small fixed set of near-surface
points (surface samples perturbed with Gaussian noise once per standard deviation,
stored in float16 with labels computed at the quantized positions). The pools of the
entire dataset are computed once upon construction
(deterministically, in parallel worker processes) and cached as an npz file keyed by
the dataset fingerprint and all content-determining parameters.

If the pools fit within ``max_pool_vram_fraction`` of the VAE device's memory budget
(its total memory scaled by the backend's configured per-process cap, see the
parameter's documentation),
they are loaded onto the device in their entirety upon construction and the
torch/jax loss variants assemble query batches directly on the device. Otherwise
both variants stream per-mesh pools to the device on demand, caching the most
recently used ones (``max_device_cached_pools``) via :class:`TorchDictLRUCache` and
:class:`JaxDictLRUCache` respectively (the latter keeping the jax variant
jit-compatible). In all cases the occupancy labels stay bit-packed on the device and
are unpacked on the fly per evaluation to save device memory.

Every loss evaluation draws fresh volume queries from the mesh's database slice
using the ``rng`` passed through the ap_gym loss interface and, if
``num_near_queries`` is set, an equally fresh subset of the mesh's stored
near-surface points (all of them otherwise). The query points are mapped through the object's pose into the
normalized platform frame and evaluated with COD-VAE's full-latent decoding
(:meth:`cod_vae.CODVAEBase.occupancy_loss_full` and its differentiable backend-native
counterparts ``cod_vae.torch.CODVAEModule.decode_full`` and
``cod_vae.jax.decode_full``), which maps them into the model's cube via the
*predicted* bounding box center and size. Errors in the predicted bounding box center
and size thus directly manifest as occupancy errors.

In addition to the occupancy terms, the loss penalizes the mean squared error
between the predicted bounding box parameters (the prediction's last four entries)
and the target's ground-truth "box", averaged over the four components (weight
``box_coeff``). This term is the bounding box's *only* gradient source: the loss
always evaluates the decoder with COD-VAE's ``stop_transform_gradient`` (the
default), as the decoder's own bounding box gradient — flowing through the triplane
interpolation — is noisy under query subsampling, hostile between the texel-sharp
optimum and distant starts, and identically zero once the predicted box does not
overlap the object (the decoder clamps queries to its cube). The MSE term instead
provides a smooth, exact localization gradient at any distance and pulls the box to
the ground truth rather than to a decoder-compensating offset. Target providers
should compute "box" with the exact functions COD-VAE's encoding uses
(``cod_vae.points_to_cube_transform`` and ``cod_vae.pack_cube_transform``), so that
it agrees bit-for-bit with the box entries
:meth:`cod_vae.CODVAEBase.encode_mesh_full` would produce for the posed mesh.

The decoder is evaluated in the VAE's own compute dtype (``vae.dtype``, selected via
the ``dtype`` argument of :meth:`cod_vae.CODVAE.from_pretrained` and exposed by the
environments as ``half_precision``), so the loss and the environment's shadow-object
reconstruction always agree on precision. The query/pose arithmetic, COD-VAE's mapping
and interpolation of the query points, and the BCE are always float32, preserving the
fidelity of the decoded occupancy and of the latent gradients for a half-precision
model.

**Reconstruction metrics.** The loss says nothing about the quality of the decoded
shape: only ~2% of the volume queries of a typical object are occupied, so a run can
sit at a respectable loss while the reconstruction is visibly wrong.
:meth:`numpy_loss_and_metrics` and its torch and jax counterparts therefore also
return, per sample, ``occ_loss``/``box_loss``/``box_share``, the unweighted
``bce_vol``/``bce_near`` halves of the occupancy term, ``occupied_frac``, and
``iou``/``precision``/``recall`` of the decoded occupancy over the volume queries only
(uniform in the object's cube, hence comparable to the IoU COD-VAE is evaluated with).
The last three are undefined for some samples; the ``_iou``/``_precision``/``_recall``
masks beside them mark those so the aggregation drops them. All come from logits the
loss already computed, so they cost no additional decode.

All three loss variants accept ``occupancy_only``, which drops the bounding box MSE
term and returns the pure occupancy loss. It exists so the expected value of the
occupancy terms under blind guessing can be estimated empirically: unlike for the MSE
term, whose blind-guessing expectation follows analytically from the variance of the
box targets, no useful analytic bound exists for the BCE of a decoded occupancy field
(occupancy probability 0.5 everywhere would yield ln(2) per query, but no latent
decodes to that field). :meth:`set_blind_guessing_stats` therefore accepts an
empirical estimate of the occupancy terms' blind-guessing expectation (obtained by
scoring a fixed mean prediction across the dataset with ``occupancy_only``, see
TactileShapeReconstructionVectorEnv) together with the standard deviation of the box
targets, from which :attr:`blind_guessing_expected_value` — and thus the ``normalized``
loss — is assembled. Until these statistics are set, a coarse heuristic
(ln(2) per occupancy unit weight and uniform width-2 box target intervals) is used.
"""

from __future__ import annotations

import ctypes
import dataclasses
import logging
import math
import multiprocessing
import os
from dataclasses import dataclass
from functools import lru_cache, partial
from typing import Any

import filelock
import numpy as np
import threadpoolctl
import tqdm
from ap_gym import LossFn
from scipy.spatial.transform import Rotation

from cod_vae import CODVAEBase
from cod_vae.training.preprocess import sample_occupancy_pools
from .constants import CACHE_BASE_DIR
from .mesh_dataset import MeshDataset
from .simple_mesh_dataset import SimpleMeshDataset
from .util import get_cache_hash

try:
    import torch
    import cod_vae.torch as cod_vae_torch
    from .torch_dict_lru_cache import TorchDictLRUCache
except ImportError:
    torch = None
    cod_vae_torch = None

try:
    import jax
    import jax.numpy as jnp
    import cod_vae.jax as cod_vae_jax
    from .jax_dict_lru_cache import JaxDictLRUCache
except ImportError:
    jax = None
    jnp = None
    cod_vae_jax = None
    JaxDictLRUCache = None

logger = logging.getLogger(__name__)


@lru_cache(maxsize=4)
def _volume_database(seed: int, size: int) -> np.ndarray:
    """The global database of uniform volume points in the [-1, 1] cube."""
    return np.random.default_rng([seed, 0]).random((size, 3), dtype=np.float32) * 2 - 1


def _compute_mesh_pool(
    mesh_index: int,
    ds,
    object_scale: float,
    vol_database_size: int,
    vol_pool_size: int,
    num_near_points: int,
    near_stddevs: tuple[float, ...],
    seed: int,
) -> tuple[int, dict[str, np.ndarray]]:
    """
    Compute the occupancy pool of one mesh (module-level so worker processes can
    pickle it): the bit-packed occupancy labels of the mesh's contiguous slice of the
    global volume point database plus its labeled near-surface points, all in the
    *raw* dataset mesh's sdf_gen cube normalization ("shifts"/"scale",
    cube = (original - shifts) * scale). Any environment-side mesh pre-processing
    must be composed into the loss targets' pose instead.
    """
    dp = SimpleMeshDataset(ds)[mesh_index]
    rng = np.random.default_rng([seed, 1, mesh_index])
    database = _volume_database(seed, vol_database_size)
    vol_start = int(rng.integers(0, vol_database_size - vol_pool_size + 1))
    pools = sample_occupancy_pools(
        dp.mesh.vertices,
        dp.mesh.faces,
        num_surface=num_near_points // len(near_stddevs),
        near_stddevs=near_stddevs,
        object_scale=object_scale,
        rng=rng,
        vol_points=database[vol_start : vol_start + vol_pool_size],
        # The near-surface points are quantized before labeling so the stored points
        # and labels are exactly consistent even within the tightest band.
        near_dtype=np.float16,
    )

    return mesh_index, {
        "vol_start": np.int64(vol_start),
        "vol_label_bits": np.packbits(pools["vol_label"]),
        "vol_pool_num_occupied": np.int32(pools["vol_label"].sum()),
        "near_points": pools["near_points"],
        "near_label_bits": np.packbits(pools["near_label"]),
        "shifts": pools["shifts"].astype(np.float32),
        "scale": np.float32(pools["scale"]),
    }


@dataclass(frozen=True)
class _OccupancyPools:
    """Precomputed occupancy pools of an entire dataset (host-side)."""

    database: np.ndarray  # (vol_database_size, 3) float32
    vol_start: np.ndarray  # (num_meshes,) int64
    vol_label_bits: np.ndarray  # (num_meshes, ceil(vol_pool_size / 8)) uint8
    vol_pool_num_occupied: np.ndarray  # (num_meshes,) int32
    near_points: np.ndarray  # (num_meshes, num_near_points, 3) float16
    near_label_bits: np.ndarray  # (num_meshes, ceil(num_near_points / 8)) uint8
    shifts: np.ndarray  # (num_meshes, 3) float32
    scale: np.ndarray  # (num_meshes,) float32


def _load_or_compute_pools(
    dataset: MeshDataset,
    object_scale: float,
    vol_database_size: int,
    vol_pool_size: int,
    num_near_points: int,
    near_stddevs: tuple[float, ...],
    seed: int,
) -> _OccupancyPools:
    """
    Deterministically compute the occupancy pools of every mesh in the dataset in
    parallel worker processes, or load them from the npz cache keyed by the dataset
    fingerprint and all content-determining parameters.
    """
    if len(dataset) == 0:
        raise ValueError("Cannot compute occupancy pools of an empty dataset.")
    kwargs = dict(
        object_scale=object_scale,
        vol_database_size=vol_database_size,
        vol_pool_size=vol_pool_size,
        num_near_points=num_near_points,
        near_stddevs=near_stddevs,
        seed=seed,
    )
    cache_dir = CACHE_BASE_DIR / "cod_vae_occupancy_pools"
    cache_dir.mkdir(parents=True, exist_ok=True)
    ds_fingerprint = dataset.huggingface_dataset._fingerprint
    cache_file = cache_dir / f"{get_cache_hash(ds_fingerprint, kwargs)}.npz"
    with filelock.FileLock(cache_dir / f"{ds_fingerprint}.lock"):
        if cache_file.exists():
            try:
                with np.load(cache_file) as data:
                    return _OccupancyPools(**data)
            except Exception as ex:
                logger.warning(
                    f"Loading the occupancy pools from cache failed with the "
                    f"following exception: {ex}"
                )
        print("Computing COD-VAE occupancy pools (the results will be cached)...")
        mesh_pools: list[dict[str, np.ndarray] | None] = [None] * len(dataset)
        # Spawned (not forked) workers: the parent typically has CUDA and JAX's
        # thread pools initialized at this point, which forking can deadlock.
        # Limit each worker to a single BLAS thread, as the workers would otherwise
        # spawn one BLAS thread per CPU core each, oversubscribing the CPU massively.
        # The initializer must be picklable for the spawn context, hence no lambda.
        with multiprocessing.get_context("spawn").Pool(
            processes=min(multiprocessing.cpu_count(), 8),
            initializer=partial(
                threadpoolctl.threadpool_limits, limits=1, user_api="blas"
            ),
        ) as pool:
            for mesh_index, mesh_pool in tqdm.tqdm(
                pool.imap_unordered(
                    partial(
                        _compute_mesh_pool,
                        ds=dataset.huggingface_dataset,
                        **kwargs,
                    ),
                    range(len(dataset)),
                ),
                total=len(dataset),
            ):
                mesh_pools[mesh_index] = mesh_pool
        pools = _OccupancyPools(
            database=_volume_database(seed, vol_database_size),
            **{
                name: np.stack([mesh_pool[name] for mesh_pool in mesh_pools])
                for name in (f.name for f in dataclasses.fields(_OccupancyPools))
                if name != "database"
            },
        )
        np.savez(cache_file, **dataclasses.asdict(pools))
        return pools


def _cuda_driver_total_memory(cuda_ordinal: int) -> int | None:
    """
    Total memory of a CUDA device in bytes via the driver API, or None if
    unavailable. The driver honors CUDA_VISIBLE_DEVICES, so CUDA runtime ordinals
    (jax's ``local_hardware_id``) address the same devices.
    """
    try:
        lib = ctypes.CDLL("libcuda.so.1")
    except OSError:
        return None
    device = ctypes.c_int()
    total = ctypes.c_size_t()
    if (
        lib.cuInit(0) != 0
        or lib.cuDeviceGet(ctypes.byref(device), cuda_ordinal) != 0
        or lib.cuDeviceTotalMem_v2(ctypes.byref(total), device) != 0
    ):
        return None
    return total.value


def _torch_unpackbits(bits: "torch.Tensor", count: int) -> "torch.Tensor":
    """Torch counterpart of ``np.unpackbits(bits, axis=-1, count=count)``."""
    shifts = torch.arange(7, -1, -1, device=bits.device, dtype=torch.uint8)
    unpacked = (bits.unsqueeze(-1) >> shifts) & 1
    return unpacked.reshape(bits.shape[:-1] + (-1,))[..., :count]


def _quaternion_to_matrix_np(quaternion: np.ndarray) -> np.ndarray:
    return Rotation.from_quat(quaternion).as_matrix().astype(np.float32)


def _quaternion_to_matrix(quaternion, xp):
    """Rotation matrices (..., 3, 3) from scalar-last quaternions (..., 4)."""
    quaternion = quaternion / ((quaternion**2).sum(-1) ** 0.5)[..., None]
    x, y, z, w = (quaternion[..., i] for i in range(4))
    return xp.stack(
        [
            xp.stack(
                [
                    1 - 2 * (y**2 + z**2),
                    2 * (x * y - z * w),
                    2 * (x * z + y * w),
                ],
                -1,
            ),
            xp.stack(
                [
                    2 * (x * y + z * w),
                    1 - 2 * (x**2 + z**2),
                    2 * (y * z - x * w),
                ],
                -1,
            ),
            xp.stack(
                [
                    2 * (x * z - y * w),
                    2 * (y * z + x * w),
                    1 - 2 * (x**2 + y**2),
                ],
                -1,
            ),
        ],
        -2,
    )


class CODVAEReconstructionLossFn(LossFn[np.ndarray, dict[str, np.ndarray]]):
    def __init__(
        self,
        vae: CODVAEBase,
        dataset: MeshDataset,
        frame_half_size: float,
        object_scale: float = 0.9,
        num_vol_queries: int = 1024,
        vol_pool_size: int = 10_000,
        vol_database_size: int = 1_000_000,
        num_near_points: int = 1024,
        num_near_queries: int | None = None,
        near_stddevs: tuple[float, ...] = (0.005, 0.05),
        vol_coeff: float = 1.0,
        near_coeff: float = 0.1,
        box_coeff: float = 1.0,
        vol_class_balance: float = 0.0,
        max_pool_vram_fraction: float = 0.25,
        max_device_cached_pools: int = 1024,
        preprocessing_seed: int = 0,
    ):
        """
        :param vae: COD-VAE model used to decode predictions. The numpy variant of this
            loss works with any backend; the torch/jax variants require the
            corresponding backend.
        :param dataset: mesh dataset the targets' "mesh_index" entries refer to. The
            meshes are assumed to be watertight. The occupancy pools of the entire
            dataset are computed (or loaded from the disk cache) upon construction.
            The targets' pose refers to the raw dataset mesh; any environment-side
            mesh pre-processing must be composed into the target pose.
        :param frame_half_size: half-extent (in meters) of the world frame the
            predictions' full latents are normalized by, i.e. platform-frame positions
            divided by this value lie in [-1, 1] (see
            TactileShapeReconstructionVectorEnv).
        :param object_scale: cube fill factor of the predictions' cube normalization;
            must match the environment's. Also used for the pools' cube normalization.
        :param num_vol_queries: number of volume query points drawn freshly from the
            mesh's database slice per evaluation.
        :param vol_pool_size: length of the contiguous slice of the global volume
            point database each mesh labels.
        :param vol_database_size: number of uniform volume points in the global
            database shared by all meshes (about 12 MB per million points).
        :param num_near_points: number of near-surface points stored per mesh (see
            ``num_near_queries``). Must be divisible by the number of
            ``near_stddevs``.
        :param num_near_queries: number of near-surface query points drawn freshly
            (without replacement) from the mesh's stored near-surface points per
            evaluation; None (the default) uses all of them.
        :param near_stddevs: standard deviations of the Gaussian surface perturbation
            generating the near-surface points (sdf_gen default (0.005, 0.05)).
        :param vol_coeff: weight of the volume occupancy BCE term (COD-VAE default 1.0).
        :param near_coeff: weight of the near-surface occupancy BCE term (COD-VAE
            default 0.1).
        :param vol_class_balance: strength, in [0, 1], with which a mesh's occupied
            volume queries are upweighted relative to its empty ones: each
            gets weight ``(n_empty / n_occupied) ** strength``, so 0.0 is the plain
            uniform average and 1.0 gives the two classes equal total weight.
        :param box_coeff: weight of the mean squared error between the predicted and
            the ground-truth bounding box parameters (the target's "box" entry),
            averaged over the four normalized components. This term is the bounding
            box's only gradient source (the decoder's own box gradient is always
            stopped, see the class documentation) and provides a smooth localization
            gradient at any distance. If 0.0, the term is dropped, the "box" target
            entry is not required, and the bounding box entries receive no gradient
            at all.
        :param max_pool_vram_fraction: maximum fraction of the VAE device's memory
            budget the pools may occupy for them to be loaded onto the device in
            their entirety upon construction (with labels stored bit-packed, about
            ``vol_pool_size / 8 + 6 * num_near_points`` bytes per mesh plus 12 bytes
            per database point). The budget is the device's total memory scaled by
            the backend's configured per-process cap: torch's per-process memory
            fraction (``torch.cuda.set_per_process_memory_fraction`` or
            ``PYTORCH_CUDA_ALLOC_CONF=per_process_memory_fraction:...``) or jax's
            allocator limit (``XLA_PYTHON_CLIENT_MEM_FRACTION``). If the pools would
            exceed it, or the budget cannot be determined, per-mesh pools are instead
            streamed to the device on demand (see ``max_device_cached_pools``).
        :param max_device_cached_pools: number of per-mesh pools cached on the device
            when the pools are not fully device-resident. A single batch must not
            reference more distinct meshes than this.
        :param preprocessing_seed: base seed of the pool preprocessing.
        """
        super().__init__()
        if num_near_queries is None:
            num_near_queries = num_near_points
        if num_vol_queries <= 0 or num_near_points <= 0 or num_near_queries <= 0:
            raise ValueError("Query counts must be positive.")
        if num_near_queries > num_near_points:
            raise ValueError(
                f"num_near_queries ({num_near_queries}) must not exceed "
                f"num_near_points ({num_near_points}), as near-surface queries are "
                f"drawn without replacement."
            )
        if num_vol_queries > vol_pool_size:
            raise ValueError(
                f"num_vol_queries ({num_vol_queries}) must not exceed vol_pool_size "
                f"({vol_pool_size}), as volume queries are drawn without replacement."
            )
        if vol_pool_size > vol_database_size:
            raise ValueError(
                f"vol_pool_size ({vol_pool_size}) must not exceed vol_database_size "
                f"({vol_database_size})."
            )
        if num_near_points % len(near_stddevs) != 0:
            raise ValueError(
                f"num_near_points ({num_near_points}) must be divisible by the number "
                f"of near_stddevs ({len(near_stddevs)})."
            )
        if not 0 <= max_pool_vram_fraction <= 1:
            raise ValueError(
                f"max_pool_vram_fraction must be in [0, 1], got "
                f"{max_pool_vram_fraction}."
            )
        if max_device_cached_pools <= 0:
            raise ValueError("max_device_cached_pools must be positive.")
        self.__vae = vae
        self.__latent_dims = vae.config.num_latents * vae.config.latent_dim
        self.__occupancy_blind_guessing_expected_value: float | None = None
        self.__box_target_std: np.ndarray | None = None
        self.__num_vol_queries = num_vol_queries
        self.__vol_pool_size = vol_pool_size
        self.__num_near_points = num_near_points
        self.__num_near_queries = num_near_queries
        self.__vol_coeff = vol_coeff
        self.__near_coeff = near_coeff
        self.__box_coeff = box_coeff
        self.__vol_class_balance = vol_class_balance
        self.__object_scale = object_scale
        self.__frame_half_size = float(frame_half_size)
        self.__pools = _load_or_compute_pools(
            dataset,
            object_scale,
            vol_database_size,
            vol_pool_size,
            num_near_points,
            tuple(near_stddevs),
            preprocessing_seed,
        )
        self.__device_pools: dict[str, Any] | None = None
        self.__torch_database = None
        self.__torch_pool_cache: TorchDictLRUCache | None = None
        self.__jax_database = None
        self.__jax_pool_cache: JaxDictLRUCache | None = None
        self.__jax_jitted = None
        self.__jax_jitted_metrics = None
        device_memory, device = self.__device_total_memory()
        pool_bytes = self.__device_pool_bytes()
        if (
            device_memory is not None
            and pool_bytes <= max_pool_vram_fraction * device_memory
        ):
            self.__load_device_pools()
        else:
            if device_memory is not None:
                detail = (
                    f"amounting to {pool_bytes / device_memory * 100: 0.2f}% of this device's memory budget. However, "
                    f"the limit is currently {max_pool_vram_fraction * 100: 0.2f}%"
                )
            else:
                detail = "and this device's memory budget could not be determined"
            logger.warning(
                f"COD-VAE sample pool would take {pool_bytes / 1024**3:0.2f}GB of memory on device {device}, "
                f"{detail}. Hence, the pool will be stored on the host and dynamically loaded, which will be slower. "
                f"Consider increasing max_pool_vram_fraction to avoid this slowdown."
            )
            pools = self.__pools
            pool_arrays = {
                "vol_start": pools.vol_start,
                "vol_label_bits": pools.vol_label_bits,
                "vol_pool_num_occupied": pools.vol_pool_num_occupied,
                "near_points": pools.near_points,
                "near_label_bits": pools.near_label_bits,
                "shifts": pools.shifts,
                "scale": pools.scale,
            }
            if self.__vae.backend == "torch":
                self.__torch_pool_cache = TorchDictLRUCache(
                    pool_arrays,
                    capacity=max_device_cached_pools,
                    device=self.__vae.device,
                )
            else:
                self.__jax_pool_cache = JaxDictLRUCache(
                    pool_arrays,
                    capacity=max_device_cached_pools,
                    device=self.__vae.device,
                )
                self.__jax_database = jax.device_put(
                    pools.database, self.__vae.device
                )

    @property
    def vae(self) -> CODVAEBase:
        return self.__vae

    def set_blind_guessing_stats(
        self,
        occupancy_expected_value: float | None,
        box_target_std: np.ndarray | float | None,
    ) -> None:
        """
        Set empirical blind-guessing statistics used by
        :attr:`blind_guessing_expected_value` (and thus by ``normalized``) in place of
        the coarse built-in heuristic (see the module documentation).

        :param occupancy_expected_value: expected value of the occupancy terms (the
            ``occupancy_only`` loss) under blind guessing, e.g. estimated by scoring a
            fixed mean prediction across the dataset. None restores the heuristic.
        :param box_target_std: per-component (or scalar) standard deviation of the
            targets' normalized bounding box parameters; the MSE term's blind-guessing
            expectation is ``box_coeff * mean(box_target_std**2)``. None restores the
            heuristic.
        """
        self.__occupancy_blind_guessing_expected_value = (
            None if occupancy_expected_value is None else float(occupancy_expected_value)
        )
        self.__box_target_std = (
            None
            if box_target_std is None
            else np.asarray(box_target_std, dtype=np.float64)
        )

    def __device_pool_bytes(self) -> int:
        """Device size of the pools (the labels stay bit-packed on the device)."""
        pools = self.__pools
        num_meshes = pools.scale.shape[0]
        per_mesh = (
            pools.vol_label_bits.shape[-1]  # bit-packed volume labels
            + self.__num_near_points * 3 * 2  # float16 near-surface points
            + pools.near_label_bits.shape[-1]  # bit-packed near-surface labels
            + 3 * 4
            + 4
            + 8
            + 4  # shifts, scale, vol_start, vol_pool_num_occupied
        )
        return pools.database.nbytes + num_meshes * per_mesh

    def __device_total_memory(self) -> tuple[float | None, Any]:
        """
        Effective memory budget of the VAE's device in bytes: its total memory
        scaled by the backend's configured per-process cap; inf for CPU, None if
        unknown.
        """
        device = self.__vae.device
        if self.__vae.backend == "torch":
            if device.type == "cpu":
                return math.inf, device
            if device.type == "cuda":
                index = (
                    device.index
                    if device.index is not None
                    else torch.cuda.current_device()
                )
                # The fraction reflects both set_per_process_memory_fraction and
                # PYTORCH_CUDA_ALLOC_CONF=per_process_memory_fraction:...
                fraction = torch.cuda.get_per_process_memory_fraction(index)
                total = torch.cuda.get_device_properties(index).total_memory
                return total * fraction, device
            return None, device
        if device.platform == "cpu":
            return math.inf, device
        # Under jax's default BFC allocator, bytes_limit is the allocator's hard cap
        # and already accounts for XLA_PYTHON_CLIENT_MEM_FRACTION, preallocated or
        # not. The platform and cuda_async allocators expose no stats; fall back to
        # the driver's total memory, scaled by XLA_PYTHON_CLIENT_MEM_FRACTION if set
        # even though these allocators do not enforce it.
        stats = device.memory_stats() or {}
        limit = stats.get("bytes_limit")
        if limit is not None:
            return limit, device
        ordinal = getattr(device, "local_hardware_id", None)
        total = None if ordinal is None else _cuda_driver_total_memory(ordinal)
        if total is None:
            return None, device
        fraction = float(os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION", 1.0))
        return total * fraction, device

    def __load_device_pools(self) -> None:
        """Load the pools of the entire dataset onto the VAE's device."""
        pools = self.__pools
        arrays = {
            "database": pools.database,
            "vol_start": pools.vol_start,
            "vol_label_bits": pools.vol_label_bits,
            "vol_pool_num_occupied": pools.vol_pool_num_occupied,
            "near_points": pools.near_points,
            "near_label_bits": pools.near_label_bits,
            "shifts": pools.shifts,
            "scale": pools.scale,
        }
        if self.__vae.backend == "torch":
            device = self.__vae.device
            self.__device_pools = {
                name: torch.from_numpy(value).to(device)
                for name, value in arrays.items()
            }
        else:
            self.__device_pools = {
                name: jax.device_put(value, self.__vae.device)
                for name, value in arrays.items()
            }

    def __require_rng(self, rng: np.random.Generator | None):
        if rng is None:
            raise ValueError(
                "CODVAEReconstructionLossFn samples fresh query points on every "
                "evaluation and thus requires an rng."
            )

    @staticmethod
    def __flatten(value, trailing_dims: int):
        shape = value.shape if trailing_dims == 0 else value.shape[:-trailing_dims]
        batch_size = math.prod(shape)
        return value.reshape((batch_size,) + value.shape[len(shape) :])

    def numpy(
        self,
        prediction: np.ndarray,
        target: dict[str, np.ndarray],
        batch_shape: tuple[int, ...] = (),
        rng: np.random.Generator | None = None,
        occupancy_only: bool = False,
    ) -> np.ndarray:
        return self.__numpy_impl(
            prediction, target, batch_shape, rng, occupancy_only, False
        )

    def numpy_loss_and_metrics(
        self,
        prediction: np.ndarray,
        target: dict[str, np.ndarray],
        batch_shape: tuple[int, ...] = (),
        rng: np.random.Generator | None = None,
        occupancy_only: bool = False,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        return self.__numpy_impl(
            prediction, target, batch_shape, rng, occupancy_only, True
        )

    def __numpy_impl(
        self,
        prediction: np.ndarray,
        target: dict[str, np.ndarray],
        batch_shape: tuple[int, ...],
        rng: np.random.Generator | None,
        occupancy_only: bool,
        want_metrics: bool,
    ):
        self.__require_rng(rng)
        rng = rng.spawn(1)[0]  # Spawn a child in order to not advance the RNG
        vae = self.__vae
        if vae.backend == "torch":
            assert isinstance(vae, cod_vae_torch.CODVAETorch)
            dev = vae.device
            with torch.no_grad():
                torch_rng = torch.Generator()
                torch_rng.manual_seed(int(rng.integers(0, 2**31 - 1)))
                result = self.__torch_impl(
                    torch.as_tensor(prediction, device=dev),
                    {k: torch.as_tensor(v, device=dev) for k, v in target.items()},
                    batch_shape,
                    torch_rng,
                    occupancy_only=occupancy_only,
                    return_metrics=want_metrics,
                )
                if want_metrics:
                    loss, metrics = result
                    return loss.cpu().numpy(), {
                        k: v.cpu().numpy() for k, v in metrics.items()
                    }
                return result.cpu().numpy()
        else:
            assert vae.backend == "jax"
            assert isinstance(vae, cod_vae_jax.CODVAEJax)
            key = jax.random.PRNGKey(rng.integers(0, 2**31 - 1))
            if want_metrics:
                if self.__jax_jitted_metrics is None:
                    self.__jax_jitted_metrics = jax.jit(
                        partial(self.__jax_impl, return_metrics=True),
                        static_argnums=(2, 4),
                    )
                loss, metrics = self.__jax_jitted_metrics(
                    prediction, target, batch_shape, key, occupancy_only
                )
                return np.array(loss), {k: np.array(v) for k, v in metrics.items()}
            if self.__jax_jitted is None:
                self.__jax_jitted = jax.jit(
                    partial(self.__jax_impl, return_metrics=False),
                    static_argnums=(2, 4),
                )
            return np.array(
                self.__jax_jitted(
                    prediction,
                    target,
                    batch_shape,
                    key,
                    occupancy_only,
                )
            )

    def torch(
        self,
        prediction: "torch.Tensor",
        target: "dict[str, torch.Tensor]",
        batch_shape: tuple[int, ...] = (),
        rng: "torch.Generator | None" = None,
        occupancy_only: bool = False,
    ) -> "torch.Tensor":
        return self.__torch_impl(
            prediction, target, batch_shape, rng, occupancy_only, False
        )

    def torch_loss_and_metrics(
        self,
        prediction: "torch.Tensor",
        target: "dict[str, torch.Tensor]",
        batch_shape: tuple[int, ...] = (),
        rng: "torch.Generator | None" = None,
        occupancy_only: bool = False,
    ) -> "tuple[torch.Tensor, dict[str, torch.Tensor]]":
        return self.__torch_impl(
            prediction, target, batch_shape, rng, occupancy_only, True
        )

    def __torch_impl(
        self,
        prediction: "torch.Tensor",
        target: "dict[str, torch.Tensor]",
        batch_shape: tuple[int, ...],
        rng: "torch.Generator | None",
        occupancy_only: bool,
        return_metrics: bool,
    ) -> "torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]":
        if self.__vae.backend != "torch":
            raise NotImplementedError(
                f"The torch variant of this loss function requires a COD-VAE model "
                f"with the torch backend, got {self.__vae.backend!r}."
            )

        self.__require_rng(rng)
        module = self.__vae.module
        # Decoding happens on the model's device; autograd moves gradients back to the
        # prediction's device if it differs.
        device = self.__vae.device

        prediction = self.__flatten(prediction.to(device=device), 1)
        position = self.__flatten(target["position"].to(device=device), 1)
        quaternion = self.__flatten(target["quaternion"].to(device=device), 1)
        batch_size = prediction.shape[0]

        # Uniform volume query subsets without replacement, drawn on the generator's
        # device (the first num_vol_queries entries of random permutations).
        vol_idx = torch.argsort(
            torch.rand(
                batch_size,
                self.__vol_pool_size,
                generator=rng,
                device=rng.device,
            ),
            dim=-1,
        )[:, : self.__num_vol_queries].to(device)
        near_idx = None
        if self.__num_near_queries < self.__num_near_points:
            near_idx = torch.argsort(
                torch.rand(
                    batch_size,
                    self.__num_near_points,
                    generator=rng,
                    device=rng.device,
                ),
                dim=-1,
            )[:, : self.__num_near_queries].to(device)

        if self.__device_pools is not None:
            pools = self.__device_pools
            mesh_indices = self.__flatten(target["mesh_index"].to(device=device), 0)
            pool = {
                name: value[mesh_indices]
                for name, value in pools.items()
                if name != "database"
            }
            database = pools["database"]
        else:
            # The mesh indices are only used host-side (device pool cache lookup).
            mesh_indices = self.__flatten(target["mesh_index"].cpu().numpy(), 0)
            if self.__torch_database is None:
                self.__torch_database = torch.from_numpy(self.__pools.database).to(
                    device
                )
            pool = self.__torch_pool_cache.get(mesh_indices)
            database = self.__torch_database
        vol_queries = database[pool["vol_start"][:, None] + vol_idx]
        vol_pool_label = _torch_unpackbits(
            pool["vol_label_bits"], self.__vol_pool_size
        )
        vol_pool_num_occupied = pool["vol_pool_num_occupied"]
        vol_label = vol_pool_label.gather(1, vol_idx)
        near_label = _torch_unpackbits(pool["near_label_bits"], self.__num_near_points)
        near_queries = pool["near_points"].to(torch.float32)
        if near_idx is not None:
            near_label = near_label.gather(1, near_idx)
            near_queries = near_queries.gather(1, near_idx[..., None].expand(-1, -1, 3))
        queries_cube = torch.cat([vol_queries, near_queries], dim=1)
        labels = torch.cat([vol_label, near_label], dim=1).to(torch.float32)
        points = (
            queries_cube / pool["scale"][:, None, None] + pool["shifts"][:, None, :]
        )

        rotations = _quaternion_to_matrix(quaternion, torch)
        queries = (
            torch.einsum("bij,bnj->bni", rotations, points) + position[:, None, :]
        ) / self.__frame_half_size
        # The decoder runs in the VAE's compute dtype; autograd carries gradients back
        # through the cast to the prediction's own dtype. The queries stay float32, as
        # COD-VAE maps and interpolates them in float32 regardless of the model's
        # dtype. The BCE stays float32.
        logits = module.decode_full(
            prediction.to(self.__vae.dtype),
            queries,
            object_scale=self.__object_scale,
            stop_transform_gradient=True,
        )
        if self.__vol_class_balance:
            occupancy = self.__balanced_occupancy_torch(
                logits, labels, vol_pool_num_occupied
            )
        else:
            occupancy = cod_vae_torch.occupancy_loss(
                logits.float(),
                labels,
                self.__num_vol_queries,
                self.__vol_coeff,
                self.__near_coeff,
            )
        loss = occupancy
        box_term = torch.zeros_like(occupancy)
        if self.__box_coeff != 0.0 and not occupancy_only:
            box_target = self.__flatten(target["box"].to(device=device), 1)
            box_error = (
                (prediction[:, -4:].float() - box_target.float()) ** 2
            ).mean(-1)
            box_term = self.__box_coeff * box_error
            loss = loss + box_term
        if not return_metrics:
            return loss.reshape(batch_shape)
        return loss.reshape(batch_shape), self.__torch_metrics(
            logits, labels, occupancy, box_term, batch_shape
        )

    def __balanced_occupancy_torch(self, logits, labels, vol_pool_num_occupied):
        """Torch counterpart of __balanced_occupancy_jax; see it for the rationale."""
        num_vol = self.__num_vol_queries
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits.float(), labels, reduction="none"
        )
        bce_vol, bce_near = bce[:, :num_vol], bce[:, num_vol:]
        pos = labels[:, :num_vol]
        n_occupied = vol_pool_num_occupied.to(torch.float32)
        n_empty = self.__vol_pool_size - n_occupied
        ratio = n_empty.clamp(min=1.0) / n_occupied.clamp(min=1.0)
        w_pos = ratio**self.__vol_class_balance
        n_pos = pos.sum(-1)
        num = (bce_vol * pos).sum(-1) * w_pos + (bce_vol * (1.0 - pos)).sum(-1)
        den = n_pos * w_pos + (num_vol - n_pos)
        vol = num / den.clamp(min=1e-12)
        return self.__vol_coeff * vol + self.__near_coeff * bce_near.mean(-1)

    def __torch_metrics(self, logits, labels, occupancy, box_term, batch_shape):
        """Torch counterpart of __jax_metrics; see the module documentation."""
        num_vol = self.__num_vol_queries
        nan = torch.tensor(float("nan"), device=logits.device)
        with torch.no_grad():
            # The torch backend exports occupancy_loss but not the elementwise BCE it
            # is built from (unlike the jax one), so use the same call it makes.
            bce = torch.nn.functional.binary_cross_entropy_with_logits(
                logits.float(), labels, reduction="none"
            )
            pred = logits[:, :num_vol] > 0
            gt = labels[:, :num_vol] > 0.5
            inter = (pred & gt).sum(-1)
            union = (pred | gt).sum(-1)
            n_pred = pred.sum(-1)
            n_gt = gt.sum(-1)
            out = {
                "occ_loss": occupancy.detach(),
                "box_loss": box_term.detach(),
                "box_share": box_term.detach()
                / torch.clamp(occupancy.detach() + box_term.detach(), min=1e-12),
                "bce_vol": bce[:, :num_vol].mean(-1),
                "bce_near": bce[:, num_vol:].mean(-1),
                "iou": torch.where(union > 0, inter / union.clamp(min=1), nan),
                "_iou": union > 0,
                "precision": torch.where(
                    n_pred > 0, inter / n_pred.clamp(min=1), nan
                ),
                "_precision": n_pred > 0,
                "recall": torch.where(n_gt > 0, inter / n_gt.clamp(min=1), nan),
                "_recall": n_gt > 0,
                "occupied_frac": n_gt / num_vol,
            }
        return {k: v.reshape(batch_shape) for k, v in out.items()}

    def jax(
        self,
        prediction: "jax.Array | np.ndarray",
        target: "dict[str, jax.Array | np.ndarray]",
        batch_shape: tuple[int, ...] = (),
        rng: "jax.Array | None" = None,
        occupancy_only: bool = False,
    ) -> "jax.Array":
        return self.__jax_impl(
            prediction, target, batch_shape, rng, occupancy_only, False
        )

    def jax_loss_and_metrics(
        self,
        prediction: "jax.Array | np.ndarray",
        target: "dict[str, jax.Array | np.ndarray]",
        batch_shape: tuple[int, ...] = (),
        rng: "jax.Array | None" = None,
        occupancy_only: bool = False,
    ) -> "tuple[jax.Array, dict[str, jax.Array]]":
        return self.__jax_impl(
            prediction, target, batch_shape, rng, occupancy_only, True
        )

    def __jax_impl(
        self,
        prediction: "jax.Array | np.ndarray",
        target: "dict[str, jax.Array | np.ndarray]",
        batch_shape: tuple[int, ...],
        rng: "jax.Array | None",
        occupancy_only: bool,
        return_metrics: bool,
    ) -> "jax.Array | tuple[jax.Array, dict[str, jax.Array]]":
        # Jit-compatible, but deliberately not pre-jitted: the streamed-pool cache
        # holds its state in refs, and jax (as of 0.6) cannot nest a jitted
        # function closing over refs inside an outer jit. Callers should jit this
        # (with batch_shape static) or the function containing it themselves.
        if self.__vae.backend != "jax":
            raise NotImplementedError(
                f"The jax variant of this loss function requires a COD-VAE model "
                f"with the jax backend, got {self.__vae.backend!r}."
            )

        self.__require_rng(rng)
        params = self.__vae.params
        config = self.__vae.config

        prediction = self.__flatten(prediction, 1)
        mesh_indices = self.__flatten(target["mesh_index"], 0)
        position = self.__flatten(target["position"], 1)
        quaternion = self.__flatten(target["quaternion"], 1)
        batch_size = prediction.shape[0]

        near_idx = None
        vol_rng = rng
        if self.__num_near_queries < self.__num_near_points:
            vol_rng, near_rng = jax.random.split(rng)
            near_idx = jax.vmap(
                lambda key: jax.random.choice(
                    key,
                    self.__num_near_points,
                    (self.__num_near_queries,),
                    replace=False,
                )
            )(jax.random.split(near_rng, batch_size))
        vol_idx = jax.vmap(
            lambda key: jax.random.choice(
                key,
                self.__vol_pool_size,
                (self.__num_vol_queries,),
                replace=False,
            )
        )(jax.random.split(vol_rng, batch_size))
        if self.__device_pools is not None:
            pools = self.__device_pools
            pool = {
                name: value[mesh_indices]
                for name, value in pools.items()
                if name != "database"
            }
            database = pools["database"]
        else:
            pool = self.__jax_pool_cache.get(mesh_indices)
            database = self.__jax_database
        vol_queries = database[pool["vol_start"][:, None] + vol_idx]
        vol_pool_label = jnp.unpackbits(
            pool["vol_label_bits"], axis=-1, count=self.__vol_pool_size
        )
        vol_pool_num_occupied = pool["vol_pool_num_occupied"]
        vol_label = jnp.take_along_axis(vol_pool_label, vol_idx, axis=1)
        near_queries = pool["near_points"].astype(jnp.float32)
        near_label = jnp.unpackbits(
            pool["near_label_bits"], axis=-1, count=self.__num_near_points
        )
        if near_idx is not None:
            near_queries = jnp.take_along_axis(near_queries, near_idx[..., None], axis=1)
            near_label = jnp.take_along_axis(near_label, near_idx, axis=1)
        queries_cube = jnp.concatenate([vol_queries, near_queries], axis=1)
        labels = jnp.concatenate([vol_label, near_label], axis=1).astype(jnp.float32)
        points = (
            queries_cube / pool["scale"][:, None, None] + pool["shifts"][:, None, :]
        )
        rotations = _quaternion_to_matrix(quaternion, jnp)
        queries = (
            jnp.einsum("bij,bnj->bni", rotations, points) + position[:, None, :]
        ) / self.__frame_half_size
        # The decoder runs in the VAE's compute dtype; casting the prediction is
        # required, as jnp promotion would otherwise silently compute in float32. The
        # queries stay float32, as COD-VAE maps and interpolates them in float32
        # regardless of the model's dtype. The BCE stays float32.
        logits = cod_vae_jax.decode_full(
            params,
            prediction.astype(self.__vae.dtype),
            queries,
            config=config,
            object_scale=self.__object_scale,
            stop_transform_gradient=True,
        )
        if self.__vol_class_balance:
            occupancy = self.__balanced_occupancy_jax(
                logits, labels, vol_pool_num_occupied
            )
        else:
            occupancy = cod_vae_jax.occupancy_loss(
                logits.astype(jnp.float32),
                labels,
                self.__num_vol_queries,
                self.__vol_coeff,
                self.__near_coeff,
            )
        loss = occupancy
        box_term = jnp.zeros_like(occupancy)
        if self.__box_coeff != 0.0 and not occupancy_only:
            box_target = self.__flatten(jnp.asarray(target["box"]), 1)
            box_error = jnp.mean(
                (
                    prediction[:, -4:].astype(jnp.float32)
                    - box_target.astype(jnp.float32)
                )
                ** 2,
                axis=-1,
            )
            box_term = self.__box_coeff * box_error
            loss = loss + box_term
        if not return_metrics:
            return loss.reshape(batch_shape)
        return loss.reshape(batch_shape), self.__jax_metrics(
            logits, labels, occupancy, box_term, batch_shape
        )

    def __balanced_occupancy_jax(self, logits, labels, vol_pool_num_occupied):
        """
        ``cod_vae.jax.occupancy_loss`` with the volume term class-balanced per mesh.

        Each occupied volume query is weighted ``(n_empty / n_occupied) **
        vol_class_balance`` relative to an empty one, counted over that mesh's whole
        precomputed pool rather than over the queries drawn this step; the near-surface
        half is left alone, being ~43% occupied already.
        """
        num_vol = self.__num_vol_queries
        bce = cod_vae_jax.bce_with_logits(logits.astype(jnp.float32), labels)
        bce_vol, bce_near = bce[:, :num_vol], bce[:, num_vol:]
        pos = labels[:, :num_vol]
        n_occupied = jnp.asarray(vol_pool_num_occupied, dtype=jnp.float32)
        n_empty = self.__vol_pool_size - n_occupied
        ratio = jnp.maximum(n_empty, 1.0) / jnp.maximum(n_occupied, 1.0)
        w_pos = ratio ** self.__vol_class_balance
        n_pos = jnp.sum(pos, axis=-1)
        num = jnp.sum(bce_vol * pos, axis=-1) * w_pos + jnp.sum(
            bce_vol * (1.0 - pos), axis=-1
        )
        den = n_pos * w_pos + (num_vol - n_pos)
        vol = num / jnp.maximum(den, 1e-12)
        return self.__vol_coeff * vol + self.__near_coeff * bce_near.mean(axis=-1)

    def __jax_metrics(self, logits, labels, occupancy, box_term, batch_shape):
        """
        Reconstruction metrics from the logits and labels the loss already produced --
        no second decode. See the module documentation for what each one is and why
        the IoU is taken over the volume queries only.
        """
        num_vol = self.__num_vol_queries
        bce = cod_vae_jax.bce_with_logits(logits.astype(jnp.float32), labels)
        # Volume queries only, so this is comparable to the IoU COD-VAE is evaluated with.
        pred = logits[:, :num_vol] > 0
        gt = labels[:, :num_vol] > 0.5
        inter = jnp.sum(pred & gt, axis=-1)
        union = jnp.sum(pred | gt, axis=-1)
        n_pred = jnp.sum(pred, axis=-1)
        n_gt = jnp.sum(gt, axis=-1)
        out = {
            "occ_loss": occupancy,
            "box_loss": box_term,
            # A share rather than raw numbers: the logged loss is normalized while these
            # are raw, and a ratio is immune to that.
            "box_share": box_term / jnp.maximum(occupancy + box_term, 1e-12),
            "bce_vol": bce[:, :num_vol].mean(axis=-1),
            "bce_near": bce[:, num_vol:].mean(axis=-1),
            # An empty union, no predicted positives or no ground-truth positives leave
            # the respective metric undefined. The "_<name>" masks say so per sample, so
            # the aggregation drops those steps instead of scoring them 0 or 1.
            "iou": jnp.where(union > 0, inter / jnp.maximum(union, 1), jnp.nan),
            "_iou": union > 0,
            "precision": jnp.where(
                n_pred > 0, inter / jnp.maximum(n_pred, 1), jnp.nan
            ),
            "_precision": n_pred > 0,
            "recall": jnp.where(n_gt > 0, inter / jnp.maximum(n_gt, 1), jnp.nan),
            "_recall": n_gt > 0,
            "occupied_frac": n_gt / num_vol,
        }
        return {k: v.reshape(batch_shape) for k, v in out.items()}

    def _lower_bound(self) -> float:
        return 0.0

    def _blind_guessing_expected_value(self) -> float:
        # Empirical statistics (see set_blind_guessing_stats) take precedence; the
        # heuristic fallbacks are: a maximum-entropy blind guess (occupancy
        # probability 0.5 everywhere) incurring ln(2) per query point, and, for the
        # bounding box term, the midpoint of each component's prediction interval
        # against targets assumed uniform over width-2 intervals (normalized center
        # within the cell, z and size in [0, 2]), incurring 2**2 / 12 per component.
        if self.__occupancy_blind_guessing_expected_value is not None:
            occupancy = self.__occupancy_blind_guessing_expected_value
        else:
            occupancy = float((self.__vol_coeff + self.__near_coeff) * np.log(2.0))
        if self.__box_target_std is not None:
            box = float(np.mean(self.__box_target_std**2))
        else:
            box = 2.0**2 / 12.0
        return occupancy + self.__box_coeff * box

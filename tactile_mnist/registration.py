from typing import Any, Callable, Iterable

import datasets
from datasets import load_dataset

import ap_gym
from tactile_mnist.tactile_shape_reconstruction_env import (
    TactileShapeReconstructionVectorEnv,
    TactileShapeReconstructionEnv,
)
from tactile_mnist.tactile_volume_estimation_env import (
    TactileVolumeEstimationVectorEnv,
    TactileVolumeEstimationEnv,
)
from .constants import *
from .minecraft_dataset import load_minecraft_item_mesh_dataset
from .sensor_noise import SensorNoiseConfig
from .simple_mesh_dataset import SimpleMeshDataset
from .tactile_classification_env import (
    TactileClassificationEnv,
    TactileClassificationVectorEnv,
)
from .tactile_perception_vector_env import (
    TactilePerceptionConfig,
)
from .tactile_pose_estimation_env import (
    TactilePoseEstimationEnv,
    TactilePoseEstimationVectorEnv,
)
from .tactile_real_snap_env import (
    TactileRealSnapClassificationEnv,
    TactileRealSnapClassificationVectorEnv,
    TactileRealSnapConfig,
    TactileRealSnapVolumeEstimationEnv,
    TactileRealSnapVolumeEstimationVectorEnv,
)
from .touch_data import TouchSingleDataset
from .util import compute_touch_window_size

# The snap variants mimic the touch selection of the TactileMNISTRealSnap environment, which chooses each touch from a
# window of the prerecorded touches of the current round. All real touch datasets contain 256 touches per round.
SNAP_TOUCH_SEQUENCE_LENGTH = 256

# All real-snap environments read the full-resolution recordings and scale them down to their output size. The
# re-rendering variants estimate a depth map from every image at 256x256, so anything smaller would upscale first.
REAL_TOUCH_DATASET = "touch-real-single-t256-320x240"
REAL_SNAP_OUTPUT_SIZE = (64, 64)

# Base depth dataset of the DR variants of the TactileMNIST environments (see SensorNoiseConfig): recorded touches
# without object contact carry no information about the touched objects, so the real MNIST recordings serve every
# TactileMNIST* environment. The other environment families have no real recordings to draw from.
REAL_BASE_DEPTH_DATASET = f"TimSchneider42/tactile-mnist-{REAL_TOUCH_DATASET}"

# Sensor height randomization of the -DR variants (see TactilePerceptionConfig.penetration_depth_reduction_std),
# calibrated such that touches reaching the platform spread like they do in the real recordings (median 0.46mm and
# 90th percentile 1.0mm less deep than the deepest touch of a round).
DR_PENETRATION_DEPTH_REDUCTION_STD = 0.00065

# Platform penetration limit of the -DR variants (see TactilePerceptionConfig.max_platform_penetration). Only applied
# to environments without sensor rotation, as the limit constrains the sensor position rather than the deepest point
# of the gel.
DR_MAX_PLATFORM_PENETRATION = 0.0

# Re-rendering variants of the real-snap environments: the recorded tactile images are mapped into the domain of the
# corresponding simulated environment (see TactileRealSnapConfig.sensor_type). The suffixes mirror the ones the
# simulated environments use, except that the plain suffix stays reserved for the unmodified recordings.
REAL_SNAP_SENSOR_TYPES = [
    ("-Taxim", "taxim"),
    ("-CycleGAN", "cycle_gan"),
    ("-Depth", "depth"),
]


def mk_snap_touch_positions(step_limit: int) -> int:
    """
    Determine how many touch positions the snap variants sample in every step.

    The number matches the window size the TactileMNISTRealSnap environment ends up with for a touch sequence of
    SNAP_TOUCH_SEQUENCE_LENGTH touches and the given step limit.
    """
    return compute_touch_window_size(SNAP_TOUCH_SEQUENCE_LENGTH, step_limit)


def mk_config(
    dataset_name: str | Callable[[str], datasets.Dataset],
    split: str,
    args: Iterable[Any],
    default_config: dict[str, Any],
    config: dict[str, Any] | None = None,
    mesh_dataset_config: dict[str, Any] | None = None,
):
    if callable(dataset_name):
        dataset = dataset_name(split)
    else:
        dataset = load_dataset(
            f"TimSchneider42/tactile-mnist-{dataset_name}", split=split
        )
    return TactilePerceptionConfig(
        SimpleMeshDataset(
            dataset,
            **({} if mesh_dataset_config is None else mesh_dataset_config),
        ),
        *args,
        # Values given in config override the defaults of the environment
        **{**default_config, **({} if config is None else config)},
    )


def register_with_dr_variant(
    id_prefix: str,
    id_suffix: str = "-v0",
    *,
    kwargs: dict[str, Any],
    real_base_depth_dataset: str | None = None,
    **register_kwargs: Any,
):
    """
    Register a simulated environment twice: once as it is and once with domain randomization.

    The domain randomized variant carries a "-DR" suffix between id_prefix and id_suffix (e.g.
    TactileMNIST-CycleGAN-DR-train-v0) and simulates the per-episode and per-frame variations of a real sensor on top
    of the rendered tactile images (see SensorNoiseConfig). If real_base_depth_dataset is given, it additionally
    emulates the artifacts of the depth estimator used by the re-rendering real-snap environments by overlaying
    estimated depth maps of recorded touches without object contact from that dataset. The depth sensor mode outputs
    depth maps rather than tactile images, which the image-space noise terms do not apply to, so its -DR variants
    only emulate the depth estimator artifacts (and the sensor height randomization). Additionally, the domain
    randomized variants randomize the sensor height the agent observes and, if the sensor cannot rotate, limit how
    far the gel may be pressed into the platform.
    """
    if kwargs["default_config"].get("sensor_type") == "depth":
        dr_sensor_noise = (
            None
            if real_base_depth_dataset is None
            else SensorNoiseConfig(
                frame_noise_std=0.0,
                episode_pattern_std=0.0,
                episode_gain_std=0.0,
                episode_offset_std=0.0,
                real_base_depth_dataset=real_base_depth_dataset,
            )
        )
    else:
        dr_sensor_noise = SensorNoiseConfig(
            real_base_depth_dataset=real_base_depth_dataset
        )
    # The platform penetration limit constrains the sensor position, which coincides with the deepest point of the gel
    # only if the sensor points straight down
    dr_max_platform_penetration = (
        None
        if kwargs["default_config"].get("allow_sensor_rotation", True)
        else DR_MAX_PLATFORM_PENETRATION
    )
    for dr_suffix, dr_config in (
        (
            "",
            dict(
                sensor_noise=None,
                penetration_depth_reduction_std=0.0,
                max_platform_penetration=None,
            ),
        ),
        (
            "-DR",
            dict(
                sensor_noise=dr_sensor_noise,
                penetration_depth_reduction_std=DR_PENETRATION_DEPTH_REDUCTION_STD,
                max_platform_penetration=dr_max_platform_penetration,
            ),
        ),
    ):
        ap_gym.register(
            id=f"{id_prefix}{dr_suffix}{id_suffix}",
            kwargs={
                **kwargs,
                "default_config": {**kwargs["default_config"], **dr_config},
            },
            **register_kwargs,
        )


def register_real_snap_with_rerender_variants(
    id_prefix: str,
    id_suffix: str = "-v0",
    *,
    kwargs: dict[str, Any],
    **register_kwargs: Any,
):
    """
    Register a real-snap environment once per sensor type.

    The plain variant returns the recorded tactile images unchanged. The re-rendering variants (e.g.
    TactileMNISTRealSnap-CycleGAN-v0) estimate a depth map from every recorded image and render it with the same
    tactile renderer the simulated environments use, which maps the real images into the simulated domain (see
    TactileRealSnapConfig.sensor_type).
    """
    for sensor_type_name, sensor_type in [("", "direct")] + REAL_SNAP_SENSOR_TYPES:
        ap_gym.register(
            id=f"{id_prefix}{sensor_type_name}{id_suffix}",
            kwargs={
                **kwargs,
                "default_config": {
                    **kwargs["default_config"],
                    "sensor_type": sensor_type,
                    "sensor_output_size": REAL_SNAP_OUTPUT_SIZE,
                },
            },
            **register_kwargs,
        )


def mk_real_snap_config(
    dataset_name: str,
    mesh_dataset_name: str,
    split: str,
    mesh_split: str,
    args: Iterable[Any],
    default_config: dict[str, Any],
    config: dict[str, Any] | None = None,
):
    dataset = load_dataset(f"TimSchneider42/tactile-mnist-{dataset_name}", split=split)
    merged_config = {**default_config, **({} if config is None else config)}
    if "mesh_dataset" not in merged_config:
        # The mesh dataset is used for visualization purposes only
        merged_config["mesh_dataset"] = SimpleMeshDataset(
            load_dataset(
                f"TimSchneider42/tactile-mnist-{mesh_dataset_name}", split=mesh_split
            )
        )
    return TactileRealSnapConfig(
        TouchSingleDataset(dataset),
        *args,
        **merged_config,
    )


def register_envs():
    for split in ["train", "test"]:
        suffixes = [f"-{split}"]
        if split == "train":
            suffixes.append("")
        for s in suffixes:
            register_real_snap_with_rerender_variants(
                "TactileMNISTRealSnap",
                f"{s}-v0",
                entry_point=lambda *args, default_config, config=None, _split=split, **kwargs: ap_gym.ActiveClassificationLogWrapper(
                    TactileRealSnapClassificationEnv(
                        mk_real_snap_config(
                            REAL_TOUCH_DATASET,
                            "mnist3d",
                            _split,
                            f"printed_{_split}",
                            args,
                            default_config,
                            config,
                        ),
                        **kwargs,
                    )
                ),
                vector_entry_point=lambda *args, default_config, config=None, _split=split, **kwargs: ap_gym.ActiveClassificationVectorLogWrapper(
                    TactileRealSnapClassificationVectorEnv(
                        mk_real_snap_config(
                            REAL_TOUCH_DATASET,
                            "mnist3d",
                            _split,
                            f"printed_{_split}",
                            args,
                            default_config,
                            config,
                        ),
                        **kwargs,
                    ),
                ),
                kwargs=dict(
                    default_config=dict(renderer_show_class_weights=True),
                ),
            )

            register_real_snap_with_rerender_variants(
                "TactileMNISTVolumeRealSnap",
                f"{s}-v0",
                entry_point=lambda *args, default_config, config=None, _split=split, **kwargs: ap_gym.ActiveRegressionLogWrapper(
                    TactileRealSnapVolumeEstimationEnv(
                        mk_real_snap_config(
                            REAL_TOUCH_DATASET,
                            "mnist3d",
                            _split,
                            f"printed_{_split}",
                            args,
                            default_config,
                            config,
                        ),
                        **kwargs,
                    )
                ),
                vector_entry_point=lambda *args, default_config, config=None, _split=split, **kwargs: ap_gym.ActiveRegressionVectorLogWrapper(
                    TactileRealSnapVolumeEstimationVectorEnv(
                        mk_real_snap_config(
                            REAL_TOUCH_DATASET,
                            "mnist3d",
                            _split,
                            f"printed_{_split}",
                            args,
                            default_config,
                            config,
                        ),
                        **kwargs,
                    ),
                ),
                kwargs=dict(
                    default_config=dict(step_limit=16),
                ),
            )

            for sensor_type_name, sensor_type in [
                ("", "taxim"),
                ("-CycleGAN", "cycle_gan"),
                ("-Depth", "depth"),
            ]:
                step_limit = 16
                for snap_suffix, snap_touch_positions in [
                    ("", None),
                    ("Snap", mk_snap_touch_positions(step_limit)),
                ]:
                    register_with_dr_variant(
                        f"TactileMNIST{snap_suffix}{sensor_type_name}",
                        f"{s}-v0",
                        entry_point=lambda *args, default_config, config=None, _split=split, **kwargs: ap_gym.ActiveClassificationLogWrapper(
                            TactileClassificationEnv(
                                mk_config(
                                    "mnist3d", _split, args, default_config, config
                                ),
                                **kwargs,
                            )
                        ),
                        vector_entry_point=lambda *args, default_config, config=None, _split=split, **kwargs: ap_gym.ActiveClassificationVectorLogWrapper(
                            TactileClassificationVectorEnv(
                                mk_config(
                                    "mnist3d", _split, args, default_config, config
                                ),
                                **kwargs,
                            ),
                        ),
                        kwargs=dict(
                            default_config=dict(
                                sensor_output_size=(64, 64),
                                allow_sensor_rotation=False,
                                max_initial_angle_perturbation=np.pi / 8,
                                step_limit=step_limit,
                                renderer_show_class_weights=True,
                                sensor_type=sensor_type,
                                snap_touch_positions=snap_touch_positions,
                                show_sensor_target_pos=snap_touch_positions is not None,
                            )
                        ),
                        real_base_depth_dataset=REAL_BASE_DEPTH_DATASET,
                    )

                for (
                    env_name,
                    ds_name,
                    smallest_dim_up,
                    allow_sensor_rotation,
                    step_limit,
                    real_base_depth_dataset,
                ) in [
                    ("TactileMNIST", "mnist3d", False, False, 16, REAL_BASE_DEPTH_DATASET),
                    ("ABC", "abc-dataset-small", True, True, 32, None),
                ]:
                    # Snap variants are only registered for the TactileMNIST environments
                    snap_variants = [("", None)]
                    if env_name == "TactileMNIST":
                        snap_variants.append(
                            ("Snap", mk_snap_touch_positions(step_limit))
                        )
                    for snap_suffix, snap_touch_positions in snap_variants:
                        register_with_dr_variant(
                            f"{env_name}Volume{snap_suffix}{sensor_type_name}",
                            f"{s}-v0",
                            entry_point=lambda *args, default_config, config=None, _split=split, _ds_name=ds_name, **kwargs: ap_gym.ActiveRegressionLogWrapper(
                                TactileVolumeEstimationEnv(
                                    mk_config(
                                        _ds_name, _split, args, default_config, config
                                    ),
                                    **kwargs,
                                )
                            ),
                            vector_entry_point=lambda *args, default_config, config=None, _split=split, _ds_name=ds_name, **kwargs: ap_gym.ActiveRegressionVectorLogWrapper(
                                TactileVolumeEstimationVectorEnv(
                                    mk_config(
                                        _ds_name, _split, args, default_config, config
                                    ),
                                    **kwargs,
                                ),
                            ),
                            kwargs=dict(
                                default_config=dict(
                                    sensor_output_size=(64, 64),
                                    allow_sensor_rotation=allow_sensor_rotation,
                                    step_limit=step_limit,
                                    sensor_type=sensor_type,
                                    cell_size=CELL_SIZE,
                                    smallest_dimension_up=smallest_dim_up,
                                    snap_touch_positions=snap_touch_positions,
                                    show_sensor_target_pos=snap_touch_positions
                                    is not None,
                                )
                            ),
                            real_base_depth_dataset=real_base_depth_dataset,
                        )

                        # No regression log wrapper here: the shape reconstruction
                        # targets are dicts identifying the ground-truth geometry, not
                        # regression vectors; the environment logs its own metrics.
                        for static_suffix, perturb_object_pose in [
                            ("", True),
                            ("Static", False),
                        ]:
                            register_with_dr_variant(
                                f"{env_name}Shape{static_suffix}{snap_suffix}{sensor_type_name}",
                                f"{s}-v0",
                                entry_point=lambda *args, default_config, config=None, _split=split, _ds_name=ds_name, **kwargs: TactileShapeReconstructionEnv(
                                    mk_config(
                                        _ds_name, _split, args, default_config, config
                                    ),
                                    **kwargs,
                                ),
                                vector_entry_point=lambda *args, default_config, config=None, _split=split, _ds_name=ds_name, **kwargs: TactileShapeReconstructionVectorEnv(
                                    mk_config(
                                        _ds_name, _split, args, default_config, config
                                    ),
                                    **kwargs,
                                ),
                                kwargs=dict(
                                    default_config=dict(
                                        sensor_output_size=(64, 64),
                                        allow_sensor_rotation=allow_sensor_rotation,
                                        step_limit=step_limit,
                                        sensor_type=sensor_type,
                                        cell_size=CELL_SIZE,
                                        smallest_dimension_up=smallest_dim_up,
                                        snap_touch_positions=snap_touch_positions,
                                        show_sensor_target_pos=snap_touch_positions
                                        is not None,
                                        perturb_object_pose=perturb_object_pose,
                                    )
                                ),
                                real_base_depth_dataset=real_base_depth_dataset,
                            )

            for sensor_type_name, sensor_type in [
                ("", "taxim"),
                ("-Depth", "depth"),
            ]:
                register_with_dr_variant(
                    f"Starstruck{sensor_type_name}",
                    f"{s}-v0",
                    entry_point=lambda *args, default_config, config=None, _split=split, **kwargs: ap_gym.ActiveClassificationLogWrapper(
                        TactileClassificationEnv(
                            mk_config(
                                "starstruck", _split, args, default_config, config
                            ),
                            **kwargs,
                        )
                    ),
                    vector_entry_point=lambda *args, default_config, config=None, _split=split, **kwargs: ap_gym.ActiveClassificationVectorLogWrapper(
                        TactileClassificationVectorEnv(
                            mk_config(
                                "starstruck", _split, args, default_config, config
                            ),
                            **kwargs,
                        ),
                    ),
                    kwargs=dict(
                        default_config=dict(
                            sensor_output_size=(64, 64),
                            allow_sensor_rotation=False,
                            randomize_initial_object_pose=False,
                            perturb_object_pose=False,
                            step_limit=32,
                            renderer_show_class_weights=True,
                            sensor_type=sensor_type,
                        ),
                    ),
                )

                for (
                    env_name,
                    ds_name,
                    smallest_dim_up,
                    allow_sensor_rotation,
                    step_limit,
                    real_base_depth_dataset,
                ) in [
                    ("ABC", "abc-dataset-small", True, True, 32, None),
                    ("TactileMNIST", "mnist3d", False, False, 16, REAL_BASE_DEPTH_DATASET),
                ]:
                    # Snap variants are only registered for the TactileMNIST environments
                    snap_variants = [("", None)]
                    if env_name == "TactileMNIST":
                        snap_variants.append(
                            ("Snap", mk_snap_touch_positions(step_limit))
                        )
                    for snap_suffix, snap_touch_positions in snap_variants:
                        register_with_dr_variant(
                            f"{env_name}CenterOfMass{snap_suffix}{sensor_type_name}",
                            f"{s}-v0",
                            entry_point=lambda *args, default_config, config=None, _split=split, _ds_name=ds_name, **kwargs: ap_gym.ActiveRegressionLogWrapper(
                                TactilePoseEstimationEnv(
                                    mk_config(
                                        _ds_name,
                                        _split,
                                        args,
                                        default_config,
                                        config,
                                    ),
                                    **kwargs,
                                )
                            ),
                            vector_entry_point=lambda *args, default_config, config=None, _split=split, _ds_name=ds_name, **kwargs: ap_gym.ActiveRegressionVectorLogWrapper(
                                TactilePoseEstimationVectorEnv(
                                    mk_config(
                                        _ds_name,
                                        _split,
                                        args,
                                        default_config,
                                        config,
                                    ),
                                    **kwargs,
                                ),
                            ),
                            kwargs=dict(
                                default_config=dict(
                                    sensor_output_size=(64, 64),
                                    allow_sensor_rotation=allow_sensor_rotation,
                                    step_limit=step_limit,
                                    cell_size=CELL_SIZE,
                                    sensor_type=sensor_type,
                                    smallest_dimension_up=smallest_dim_up,
                                    snap_touch_positions=snap_touch_positions,
                                    show_sensor_target_pos=snap_touch_positions
                                    is not None,
                                ),
                                frame_position_mode="inertia_frame",
                                frame_rotation_mode=None,
                            ),
                            real_base_depth_dataset=real_base_depth_dataset,
                        )

    # Minecraft item meshes are generated on the fly from Mojang's official
    # assets; the resulting dataset has no train/test split
    minecraft_items = lambda split: load_minecraft_item_mesh_dataset()

    for sensor_type_name, sensor_type in [
        ("", "taxim"),
        ("-Depth", "depth"),
    ]:
        register_with_dr_variant(
            f"Minecraft{sensor_type_name}",
            entry_point=lambda *args, default_config, config=None, **kwargs: ap_gym.ActiveClassificationLogWrapper(
                TactileClassificationEnv(
                    mk_config(
                        minecraft_items,
                        "train",
                        args,
                        default_config,
                        config,
                        dict(cache_size="full"),
                    ),
                    **kwargs,
                )
            ),
            vector_entry_point=lambda *args, default_config, config=None, **kwargs: ap_gym.ActiveClassificationVectorLogWrapper(
                TactileClassificationVectorEnv(
                    mk_config(
                        minecraft_items,
                        "train",
                        args,
                        default_config,
                        config,
                        dict(cache_size="full"),
                    ),
                    **kwargs,
                ),
            ),
            kwargs=dict(
                default_config=dict(
                    sensor_output_size=(64, 64),
                    allow_sensor_rotation=False,
                    step_limit=32,
                    sensor_type=sensor_type,
                    renderer_show_orig_mesh_colors=True,
                )
            ),
        )

        # No regression log wrapper here: the shape reconstruction targets are dicts
        # identifying the ground-truth geometry, not regression vectors; the
        # environment logs its own metrics.
        for static_suffix, perturb_object_pose in [
            ("", True),
            ("Static", False),
        ]:
            register_with_dr_variant(
                f"MinecraftShape{static_suffix}{sensor_type_name}",
                entry_point=lambda *args, default_config, config=None, **kwargs: TactileShapeReconstructionEnv(
                    mk_config(
                        minecraft_items,
                        "train",
                        args,
                        default_config,
                        config,
                        dict(cache_size="full"),
                    ),
                    **kwargs,
                ),
                vector_entry_point=lambda *args, default_config, config=None, **kwargs: TactileShapeReconstructionVectorEnv(
                    mk_config(
                        minecraft_items,
                        "train",
                        args,
                        default_config,
                        config,
                        dict(cache_size="full"),
                    ),
                    **kwargs,
                ),
                kwargs=dict(
                    default_config=dict(
                        sensor_output_size=(64, 64),
                        allow_sensor_rotation=False,
                        step_limit=32,
                        sensor_type=sensor_type,
                        cell_size=CELL_SIZE,
                        smallest_dimension_up=False,
                        renderer_show_orig_mesh_colors=True,
                        perturb_object_pose=perturb_object_pose,
                    )
                ),
            )

        for env_name, ds_name, sizes, step_limit, orig_colors in [
            ("Toolbox", "wrench", (("", 0.3), ("-small", 0.25)), 64, False),
            ("MinecraftPose", minecraft_items, (("", 0.2),), 64, True),
        ]:
            for size_name, size in sizes:
                register_with_dr_variant(
                    f"{env_name}{size_name}{sensor_type_name}",
                    entry_point=lambda *args, default_config, config=None, _ds_name=ds_name, **kwargs: ap_gym.ActiveRegressionLogWrapper(
                        TactilePoseEstimationEnv(
                            mk_config(
                                _ds_name,
                                "train",
                                args,
                                default_config,
                                config,
                                dict(cache_size="full"),
                            ),
                            **kwargs,
                        )
                    ),
                    vector_entry_point=lambda *args, default_config, config=None, _ds_name=ds_name, **kwargs: ap_gym.ActiveRegressionVectorLogWrapper(
                        TactilePoseEstimationVectorEnv(
                            mk_config(
                                _ds_name,
                                "train",
                                args,
                                default_config,
                                config,
                                dict(cache_size="full"),
                            ),
                            **kwargs,
                        ),
                    ),
                    kwargs=dict(
                        default_config=dict(
                            sensor_output_size=(64, 64),
                            allow_sensor_rotation=False,
                            step_limit=step_limit,
                            cell_size=(size, size),
                            cell_padding=tuple(
                                np.array([0.005, 0.005]) + GELSIGHT_MINI_OUTER_SIZE / 2
                            ),
                            sensor_type=sensor_type,
                            renderer_show_orig_mesh_colors=orig_colors,
                        ),
                        frame_position_mode="model",
                        frame_rotation_mode="model",
                    ),
                )

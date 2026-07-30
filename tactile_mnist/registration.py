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
            ap_gym.register(
                id=f"TactileMNISTRealSnap{s}-v0",
                entry_point=lambda *args, default_config, config=None, _split=split, **kwargs: ap_gym.ActiveClassificationLogWrapper(
                    TactileRealSnapClassificationEnv(
                        mk_real_snap_config(
                            "touch-real-single-t256-64x64",
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
                            "touch-real-single-t256-64x64",
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

            ap_gym.register(
                id=f"TactileMNISTVolumeRealSnap{s}-v0",
                entry_point=lambda *args, default_config, config=None, _split=split, **kwargs: ap_gym.ActiveRegressionLogWrapper(
                    TactileRealSnapVolumeEstimationEnv(
                        mk_real_snap_config(
                            "touch-real-single-t256-64x64",
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
                            "touch-real-single-t256-64x64",
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
                    ap_gym.register(
                        id=f"TactileMNIST{snap_suffix}{sensor_type_name}{s}-v0",
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
                    )

                for env_name, ds_name, smallest_dim_up, allow_sensor_rotation, step_limit in [
                    ("TactileMNIST", "mnist3d", False, False, 16),
                    ("ABC", "abc-dataset-small", True, True, 32),
                ]:
                    # Snap variants are only registered for the TactileMNIST environments
                    snap_variants = [("", None)]
                    if env_name == "TactileMNIST":
                        snap_variants.append(
                            ("Snap", mk_snap_touch_positions(step_limit))
                        )
                    for snap_suffix, snap_touch_positions in snap_variants:
                        ap_gym.register(
                            id=f"{env_name}Volume{snap_suffix}{sensor_type_name}{s}-v0",
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
                        )

                        ap_gym.register(
                            id=f"{env_name}Shape{snap_suffix}{sensor_type_name}{s}-v0",
                            entry_point=lambda *args, default_config, config=None, _split=split, _ds_name=ds_name, **kwargs: ap_gym.ActiveRegressionLogWrapper(
                                TactileShapeReconstructionEnv(
                                    mk_config(
                                        _ds_name, _split, args, default_config, config
                                    ),
                                    **kwargs,
                                )
                            ),
                            vector_entry_point=lambda *args, default_config, config=None, _split=split, _ds_name=ds_name, **kwargs: ap_gym.ActiveRegressionVectorLogWrapper(
                                TactileShapeReconstructionVectorEnv(
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
                        )

            for sensor_type_name, sensor_type in [
                ("", "taxim"),
                ("-Depth", "depth"),
            ]:
                ap_gym.register(
                    id=f"Starstruck{sensor_type_name}{s}-v0",
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

                for env_name, ds_name, smallest_dim_up, allow_sensor_rotation, step_limit in [
                    ("ABC", "abc-dataset-small", True, True, 32),
                    ("TactileMNIST", "mnist3d", False, False, 16),
                ]:
                    # Snap variants are only registered for the TactileMNIST environments
                    snap_variants = [("", None)]
                    if env_name == "TactileMNIST":
                        snap_variants.append(
                            ("Snap", mk_snap_touch_positions(step_limit))
                        )
                    for snap_suffix, snap_touch_positions in snap_variants:
                        ap_gym.register(
                            id=f"{env_name}CenterOfMass{snap_suffix}{sensor_type_name}{s}-v0",
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
                        )

    # Minecraft item meshes are generated on the fly from Mojang's official
    # assets; the resulting dataset has no train/test split
    minecraft_items = lambda split: load_minecraft_item_mesh_dataset()

    for sensor_type_name, sensor_type in [
        ("", "taxim"),
        ("-Depth", "depth"),
    ]:
        ap_gym.register(
            id=f"Minecraft{sensor_type_name}-v0",
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

        ap_gym.register(
            id=f"MinecraftShape{sensor_type_name}-v0",
            entry_point=lambda *args, default_config, config=None, **kwargs: ap_gym.ActiveRegressionLogWrapper(
                TactileShapeReconstructionEnv(
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
            vector_entry_point=lambda *args, default_config, config=None, **kwargs: ap_gym.ActiveRegressionVectorLogWrapper(
                TactileShapeReconstructionVectorEnv(
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
                    cell_size=CELL_SIZE,
                    smallest_dimension_up=False,
                    renderer_show_orig_mesh_colors=True,
                )
            ),
        )

        for env_name, ds_name, sizes, step_limit, orig_colors in [
            ("Toolbox", "wrench", (("", 0.3), ("-small", 0.25)), 64, False),
            ("MinecraftPose", minecraft_items, (("", 0.2),), 64, True),
        ]:
            for size_name, size in sizes:
                ap_gym.register(
                    id=f"{env_name}{size_name}{sensor_type_name}-v0",
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

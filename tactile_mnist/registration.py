from typing import Any, Iterable

from datasets import load_dataset

import ap_gym
from tactile_mnist.tactile_volume_estimation_env import (
    TactileVolumeEstimationVectorEnv,
    TactileVolumeEstimationEnv,
)
from .constants import *
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


def mk_config(
    dataset_name: str,
    split: str,
    args: Iterable[Any],
    default_config: dict[str, Any],
    config: dict[str, Any] | None = None,
    mesh_dataset_config: dict[str, Any] | None = None,
):
    return TactilePerceptionConfig(
        SimpleMeshDataset(
            load_dataset(f"TimSchneider42/tactile-mnist-{dataset_name}", split=split),
            **({} if mesh_dataset_config is None else mesh_dataset_config),
        ),
        *args,
        **default_config,
        **({} if config is None else config),
    )


def register_envs():
    for split in ["train", "test"]:
        suffixes = [f"-{split}"]
        if split == "train":
            suffixes.append("")
        for s in suffixes:
            for sensor_type_name, sensor_type in [
                ("", "taxim"),
                ("-CycleGAN", "cycle_gan"),
                ("-Depth", "depth"),
            ]:
                ap_gym.register(
                    id=f"TactileMNIST{sensor_type_name}{s}-v0",
                    entry_point=lambda *args, default_config, config=None, _split=split, **kwargs: ap_gym.ActiveClassificationLogWrapper(
                        TactileClassificationEnv(
                            mk_config("mnist3d", _split, args, default_config, config),
                            **kwargs,
                        )
                    ),
                    vector_entry_point=lambda *args, default_config, config=None, _split=split, **kwargs: ap_gym.ActiveClassificationVectorLogWrapper(
                        TactileClassificationVectorEnv(
                            mk_config("mnist3d", _split, args, default_config, config),
                            **kwargs,
                        ),
                    ),
                    kwargs=dict(
                        default_config=dict(
                            sensor_output_size=(64, 64),
                            allow_sensor_rotation=False,
                            max_initial_angle_perturbation=np.pi / 8,
                            renderer_show_class_weights=True,
                            sensor_type=sensor_type,
                        )
                    ),
                )

                for env_name, ds_name, smallest_dim_up, allow_sensor_rotation in [
                    ("TactileMNIST", "mnist3d", False, False),
                    ("ABC", "abc-dataset-small", True, True),
                ]:
                    ap_gym.register(
                        id=f"{env_name}Volume{sensor_type_name}{s}-v0",
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
                                step_limit=32,
                                sensor_type=sensor_type,
                                cell_size=CELL_SIZE,
                                smallest_dimension_up=smallest_dim_up,
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

                for env_name, ds_name, smallest_dim_up in [
                    ("ABC", "abc-dataset-small", True),
                ]:
                    ap_gym.register(
                        id=f"{env_name}CenterOfMass{sensor_type_name}{s}-v0",
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
                                allow_sensor_rotation=True,
                                step_limit=32,
                                cell_size=CELL_SIZE,
                                sensor_type=sensor_type,
                                smallest_dimension_up=smallest_dim_up,
                            ),
                            frame_position_mode="inertia_frame",
                            frame_rotation_mode=None,
                        ),
                    )

    for sensor_type_name, sensor_type in [
        ("", "taxim"),
        ("-Depth", "depth"),
    ]:
        for env_name, ds_name, sizes, step_limit, orig_colors in [
            ("Toolbox", "wrench", (("", 0.3), ("-small", 0.25)), 64, False),
            ("Minecraft", "minecraft-items-dedup", (("", 0.2),), 32, True),
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

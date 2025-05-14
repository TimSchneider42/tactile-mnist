from typing import Any, Iterable

import gymnasium as gym
from datasets import load_dataset

import ap_gym
from tactile_mnist.tactile_volume_estimation_env import (
    TactileVolumeEstimationVectorEnv,
    TactileVolumeEstimationEnv,
)
from .constants import *
from .mesh_dataset import MeshDataset
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
        MeshDataset(
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
                gym.envs.registration.register(
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

                gym.envs.registration.register(
                    id=f"TactileMNISTVolume{sensor_type_name}{s}-v0",
                    entry_point=lambda *args, default_config, config=None, _split=split, **kwargs: ap_gym.ActiveRegressionLogWrapper(
                        TactileVolumeEstimationEnv(
                            mk_config("mnist3d", _split, args, default_config, config),
                            **kwargs,
                        )
                    ),
                    vector_entry_point=lambda *args, default_config, config=None, _split=split, **kwargs: ap_gym.ActiveRegressionVectorLogWrapper(
                        TactileVolumeEstimationVectorEnv(
                            mk_config("mnist3d", _split, args, default_config, config),
                            **kwargs,
                        ),
                    ),
                    kwargs=dict(
                        default_config=dict(
                            sensor_output_size=(64, 64),
                            allow_sensor_rotation=False,
                            step_limit=32,
                            sensor_type=sensor_type,
                        )
                    ),
                )

            for sensor_type_name, sensor_type in [
                ("", "taxim"),
                ("-Depth", "depth"),
            ]:
                gym.envs.registration.register(
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

    for size_name, size in [("", 0.3), ("-small", 0.25)]:
        for sensor_type_name, sensor_type in [
            ("", "taxim"),
            ("-Depth", "depth"),
        ]:
            gym.envs.registration.register(
                id=f"Toolbox{size_name}{sensor_type_name}-v0",
                entry_point=lambda *args, default_config, config=None, **kwargs: ap_gym.ActiveRegressionLogWrapper(
                    TactilePoseEstimationEnv(
                        mk_config(
                            f"wrench",
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
                    TactilePoseEstimationVectorEnv(
                        mk_config(
                            f"wrench",
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
                        step_limit=64,
                        cell_size=(size, size),
                        cell_padding=tuple(
                            np.array([0.005, 0.005]) + GELSIGHT_MINI_OUTER_SIZE / 2
                        ),
                        sensor_type=sensor_type,
                    ),
                    frame_position_mode="model",
                    frame_rotation_mode="model",
                ),
            )

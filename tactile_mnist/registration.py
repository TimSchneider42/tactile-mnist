from typing import Any, Iterable

import gymnasium as gym

import ap_gym
from .constants import *
from .mesh_dataset import MeshDataset
from .resource import get_remote_resource
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
    args: Iterable[Any],
    default_config: dict[str, Any],
    config: dict[str, Any] | None = None,
    mesh_dataset_config: dict[str, Any] | None = None,
):
    return TactilePerceptionConfig(
        MeshDataset.load(
            get_remote_resource(dataset_name),
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
            gym.envs.registration.register(
                id=f"TactileMNIST{s}-v0",
                entry_point=lambda *args, default_config, config=None, _split=split, **kwargs: ap_gym.ActiveClassificationLogWrapper(
                    TactileClassificationEnv(
                        mk_config(f"mnist3d-v0/{_split}", args, default_config, config),
                        **kwargs,
                    )
                ),
                vector_entry_point=lambda *args, default_config, config=None, _split=split, **kwargs: ap_gym.ActiveClassificationVectorLogWrapper(
                    TactileClassificationVectorEnv(
                        mk_config(f"mnist3d-v0/{_split}", args, default_config, config),
                        **kwargs,
                    ),
                ),
                kwargs=dict(
                    default_config=dict(
                        sensor_output_size=(64, 64),
                        allow_sensor_rotation=False,
                        max_initial_angle_perturbation=np.pi / 8,
                        renderer_show_class_weights=True,
                    )
                ),
            )

            gym.envs.registration.register(
                id=f"Starstruck{s}-v0",
                entry_point=lambda *args, default_config, config=None, _split=split, **kwargs: ap_gym.ActiveClassificationLogWrapper(
                    TactileClassificationEnv(
                        mk_config(
                            f"starstruck-v0/{_split}", args, default_config, config
                        ),
                        **kwargs,
                    )
                ),
                vector_entry_point=lambda *args, default_config, config=None, _split=split, **kwargs: ap_gym.ActiveClassificationVectorLogWrapper(
                    TactileClassificationVectorEnv(
                        mk_config(
                            f"starstruck-v0/{_split}", args, default_config, config
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
                    ),
                ),
            )

    for size_name, size in [("", 0.3), ("-small", 0.25)]:
        gym.envs.registration.register(
            id=f"Toolbox{size_name}-v0",
            entry_point=lambda *args, default_config, config=None, **kwargs: ap_gym.ActiveRegressionLogWrapper(
                TactilePoseEstimationEnv(
                    mk_config(
                        f"wrench-v0",
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
                        f"wrench-v0",
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
                    max_tilt_angle=np.pi,
                    cell_padding=tuple(
                        np.array([0.005, 0.005]) + GELSIGHT_MINI_OUTER_SIZE / 2
                    ),
                ),
                frame_position_mode="model",
                frame_rotation_mode="model",
            ),
        )

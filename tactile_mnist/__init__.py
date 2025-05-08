from .touch_data import (
    TouchMetadata,
    TouchSeqMetadata,
    TouchSingleMetadata,
    TouchData,
    TouchSeq,
    TouchSingle,
    BaseTouchDataset,
    LoadedTouchDataset,
    TouchDataset,
    BaseTouchDatasetSeq,
    BaseTouchDatasetSingle,
)
from .data_loader import (
    TerminateSignal,
    MultiThreadedPipeline,
    BufferedDataLoader,
    TouchDatasetRoundIterator,
    TouchDatasetRoundSubsampler,
    TouchDatasetDataLoader,
    TouchDatasetDataPointCreator,
    TouchDatasetRoundLoader,
)
from .mesh_dataset import MeshMetadata, MeshDataPoint, MeshDataset
from .constants import (
    CELL_SIZE,
    CELL_PADDING,
    GRID_BORDER_THICKNESS,
    GELSIGHT_MINI_OUTER_SIZE,
    GELSIGHT_MINI_GEL_THICKNESS_MM,
    GELSIGHT_MINI_IMAGE_SIZE_PX,
)
from .dataset import PartialDataPoint, Dataset
from .iterable_types import (
    SeekableIterable,
    SizedIterable,
    SeekableSizedIterable,
    ShiftedSizedIterable,
)
from .resource import Resource, get_resource, get_remote_resource
from .constants import *
from .tactile_perception_vector_env import (
    TactilePerceptionVectorEnv,
    TactilePerceptionConfig,
)
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


def register_envs():
    import gymnasium as gym
    import ap_gym

    for split in ["train", "test"]:
        suffixes = [f"-{split}"]
        if split == "train":
            suffixes.append("")
        for s in suffixes:
            gym.envs.registration.register(
                id=f"TactileMNIST{s}-v0",
                entry_point=lambda *args, default_config, config=None, _split=split, **kwargs: ap_gym.ActiveClassificationLogWrapper(
                    TactileClassificationEnv(
                        TactilePerceptionConfig(
                            MeshDataset.load(
                                get_remote_resource(f"mnist3d-v0/{_split}")
                            ),
                            *args,
                            **default_config,
                            **({} if config is None else config),
                        ),
                        **kwargs,
                    )
                ),
                vector_entry_point=lambda *args, default_config, config=None, _split=split, **kwargs: ap_gym.ActiveClassificationVectorLogWrapper(
                    TactileClassificationVectorEnv(
                        TactilePerceptionConfig(
                            MeshDataset.load(
                                get_remote_resource(f"mnist3d-v0/{_split}")
                            ),
                            *args,
                            **default_config,
                            **({} if config is None else config),
                        ),
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
                        TactilePerceptionConfig(
                            MeshDataset.load(
                                get_remote_resource(f"starstruck-v0/{_split}")
                            ),
                            *args,
                            **default_config,
                            **({} if config is None else config),
                        ),
                        **kwargs,
                    )
                ),
                vector_entry_point=lambda *args, default_config, config=None, _split=split, **kwargs: ap_gym.ActiveClassificationVectorLogWrapper(
                    TactileClassificationVectorEnv(
                        TactilePerceptionConfig(
                            MeshDataset.load(
                                get_remote_resource(f"starstruck-v0/{_split}")
                            ),
                            *args,
                            **default_config,
                            **({} if config is None else config),
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
                    TactilePerceptionConfig(
                        MeshDataset.load(
                            get_remote_resource(f"wrench-v0"), cache_size="full"
                        ),
                        *args,
                        **default_config,
                        **({} if config is None else config),
                    ),
                    **kwargs,
                )
            ),
            vector_entry_point=lambda *args, default_config, config=None, **kwargs: ap_gym.ActiveRegressionVectorLogWrapper(
                TactilePoseEstimationVectorEnv(
                    TactilePerceptionConfig(
                        MeshDataset.load(
                            get_remote_resource(f"wrench-v0"), cache_size="full"
                        ),
                        *args,
                        **default_config,
                        **({} if config is None else config),
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


register_envs()

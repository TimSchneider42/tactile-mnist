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
    CELL_MARGIN,
    GRID_BORDER_THICKNESS,
    GELSIGHT_DIMS,
    GELSIGHT_GEL_THICKNESS_MM,
    GELSIGHT_IMAGE_SIZE_PX,
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
from .tactile_classification_env import TactileClassificationEnv
from .tactile_classification_vector_env import TactileClassificationVectorEnv


def register_envs():
    import gymnasium as gym

    for split in ["train", "test"]:
        suffixes = [f"-{split}"]
        if split == "train":
            suffixes.append("")
        for s in suffixes:
            gym.envs.registration.register(
                id=f"TactileMNIST{s}-v0",
                entry_point=lambda *args, **kwargs: TactileClassificationEnv(
                    MeshDataset.load(get_remote_resource(f"mnist3d_split_v0/{split}")),
                    *args,
                    **kwargs,
                ),
                vector_entry_point=lambda *args, **kwargs: TactileClassificationVectorEnv(
                    MeshDataset.load(get_remote_resource(f"mnist3d_split_v0/{split}")),
                    *args,
                    **kwargs,
                ),
                kwargs=dict(sensor_output_size=(64, 64), allow_sensor_rotation=False),
                max_episode_steps=16,
            )

            gym.envs.registration.register(
                id=f"Starstruck{s}-v0",
                entry_point=lambda *args, **kwargs: TactileClassificationEnv(
                    MeshDataset.load(
                        get_remote_resource(f"starstruck_split_v0/{split}")
                    ),
                    *args,
                    **kwargs,
                ),
                vector_entry_point=lambda *args, **kwargs: TactileClassificationVectorEnv(
                    MeshDataset.load(
                        get_remote_resource(f"starstruck_split_v0/{split}")
                    ),
                    *args,
                    **kwargs,
                ),
                kwargs=dict(
                    sensor_output_size=(64, 64),
                    allow_sensor_rotation=False,
                    randomize_initial_object_pose=False,
                    perturb_object_pose=False,
                ),
                max_episode_steps=32,
            )


register_envs()

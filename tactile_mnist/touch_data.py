from __future__ import annotations

import json
import pickle
import shutil
import warnings
from abc import abstractmethod
from contextlib import nullcontext, contextmanager
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from tarfile import TarFile
from tempfile import TemporaryDirectory, NamedTemporaryFile
from typing import (
    List,
    Any,
    TypeVar,
    Generic,
    Sized,
    Set,
    Sequence,
    Literal,
    ClassVar,
    Callable,
    overload,
)

import cv2
import numpy as np

try:
    import torch
    import torchvision.io.image
except ImportError:
    torch = torchvision = None
from transformation import Transformation

from .fast_tar_file_reader import FastTarFileReader
from .dataset import PartialDataPoint, Dataset, PartialDataPointType, FullDataPointType
from .storage_utils import save_data, load_data


class DataPointLoadError(Exception):
    pass


DataPointType = TypeVar("DataPointType", bound="TouchData")
DataType = TypeVar("DataType")


@dataclass(frozen=True)
class TouchMetadata(PartialDataPoint[DataPointType], Generic[DataPointType, DataType]):
    label: int
    pos_in_cell: np.ndarray
    object_id: Any
    round_id: str | int
    touch_no: int
    info: dict[str, Any]

    SUPPORTED_FILE_FORMATS: ClassVar[tuple[str, ...]] = ("pkl", "npz")

    @classmethod
    def store(
        cls,
        output_dir: Path,
        data_points: Sequence[TouchMetadata],
        format: Literal["pkl", "npz"] = "npz",
    ):
        if format == "pkl":
            with (output_dir / "metadata.pkl").open("wb") as f:
                pickle.dump(data_points, f)
        else:
            meta_metadata = {"type": type(data_points[0]).__name__}
            save_data(output_dir / "metadata.npz", data_points, metadata=meta_metadata)
        with (output_dir / "summary.json").open("w") as f:
            json.dump(
                {
                    "length": len(data_points),
                    "round_count": len(set(dp.round_id for dp in data_points)),
                    "labels": sorted(set(dp.label for dp in data_points)),
                },
                f,
            )

    def copy_data(self, source: Path | TarFile, destination: Path | TarFile):
        if isinstance(source, Path) and isinstance(destination, Path):
            shutil.copy2(source / self.data_filename, destination / self.data_filename)
        elif isinstance(source, Path) and isinstance(destination, TarFile):
            destination.add(source / self.data_filename, arcname=self.data_filename)
        elif isinstance(source, TarFile) and isinstance(destination, Path):
            source.extract(self.data_filename, destination)
        else:
            destination.addfile(
                source.getmember(self.data_filename),
                source.extractfile(self.data_filename),
            )

    @contextmanager
    def _ensure_extracted(self, data_source: Path | FastTarFileReader) -> Path:
        if isinstance(data_source, FastTarFileReader):
            temp_file = NamedTemporaryFile()
        else:
            temp_file = nullcontext()
        with temp_file:
            if isinstance(data_source, FastTarFileReader):
                temp_path = Path(temp_file.name)
                with temp_path.open("wb") as f_out:
                    with data_source.extractfile(self.data_filename) as f_in:
                        while len(data := f_in.read(2**16)) > 0:
                            f_out.write(data)
                yield temp_path
            else:
                yield data_source / self.data_filename

    @abstractmethod
    def _get_data_file_extension(self) -> str:
        pass

    @abstractmethod
    def load_data(self, data_source: Path | FastTarFileReader) -> DataType:
        pass

    @abstractmethod
    def from_data(self, data: DataType) -> DataPointType:
        pass

    def load_full(self, data_source: Path | FastTarFileReader) -> DataPointType:
        return self.from_data(self.load_data(data_source))

    @staticmethod
    def load(input_dir: Path) -> List[TouchMetadata]:
        pkl_file = input_dir / "metadata.pkl"
        if pkl_file.exists():
            with pkl_file.open("rb") as f:
                return pickle.load(f)
        else:
            data, meta_metadata = load_data(input_dir / "metadata.npz")
            dp_type_str = meta_metadata["type"]
            dp_type = (
                TouchSeqMetadata
                if dp_type_str == "TouchSeqMetadata"
                else TouchSingleMetadata
            )
            return [dp_type(**e) for e in data]

    @property
    def unique_id(self):
        return f"{self.round_id}_{self.touch_no}"

    @property
    def data_file_suffix(self):
        return self._get_data_file_extension()

    @property
    def data_filename(self):
        return f"{self.unique_id}.{self.data_file_suffix}"

    @classmethod
    def is_dataset(cls, path: Path) -> bool:
        return any(
            (path / f"metadata.{ext}").exists()
            for ext in TouchMetadata.SUPPORTED_FILE_FORMATS
        )


@dataclass(frozen=True)
class TouchSeqMetadata(TouchMetadata["TouchSeq", np.ndarray]):
    touch_start_time_rel: float
    touch_end_time_rel: float
    time_stamp_rel_seq: np.ndarray
    gel_position_cell_frame_seq: np.ndarray
    gel_orientation_cell_frame_seq: np.ndarray

    def load_data(self, data_source: Path | TarFile) -> np.ndarray:
        try:
            with self._ensure_extracted(data_source) as data_path:
                if torch is None:
                    video_reader = cv2.VideoCapture(str(data_path))
                    frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
                    width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    output = np.empty((frames, height, width, 3), dtype=np.uint8)
                    for i in range(frames):
                        output[i] = cv2.cvtColor(
                            video_reader.read()[1], cv2.COLOR_BGR2RGB
                        )
                else:
                    # pts_unit="sec" is needed to suppress a warning here
                    output = torchvision.io.read_video(
                        data_path, pts_unit="sec", output_format="THWC"
                    )[0].numpy()
                assert (
                    output.shape[0]
                    == len(self.time_stamp_rel_seq)
                    == len(self.gel_pose_cell_frame_seq)
                )
            return output
        except Exception:
            raise DataPointLoadError(
                f"Failed to load datapoint {self.data_filename} from {data_source}."
            )

    def from_data(self, data: np.ndarray) -> TouchSeq:
        return TouchSeq(self, data)

    def _get_data_file_extension(self) -> str:
        return "avi"

    @property
    def gel_pose_cell_frame_seq(self) -> List[Transformation]:
        return [
            Transformation.from_pos_quat(p, q)
            for p, q in zip(
                self.gel_position_cell_frame_seq, self.gel_orientation_cell_frame_seq
            )
        ]


@dataclass(frozen=True)
class TouchSingleMetadata(TouchMetadata["TouchSingle", bytes]):
    gel_position_cell_frame: np.ndarray
    gel_orientation_cell_frame: np.ndarray

    def load_data(self, data_source: Path | FastTarFileReader) -> bytes:
        try:
            if isinstance(data_source, FastTarFileReader):
                with data_source.extractfile(self.data_filename) as f:
                    return f.read()
            else:
                with (data_source / self.data_filename).open("rb") as f:
                    return f.read()
        except Exception:
            raise DataPointLoadError(
                f"Failed to load data of datapoint {self.data_filename} from {data_source}."
            )

    def from_data(self, data: bytes) -> TouchSingle:
        try:
            if torch is None:
                img = cv2.cvtColor(
                    cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR),
                    cv2.COLOR_BGR2RGB,
                )
            else:
                # We need this to suppress the warning from torch that data is not writable
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    data_torch = torch.frombuffer(data, dtype=torch.uint8)
                img = (
                    torchvision.io.image.decode_image(data_torch)
                    .permute((1, 2, 0))
                    .numpy()
                )
            return TouchSingle(self, img)
        except Exception:
            raise DataPointLoadError(
                f"Failed to load create datapoint {self.data_filename} from data."
            )

    def _get_data_file_extension(self) -> str:
        return "jpeg"

    @property
    def gel_pose_cell_frame(self) -> Transformation:
        return Transformation.from_pos_quat(
            self.gel_position_cell_frame, self.gel_orientation_cell_frame
        )


MetadataType = TypeVar("MetadataType", bound=TouchMetadata)


@dataclass(frozen=True)
class TouchData(Generic[MetadataType]):
    metadata: MetadataType

    def save_non_metadata(self, destination: Path | TarFile):
        if isinstance(destination, TarFile):
            with NamedTemporaryFile(suffix="." + self.metadata.data_file_suffix) as f:
                path = Path(f.name)
                self._write_data(path)
                destination.add(path, self.metadata.data_filename)
        else:
            self._write_data(destination / self.metadata.data_filename)

    @abstractmethod
    def _write_data(self, path: Path):
        pass


@dataclass(frozen=True)
class TouchSeq(TouchData[TouchSeqMetadata]):
    sensor_image_seq: np.ndarray

    def _write_data(self, path: Path):
        size = self.sensor_image_seq.shape[1:-1]
        frame_time_stamps = np.array(self.metadata.time_stamp_rel_seq)
        fps = int((1 / (frame_time_stamps[1:] - frame_time_stamps[:-1])).mean().round())

        video_writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, tuple(reversed(size))
        )
        for frame in self.sensor_image_seq:
            video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR))
        video_writer.release()


@dataclass(frozen=True)
class TouchSingle(TouchData[TouchSingleMetadata]):
    sensor_image: np.ndarray

    def _write_data(self, path: Path):
        cv2.imwrite(str(path), cv2.cvtColor(self.sensor_image, cv2.COLOR_RGB2BGR))


DatasetType = TypeVar("DatasetType", bound="BaseTouchDataset")
SubDatasetType = TypeVar("SubDatasetType", bound="BaseTouchDataset")


class RoundSequence(Sequence[SubDatasetType], Generic[DatasetType, SubDatasetType]):
    def __init__(self, base_dataset: DatasetType):
        self.__base_dataset = base_dataset
        self.__boundaries = base_dataset.round_boundaries

    @overload
    @abstractmethod
    def __getitem__(self, index: int) -> SubDatasetType:
        ...

    @overload
    @abstractmethod
    def __getitem__(self, index: slice) -> Sequence[SubDatasetType]:
        ...

    def __getitem__(self, index):
        if isinstance(index, slice):
            start = self.__boundaries[index.start]
            end = self.__boundaries[index.start + 1]
            return tuple(self.__base_dataset[s:e] for s, e in zip(start, end))
        else:
            start = self.__boundaries[index]
            end = self.__boundaries[index + 1]
            return self.__base_dataset[start:end]

    def __len__(self):
        return self.__base_dataset.round_count


class BaseTouchDataset(
    Dataset[MetadataType, DataPointType, SubDatasetType],
    Generic[MetadataType, DataPointType, SubDatasetType],
):
    def store(self, path: Path, archive: bool = True):
        path.mkdir()
        TouchMetadata.store(path, self.partial)
        if archive:
            with TarFile(path / "data.tar", "w") as tarfile:
                for i, dp in enumerate(self.metadata):
                    dp.copy_data(self.load_full_kwargs[i]["data_source"], tarfile)
        else:
            for i, dp in enumerate(self.metadata):
                dp.copy_data(self.load_full_kwargs[i]["data_source"], path)

    @property
    def metadata(self) -> tuple[MetadataType, ...]:
        return self.partial

    @cached_property
    def round_ids(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array([md.round_id for md in self.metadata])

    @cached_property
    def unique_round_ids(self) -> np.ndarray:
        return np.unique(self.round_ids)

    @cached_property
    def round_boundaries(self) -> tuple[np.ndarray, np.ndarray]:
        boundaries = np.where(self.round_ids[:-1] != self.round_ids[1:])[0] + 1
        return np.concatenate([[0], boundaries, [len(self.round_ids)]])

    @cached_property
    def rounds(
        self,
    ) -> dict[str | int, BaseTouchDataset[MetadataType, DataPointType, SubDatasetType]]:
        return {
            self.round_ids[s]: self[s:e]
            for s, e in zip(self.round_boundaries[:-1], self.round_boundaries[1:])
        }

    @property
    def round_sequence(
        self,
    ) -> Sequence[BaseTouchDataset[MetadataType, DataPointType, SubDatasetType]]:
        return RoundSequence(self)

    @property
    def round_count(self) -> int:
        return len(self.unique_round_ids)


class _BaseTouchDatasetType(
    BaseTouchDataset[
        MetadataType,
        DataPointType,
        "_BaseTouchDatasetType[MetadataType, DataPointType]",
    ],
    Generic[MetadataType, DataPointType],
):
    pass


BaseTouchDatasetSingle = _BaseTouchDatasetType[TouchSingleMetadata, TouchSingle]
BaseTouchDatasetSeq = _BaseTouchDatasetType[TouchSeqMetadata, TouchSeq]


class LoadedTouchDataset(
    _BaseTouchDatasetType[MetadataType, DataPointType],
    Generic[MetadataType, DataPointType],
):
    def __init__(
        self, path: Path, mode: Literal["in_place", "extract", "in_memory"] = "in_place"
    ):
        self.__tarfile: TarFile | None = None
        self.__temp_dir: TemporaryDirectory | None = None
        self.__path = path

        archive_path = path / "data.tar"
        is_archived = archive_path.exists()
        if is_archived:
            if mode == "extract":
                self.__temp_dir = TemporaryDirectory()
                temp_path = Path(self.__temp_dir.name)
                with TarFile(archive_path) as tarfile:
                    tarfile.extractall(temp_path)
                self.__data_source = temp_path
            else:
                self.__tarfile = FastTarFileReader(
                    archive_path, in_memory=mode == "in_memory"
                ).__enter__()
                self.__data_source = self.__tarfile
        else:
            if mode == "in_memory":
                self.__temp_dir = TemporaryDirectory(dir="/dev/shm")
                temp_path = Path(self.__temp_dir.name)
                for file in path.iterdir():
                    if file.suffix in [".jpeg", ".avi"]:
                        shutil.copy2(file, temp_path / file.name)
                self.__data_source = temp_path
            else:
                self.__data_source = path
        metadata = TouchMetadata.load(path)
        super().__init__(metadata, load_full_kwargs={"data_source": self.__data_source})

    def release(self):
        if self.__temp_dir is not None:
            self.__temp_dir.cleanup()
        self.__temp_dir = None
        if self.__tarfile is not None:
            self.__tarfile.__exit__(None, None, None)
        self.__tarfile = None

    @classmethod
    def _instantiate(
        cls,
        partial_data_points: tuple[PartialDataPointType, ...],
        load_full_kwargs: tuple[dict[str, Any], ...],
        cached_get_item_func: Callable[[int, int], FullDataPointType] | None = None,
    ):
        return BaseTouchDataset(
            partial_data_points,
            load_full_kwargs,
            _cached_get_item_func=cached_get_item_func,
        )


class TouchDataset(Sized, Generic[MetadataType, DataPointType]):
    def __init__(
        self, path: Path, mode: Literal["in_place", "extract", "in_memory"] = "in_place"
    ):
        self.__path = path
        self.__dataset: LoadedTouchDataset[MetadataType, DataPointType] | None = None
        self.__mode = mode
        with (path / "summary.json").open() as f:
            summary = json.load(f)
        self.__length: int = summary["length"]
        self.__round_count: int = summary["round_count"]
        self.__labels: Set[Any] = set(summary["labels"])

    def __enter__(self) -> LoadedTouchDataset[MetadataType, DataPointType]:
        self.__dataset = LoadedTouchDataset(self.__path, mode=self.__mode)
        return self.__dataset

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__dataset.release()
        self.__dataset = None

    def __len__(self) -> int:
        return self.__length

    @property
    def round_count(self) -> int:
        return self.__round_count

    @property
    def labels(self) -> Set[Any]:
        return self.__labels

    @property
    def path(self) -> Path:
        return self.__path

    @property
    def mode(self) -> Literal["in_place", "extract", "in_memory"]:
        return self.__mode

    @staticmethod
    def open_all(
        path: Path, mode: Literal["in_place", "extract", "in_memory"] = "in_place"
    ) -> List[TouchDataset]:
        return [
            TouchDataset(p, mode=mode)
            for p in path.glob("**/*")
            if TouchMetadata.is_dataset(p)
        ]

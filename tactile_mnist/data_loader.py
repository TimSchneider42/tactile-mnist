from __future__ import annotations

import concurrent
import os
from abc import abstractmethod, ABC
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from queue import Queue, Full
from threading import Thread
from typing import (
    Iterable,
    TypeVar,
    Generic,
    Iterator,
    Sequence,
    Any,
)

import numpy as np

from .iterable_types import SeekableSizedIterable
from .peekable_queue import PeekableQueue
from .touch_data import (
    _BaseTouchDatasetType,
    TouchMetadata,
    TouchDataset,
    LoadedTouchDataset,
)

DataPointType = TypeVar("DataPointType", bound="TouchData")
DataType = TypeVar("DataType")


@dataclass
class TerminateSignal:
    set: bool = False


InputDatapointType = TypeVar("InputDatapointType")
OutputDatapointType = TypeVar("OutputDatapointType")
MetadataType = TypeVar("MetadataType", bound=TouchMetadata)


class MultiThreadedPipeline(
    SeekableSizedIterable[OutputDatapointType],
    Generic[InputDatapointType, OutputDatapointType],
    ABC,
):
    def __init__(
        self,
        data: SeekableSizedIterable[InputDatapointType],
        num_workers: int | None = None,
    ):
        self.__data = data
        self.__num_workers = os.cpu_count() if num_workers is None else num_workers

    @abstractmethod
    def _process(self, datapoint: InputDatapointType) -> OutputDatapointType:
        pass

    def __queue_feeder(
        self,
        start: int,
        queue: Queue[concurrent.futures.Future | Exception | None],
        executor: ThreadPoolExecutor,
        terminate_signal: TerminateSignal,
    ):
        try:
            for dp in self.data.iter(start=start):
                success = False
                future = executor.submit(self._process, dp)
                while not success:
                    if terminate_signal.set:
                        return
                    try:
                        queue.put(future, timeout=0.1)
                        success = True
                    except Full:
                        pass
            queue.put(None)
        except Exception as ex:
            queue.put(ex)

    def _iter(self, start: int):
        with ThreadPoolExecutor(self.__num_workers) as executor:
            queue: Queue[concurrent.futures.Future | None] = Queue(
                maxsize=self.__num_workers
            )
            terminate_signal = TerminateSignal()
            feeder = Thread(
                target=self.__queue_feeder,
                args=(start, queue, executor, terminate_signal),
            )
            try:
                feeder.start()
                while (data := queue.get()) is not None:
                    if isinstance(data, Exception):
                        raise data
                    yield data.result()
            finally:
                terminate_signal.set = True
                feeder.join()

    def __len__(self):
        return len(self.data)

    @property
    def data(self):
        return self.__data

    @property
    def num_workers(self):
        return self.__num_workers


@dataclass(frozen=True)
class BufferedDataLoader(SeekableSizedIterable[DataPointType], Generic[DataPointType]):
    data: SeekableSizedIterable[DataPointType]
    buffer_size: int = 3

    def _worker_func(
        self, output_queue: Queue, terminate_signal: TerminateSignal, start: int
    ):
        try:
            for dp in self.data.iter(start=start):
                success = False
                while not success:
                    if terminate_signal.set:
                        return
                    try:
                        output_queue.put(dp, timeout=0.1)
                        success = True
                    except Full:
                        pass
            output_queue.put(None)
        except Exception as ex:
            output_queue.put(ex)

    def _iter(self, start: int):
        terminate_signal = TerminateSignal()
        queue = Queue(maxsize=self.buffer_size)
        worker = Thread(target=self._worker_func, args=(queue, terminate_signal, start))
        try:
            worker.start()
            while (next_elem := queue.get()) is not None:
                if isinstance(next_elem, Exception):
                    raise next_elem
                yield next_elem
        finally:
            terminate_signal.set = True
            if worker.is_alive():
                worker.join()

    def __len__(self):
        return len(self.data)


class TouchDatasetRoundIterator(
    SeekableSizedIterable[_BaseTouchDatasetType[MetadataType, DataPointType]],
    Generic[MetadataType, DataPointType],
):
    @dataclass
    class _TerminateSignal:
        set: bool = False

    def __init__(
        self,
        datasets: Iterable[TouchDataset[MetadataType, DataPointType]],
        seed: int | None = None,
        dataset_prefetch_count: int = 0,
        shuffle: bool = False,
    ):
        self.__datasets = list(datasets)
        self.__seed = seed
        self.__dataset_prefetch_count = dataset_prefetch_count
        self.__shuffle = shuffle

    def _load_datasets(
        self,
        datasets: Iterable[TouchDataset[MetadataType, DataPointType]],
        ds_seeds: Iterable[int],
        output_queue: PeekableQueue,
        terminate_signal: _TerminateSignal,
    ):
        def load_ds(
            ds: TouchDataset[MetadataType, DataPointType], ds_seed: int
        ) -> tuple[
            LoadedTouchDataset[MetadataType, DataPointType],
            Any,
            Sequence[_BaseTouchDatasetType[MetadataType, DataPointType]],
        ]:
            ds_loaded = ds.__enter__()
            rng = np.random.default_rng(ds_seed)
            rounds = ds_loaded.round_sequence
            indices = np.arange(len(rounds))
            if self.__shuffle:
                indices = rng.permutation(indices)
            return ds_loaded, indices, rounds

        with ThreadPoolExecutor(
            max_workers=self.__dataset_prefetch_count + 1
        ) as executor:
            try:
                for ds, seed in zip(datasets, ds_seeds):
                    if terminate_signal.set:
                        return
                    output_queue.put((ds, executor.submit(load_ds, ds, seed)))
            except Exception as ex:
                output_queue.put(ex)
            finally:
                output_queue.put(None)

    def _iter(
        self, start: int
    ) -> Iterator[_BaseTouchDatasetType[MetadataType, DataPointType]]:
        rng = np.random.default_rng(self.__seed)
        datasets = self.__datasets
        if self.__shuffle:
            datasets = [datasets[i] for i in rng.permutation(np.arange(len(datasets)))]
        dataset_seeds = rng.integers(0, 2**32 - 1, size=len(datasets), endpoint=True)
        dataset_queue = PeekableQueue(maxsize=self.__dataset_prefetch_count)
        terminate_signal = self._TerminateSignal()
        dataset_start_index = np.searchsorted(
            np.cumsum([ds.round_count for ds in datasets]), start, side="right"
        )
        dataset_feeder = Thread(
            target=self._load_datasets,
            args=(
                datasets[dataset_start_index:],
                dataset_seeds[dataset_start_index:],
                dataset_queue,
                terminate_signal,
            ),
        )
        try:
            dataset_feeder.start()
            processed_data_points = sum(
                [ds.round_count for ds in datasets[:dataset_start_index]]
            )
            while (ds_tuple := dataset_queue.peek()) is not None:
                if isinstance(ds_tuple, Exception):
                    raise ds_tuple
                ds, ds_loaded_tuple_future = ds_tuple
                ds_loaded, indices, rounds = ds_loaded_tuple_future.result()
                start_index = max(start - processed_data_points, 0)
                for index in indices[start_index:]:
                    yield rounds[index]
                ds.__exit__(None, None, None)
                dataset_queue.get()
                processed_data_points += len(rounds)
        finally:
            terminate_signal.set = True
            termination_acknowledged = False
            exception = None
            while not termination_acknowledged:
                ds_tuple = dataset_queue.get()
                if ds_tuple is None:
                    termination_acknowledged = True
                elif isinstance(ds_tuple, Exception):
                    exception = ds_tuple
                else:
                    ds, ds_loaded_tuple_future = ds_tuple
                    ds_loaded_tuple_future.result()
                    ds.__exit__(None, None, None)
            dataset_feeder.join()
            if exception is not None:
                raise exception

    def __len__(self) -> int:
        return sum(ds.round_count for ds in self.__datasets)


@dataclass(frozen=True)
class TouchDatasetRoundSubsampler(
    SeekableSizedIterable[_BaseTouchDatasetType[MetadataType, DataPointType]]
):
    data: SeekableSizedIterable[_BaseTouchDatasetType[MetadataType, DataPointType]]
    step_count: int
    seed: int = 0

    def _iter(self, start: int):
        rng = np.random.default_rng(self.seed)
        for data_round in self.data.iter(start=start):
            idx = np.sort(
                rng.choice(
                    np.arange(len(data_round)), size=self.step_count, replace=False
                )
            )
            yield data_round[idx]

    def __len__(self) -> int:
        return len(self.data)


@dataclass(frozen=True)
class TouchDatasetDataLoader(
    SeekableSizedIterable[tuple[tuple[MetadataType, DataType], ...]],
    Generic[MetadataType, DataType],
):
    data: SeekableSizedIterable[_BaseTouchDatasetType[MetadataType, DataPointType]]
    num_workers: int | None = None
    use_multithreading: bool = True

    def _iter(self, start: int):
        if self.use_multithreading:
            num_workers = (
                os.cpu_count() if self.num_workers is None else self.num_workers
            )
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                for data_round in self.data.iter(start=start):
                    yield tuple(
                        executor.map(
                            lambda dp, kw: (dp, dp.load_data(**kw)),
                            data_round.metadata,
                            data_round.load_full_kwargs,
                            chunksize=len(data_round.metadata) // num_workers,
                        )
                    )
        else:
            for data_round in self.data.iter(start=start):
                yield tuple(
                    map(
                        lambda dp, kw: (dp, dp.load_data(**kw)),
                        data_round.metadata,
                        data_round.load_full_kwargs,
                    )
                )

    def __len__(self) -> int:
        return len(self.data)


class TouchDatasetDataPointCreator(
    MultiThreadedPipeline[
        tuple[tuple[MetadataType, DataType], ...], tuple[DataPointType, ...]
    ],
    Generic[MetadataType, DataType, DataPointType],
):
    def __init__(
        self,
        data: SeekableSizedIterable[tuple[tuple[MetadataType, DataType], ...]],
        num_workers_inner: int | None = None,
        num_workers_outer: int | None = None,
        use_per_round_multithreading: bool = True,
    ):
        super().__init__(data, num_workers_outer)
        self.__num_workers_inner = (
            os.cpu_count() if num_workers_inner is None else num_workers_inner
        )
        if use_per_round_multithreading:
            self.__executor = ThreadPoolExecutor(
                max_workers=self.__num_workers_inner * self.num_workers
            )
        else:
            self.__executor = None

    def _process(
        self, datapoint: tuple[tuple[MetadataType, DataType], ...]
    ) -> tuple[DataPointType, ...]:
        if self.__executor is not None:
            return tuple(
                self.__executor.map(
                    lambda dp: dp[0].from_data(dp[1]),
                    datapoint,
                    chunksize=len(datapoint) // self.__num_workers_inner,
                )
            )
        else:
            return tuple(map(lambda dp: dp[0].from_data(dp[1]), datapoint))


def TouchDatasetRoundLoader(
    data: SeekableSizedIterable[_BaseTouchDatasetType[MetadataType, DataPointType]],
    num_workers: int | None = None,
    use_per_round_multithreading: bool = True,
):
    itr = BufferedDataLoader(
        TouchDatasetDataLoader(
            data,
            num_workers=num_workers,
            use_multithreading=use_per_round_multithreading,
        )
    )
    return TouchDatasetDataPointCreator(
        itr,
        num_workers_inner=num_workers,
        num_workers_outer=num_workers,
        use_per_round_multithreading=use_per_round_multithreading,
    )

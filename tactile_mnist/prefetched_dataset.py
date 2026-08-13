from __future__ import annotations

import threading
from collections import deque
from typing import Any, Callable, Generic, Iterable, TypeVar

from .dataset import Dataset

DataPointType = TypeVar("DataPointType")


class PrefetchedDataset(Generic[DataPointType]):
    """
    Wrapper around a Dataset that loads data points ahead of time in a background thread.

    Indices passed to prefetch() are appended to an unbounded input queue, from which a loader thread loads them one
    by one into a bounded output queue. The wrapper assumes that elements are collected via __getitem__ in the same
    order in which they were requested, in which case __getitem__ just hands out the head of the output queue (or
    blocks until it is loaded). capacity controls how many loaded data points may sit in the output queue beyond the
    next one, so with the default of 0, only the next requested element is prefetched.

    Out-of-order collection is supported as a fallback: elements preceding the requested one are discarded from the
    output queue and then from the input queue until the requested element appears. If it was never requested, both
    queues are cleared entirely and the element is loaded synchronously on the fly.

    Prefetching is only active while the wrapper is open (see open() and close(), or use it as a context manager).
    When closed, __getitem__ loads through the wrapped dataset directly.

    Since data point fields load lazily, load_fn should touch the expensive fields to materialize them in the loader
    thread (e.g. ``lambda dp: dp.mesh`` for mesh datasets).
    """

    def __init__(
        self,
        dataset: Dataset[DataPointType, Any],
        capacity: int = 0,
        load_fn: Callable[[DataPointType], Any] | None = None,
    ):
        self.__dataset = dataset
        self.__capacity = capacity
        self.__load_fn = load_fn
        self.__cond = threading.Condition()
        self.__input_queue: deque[int] = deque()
        self.__output_queue: deque[
            tuple[int, DataPointType | None, BaseException | None]
        ] = deque()
        self.__loading_index: int | None = None
        self.__discard_loading = False
        self.__loader_thread: threading.Thread | None = None
        self.__open = False

    def open(self) -> "PrefetchedDataset[DataPointType]":
        """Start the loader thread, enabling prefetching."""
        with self.__cond:
            assert not self.__open, "PrefetchedDataset is already open."
            self.__open = True
            self.__loader_thread = threading.Thread(
                target=self.__loader,
                name=f"{type(self.__dataset).__name__}-prefetch",
                daemon=True,
            )
            self.__loader_thread.start()
        return self

    def close(self):
        """Drop all scheduled loads and shut the loader thread down. A no-op if not open."""
        with self.__cond:
            loader_thread = self.__loader_thread
            self.__loader_thread = None
            self.__open = False
            self.__input_queue.clear()
            self.__output_queue.clear()
            self.__cond.notify_all()
        if loader_thread is not None:
            loader_thread.join()

    def __enter__(self) -> "PrefetchedDataset[DataPointType]":
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __loader(self):
        while True:
            with self.__cond:
                self.__cond.wait_for(
                    lambda: not self.__open
                    or (
                        len(self.__input_queue) > 0
                        and len(self.__output_queue) <= self.__capacity
                    )
                )
                if not self.__open:
                    return
                index = self.__input_queue.popleft()
                self.__loading_index = index
                self.__discard_loading = False
            data_point = error = None
            try:
                data_point = self.__load(index)
            except BaseException as ex:
                error = ex
            with self.__cond:
                if self.__open and not self.__discard_loading:
                    self.__output_queue.append((index, data_point, error))
                self.__loading_index = None
                self.__discard_loading = False
                self.__cond.notify_all()

    def __load(self, index: int) -> DataPointType:
        data_point = self.__dataset[index]
        if self.__load_fn is not None:
            self.__load_fn(data_point)
        return data_point

    def __normalize_index(self, index: int) -> int:
        length = len(self.__dataset)
        if not -length <= index < length:
            raise IndexError(
                f"Index {index} is out of bounds for data set of size {length}."
            )
        return index + length if index < 0 else index

    def prefetch(self, indices: Iterable[int]) -> None:
        """Mark the given indices for prefetching, in the order in which they are going to be collected."""
        with self.__cond:
            if not self.__open:
                raise RuntimeError(
                    "prefetch() can only be called while the PrefetchedDataset is open."
                )
            for index in indices:
                self.__input_queue.append(self.__normalize_index(int(index)))
            self.__cond.notify_all()

    def __pop_output(self) -> DataPointType:
        _, data_point, error = self.__output_queue.popleft()
        self.__cond.notify_all()
        if error is not None:
            raise error
        return data_point

    def __getitem__(self, index: int) -> DataPointType:
        index = self.__normalize_index(int(index))
        with self.__cond:
            if self.__open:
                if any(index == i for i, _, _ in self.__output_queue):
                    # Elements preceding the requested one are out of order and thus discarded
                    while self.__output_queue[0][0] != index:
                        self.__output_queue.popleft()
                    return self.__pop_output()
                loading_valid = (
                    self.__loading_index == index and not self.__discard_loading
                )
                if loading_valid or index in self.__input_queue:
                    # Everything in the output queue precedes the requested element, and so does everything in the
                    # input queue before it
                    self.__output_queue.clear()
                    if not loading_valid:
                        if self.__loading_index is not None:
                            self.__discard_loading = True
                        while self.__input_queue[0] != index:
                            self.__input_queue.popleft()
                    self.__cond.notify_all()
                    self.__cond.wait_for(
                        lambda: not self.__open
                        or (
                            len(self.__output_queue) > 0
                            and self.__output_queue[0][0] == index
                        )
                    )
                    if not self.__open:
                        raise RuntimeError(
                            "The PrefetchedDataset was closed while waiting for a load."
                        )
                    return self.__pop_output()
                # The requested element was never marked for prefetching, so all scheduled loads are considered
                # stale and it is loaded synchronously instead
                self.__output_queue.clear()
                self.__input_queue.clear()
                if self.__loading_index is not None:
                    self.__discard_loading = True
                self.__cond.notify_all()
        return self.__dataset[index]

    def __len__(self) -> int:
        return len(self.__dataset)

    @property
    def dataset(self) -> Dataset[DataPointType, Any]:
        return self.__dataset

    @property
    def capacity(self) -> int:
        return self.__capacity

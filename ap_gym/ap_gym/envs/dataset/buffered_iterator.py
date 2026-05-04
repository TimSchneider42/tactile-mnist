from __future__ import annotations

import weakref
from queue import Queue, Full
from threading import Thread, Event
from typing import Iterator, Generic, TypeVar

InnerIteratorType = TypeVar("InnerIteratorType")


class BufferedIterator(Iterator[InnerIteratorType], Generic[InnerIteratorType]):
    def __init__(self, iterator: Iterator[InnerIteratorType], buffer_size: int = 1):
        self.__iterator = iterator
        self.__buffer = Queue(maxsize=buffer_size)
        self.__termination_signal = Event()
        self.__thread = Thread(
            target=self.__thread_func,
            args=(self.__iterator, self.__buffer, self.__termination_signal),
            daemon=True,
        )
        weakref.finalize(
            self, self._thread_shutdown, self.__thread, self.__termination_signal
        )
        self.__thread.start()

    def __next__(self) -> InnerIteratorType:
        res = self.__buffer.get()
        if isinstance(res, Exception):
            raise res
        return res

    def close(self):
        self._thread_shutdown(self.__thread, self.__termination_signal)
        self.__thread = None
        # Thread safety is not an issue here, since the thread is already terminated
        self.__buffer.queue.clear()

    @staticmethod
    def _thread_shutdown(thread: Thread | None, termination_signal: Event):
        if thread is not None:
            termination_signal.set()
            thread.join()

    @staticmethod
    def __thread_func(
        iterator: Iterator[InnerIteratorType],
        buffer: Queue[InnerIteratorType],
        termination_signal: Event,
    ):
        try:
            for item in iterator:
                while not termination_signal.is_set():
                    try:
                        buffer.put(item, timeout=0.05)
                        break
                    except Full:
                        continue
                else:
                    break
        except Exception as e:
            buffer.put(e)

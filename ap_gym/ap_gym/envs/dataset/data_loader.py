from __future__ import annotations

from typing import Iterator, Generic, TypeVar

from .buffered_iterator import BufferedIterator

InnerIteratorType = TypeVar("InnerIteratorType")


class DataLoader(
    Iterator[InnerIteratorType],
    Generic[InnerIteratorType],
):
    def __init__(
        self,
        iterator: Iterator[InnerIteratorType],
        prefetch: bool = True,
        prefetch_buffer_size: int = 1,
    ):
        self.__iterator = iterator
        self.__prefetch = prefetch

        if self.__prefetch:
            self.__iterator = BufferedIterator(
                self.__iterator, buffer_size=prefetch_buffer_size
            )

    def __next__(self):
        return next(self.__iterator)

    def close(self):
        if isinstance(self.__iterator, BufferedIterator):
            self.__iterator.close()

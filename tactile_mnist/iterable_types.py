from __future__ import annotations

from abc import abstractmethod, ABC
from typing import TypeVar, Iterator, Iterable, Generic, Protocol

T = TypeVar("T")


class SizedIterable(Iterable[T], Protocol[T]):
    def __len__(self) -> int:
        ...


class SeekableIterable(Iterable[T], Generic[T], ABC):
    @abstractmethod
    def _iter(self, start: int) -> Iterable[T]:
        pass

    def iter(self, start: int = 0) -> Iterable[T]:
        return self._iter(start)

    def seek(self, offset: int = 0) -> Iterable[T]:
        return self.iter(offset)

    def __iter__(self) -> Iterator[T]:
        return iter(self.iter(start=0))


class ShiftedSizedIterable(SizedIterable[T], Generic[T], ABC):
    def __init__(self, inner_iterator: SeekableSizedIterable[T], offset: int):
        self._inner_iterator = inner_iterator
        self._offset = offset

    def __iter__(self) -> Iterator[T]:
        return iter(self._inner_iterator.iter(start=self._offset))

    def __len__(self) -> int:
        return len(self._inner_iterator) - self._offset


class SeekableSizedIterable(SeekableIterable[T], SizedIterable[T], Generic[T], ABC):
    def seek(self, offset: int = 0) -> ShiftedSizedIterable[T]:
        if offset not in range(len(self)):
            raise IndexError(f"Index {offset} out of range [0, {len(self)}).")
        return ShiftedSizedIterable(self, offset)

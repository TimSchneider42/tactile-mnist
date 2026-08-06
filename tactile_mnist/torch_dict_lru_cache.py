"""
Device-side LRU cache over the rows of a dict of host-side numpy arrays.

:class:`TorchDictLRUCache` is attached to a dict of numpy arrays sharing their
leading axis length and acts as a ``Sequence[dict[str, torch.Tensor]]``: indexing it
(or calling :meth:`TorchDictLRUCache.get`) with a single index returns the arrays'
rows at that index as torch tensors on the cache's device, and indexing it with
multiple indices returns the corresponding rows concatenated along a new leading
axis (equivalent to ``data[key][indices]``). Rows are held in preallocated device
buffers of a fixed row capacity; rows that miss are staged from the host and
overwrite the least recently used resident rows in place (rows requested in the same
lookup are pinned and never evicted by it). Returned tensors are copies, so they
remain valid after their rows are evicted. The cache is not thread-safe.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence

import numpy as np
import torch


class TorchDictLRUCache(Sequence["dict[str, torch.Tensor]"]):
    def __init__(
        self,
        data: "dict[str, np.ndarray]",
        capacity: int,
        device: "torch.device | str" = "cpu",
    ):
        """
        :param data: host-side numpy arrays sharing their leading (row) axis length.
        :param capacity: maximum number of rows held on the device (clamped to the
            number of rows in ``data``).
        :param device: device the cache buffers live on and results are returned on.
        """
        if not data:
            raise ValueError("data must contain at least one array.")
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}.")
        num_rows = None
        for name, array in data.items():
            if not isinstance(array, np.ndarray) or array.ndim < 1:
                raise ValueError(
                    f"data[{name!r}] must be a numpy array with a leading row axis."
                )
            if num_rows is None:
                num_rows = array.shape[0]
            elif array.shape[0] != num_rows:
                raise ValueError(
                    f"All arrays must share the leading axis length, got "
                    f"{ {n: a.shape[0] for n, a in data.items()} }."
                )
        self.__data = dict(data)
        self.__num_rows = num_rows
        self.__capacity = min(int(capacity), num_rows)
        self.__device = torch.device(device)
        self.__storage = {
            name: torch.empty(
                (self.__capacity,) + array.shape[1:],
                dtype=torch.from_numpy(array[:0]).dtype,
                device=self.__device,
            )
            for name, array in data.items()
        }
        self.__slot_of: OrderedDict[int, int] = OrderedDict()
        self.__free_slots = list(range(self.__capacity))
        self.__hits = 0
        self.__misses = 0

    @property
    def capacity(self) -> int:
        return self.__capacity

    @property
    def device(self) -> "torch.device":
        return self.__device

    @property
    def hits(self) -> int:
        """Requested rows served from the cache so far (duplicates included)."""
        return self.__hits

    @property
    def misses(self) -> int:
        """Requested rows staged from the host so far."""
        return self.__misses

    def __len__(self) -> int:
        return self.__num_rows

    def __getitem__(self, index) -> "dict[str, torch.Tensor]":
        return self.get(index)

    def get(self, index) -> "dict[str, torch.Tensor]":
        """
        The rows of the attached arrays at ``index`` as torch tensors on the cache's
        device. ``index`` may be a single (negative-allowed) integer, in which case
        the tensors are the plain rows, or a flat sequence/array/tensor of integers
        or a slice, in which case the rows are concatenated along a new leading axis
        (equivalent to ``data[key][index]``). A single lookup must not request more
        distinct rows than the cache capacity.
        """
        if isinstance(index, torch.Tensor):
            index = index.cpu().numpy()
        if isinstance(index, slice):
            index = np.arange(self.__num_rows)[index]
        indices = np.asarray(index)
        if indices.size == 0:
            indices = indices.astype(np.int64)
        if indices.dtype == object or not np.issubdtype(indices.dtype, np.integer):
            raise TypeError(f"Indices must be integers, got dtype {indices.dtype}.")
        if indices.ndim > 1:
            raise ValueError(
                f"Indices must be a scalar or a flat sequence, got shape "
                f"{indices.shape}."
            )
        single = indices.ndim == 0
        flat = np.atleast_1d(indices).astype(np.int64)
        flat = np.where(flat < 0, flat + self.__num_rows, flat)
        if flat.size and (int(flat.min()) < 0 or int(flat.max()) >= self.__num_rows):
            raise IndexError(
                f"Indices must lie in [{-self.__num_rows}, {self.__num_rows}), got "
                f"values in [{int(np.min(indices))}, {int(np.max(indices))}]."
            )
        slots = torch.from_numpy(self.__assign_slots(flat)).to(self.__device)
        result = {name: storage[slots] for name, storage in self.__storage.items()}
        if single:
            result = {name: value[0] for name, value in result.items()}
        return result

    def __assign_slots(self, indices: np.ndarray) -> np.ndarray:
        """
        Cache slots serving the given row indices, updating recency, assigning
        (LRU-evicted) slots to missed rows, and staging the missed rows into the
        device buffers. Rows requested in this batch are pinned, so the returned
        assignment is stable for the entire lookup.
        """
        pinned = set(map(int, indices))
        if len(pinned) > self.__capacity:
            raise ValueError(
                f"A single lookup requests {len(pinned)} distinct rows, which "
                f"exceeds the cache capacity of {self.__capacity}."
            )
        slots = np.empty(indices.shape[0], np.int64)
        miss_rows: list[int] = []
        miss_slots: list[int] = []
        for position, index in enumerate(map(int, indices)):
            slot = self.__slot_of.get(index)
            if slot is None:
                if self.__free_slots:
                    slot = self.__free_slots.pop()
                else:
                    victim = next(row for row in self.__slot_of if row not in pinned)
                    slot = self.__slot_of.pop(victim)
                self.__slot_of[index] = slot
                miss_rows.append(index)
                miss_slots.append(slot)
                self.__misses += 1
            else:
                self.__slot_of.move_to_end(index)
                self.__hits += 1
            slots[position] = slot
        if miss_rows:
            miss_slot_tensor = torch.from_numpy(np.asarray(miss_slots, np.int64)).to(
                self.__device
            )
            for name, storage in self.__storage.items():
                staged = torch.from_numpy(self.__data[name][miss_rows])
                storage.index_copy_(0, miss_slot_tensor, staged.to(self.__device))
        return slots

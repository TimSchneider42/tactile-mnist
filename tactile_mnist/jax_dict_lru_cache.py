"""
Device-side LRU cache over the rows of a dict of host-side numpy arrays, with
jit-compatible lookups.

:class:`JaxDictLRUCache` is attached to a dict of numpy arrays sharing their
leading axis length and acts as a ``Sequence[dict[str, jax.Array]]``: indexing it
(or calling :meth:`JaxDictLRUCache.get`) with a single index returns the arrays'
rows at that index as jax arrays on the cache's device, and indexing it with
multiple indices returns the corresponding rows concatenated along a new leading
axis (equivalent to ``data[key][indices]``).

All cache state lives in jax array refs (``jax.experimental.mutable_array``): the
row storage buffers, the source row held by each slot, per-slot last-access steps,
and the hit/miss counters. Under jit the refs are closed-over mutable inputs
rather than inlined constants, so lookups trace once and afterwards read and
update the live cache. A lookup compares every requested index against every
resident slot key on the device (an ``n x capacity`` comparison, so the capacity
should stay moderate) and refreshes the access step of the slots it hits. Misses
are staged by a ``jax.pure_callback`` that receives the requested indices, their
hit slots, and the slot keys/access steps, assigns the least recently used slots
to the missed rows (slots serving this lookup are pinned), and returns the
assignment plus the missed rows gathered from the host arrays. Callback outputs
have static shapes, so there is one callback branch per staging bucket size (the
request size, halved repeatedly down to 16, plus a callback-free all-hit branch)
and a ``lax.switch`` picks the smallest bucket fitting the miss count on the
device; the host transfer is thus less than twice the miss count, and zero for
fully hit lookups. Callback operands are host-side copies, so the callback cannot
write the device buffers itself; instead the branches return the staged rows,
padded on the device to the request size, and the storage refs are scattered into
in place *outside* the switch (unused staging entries target a scratch row). The
writes must stay outside it: jax refuses to partial-evaluate a cond whose branches
carry state effects ("State effect not supported in cond partial-eval"), which
would make every lookup — and hence anything computed from one, such as the
COD-VAE reconstruction loss — non-differentiable.

Returned arrays are gathered copies, so they remain valid after their rows are
evicted. Buffer dtypes and the access-step longs follow jax's dtype
canonicalization (e.g. float64 rows are stored as float32 unless x64 is enabled).
Rows fetched by the same lookup share an access step, so LRU ties between them
are broken arbitrarily. A single lookup must not request more distinct rows than
the cache capacity. The cache is not thread-safe.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence

import numpy as np

import jax
import jax.numpy as jnp

try:
    from jax import new_ref as _new_ref
except ImportError:
    try:
        from jax.experimental import mutable_array as _new_ref
    except ImportError:
        # jax < 0.6.1 has no public export of mutable_array
        from jax._src.core import mutable_array as _new_ref


class JaxDictLRUCache(Sequence["dict[str, jax.Array]"]):
    def __init__(
        self,
        data: "dict[str, np.ndarray]",
        capacity: int,
        device: "jax.Device | str | None" = None,
    ):
        """
        :param data: host-side numpy arrays sharing their leading (row) axis length.
        :param capacity: maximum number of rows held on the device (clamped to the
            number of rows in ``data``).
        :param device: device the cache buffers live on and results are returned on
            (a :class:`jax.Device` or a platform string like ``"cpu"`` or
            ``"gpu:0"``; jax's default device if omitted).
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
        self.__device = self.__resolve_device(device)
        self.__elem_shapes = {name: array.shape[1:] for name, array in data.items()}
        self.__dtypes = {
            name: jax.dtypes.canonicalize_dtype(array.dtype)
            for name, array in data.items()
        }
        self.__step_dtype = jax.dtypes.canonicalize_dtype(np.int64)

        def make_ref(value):
            return _new_ref(jax.device_put(value, self.__device))

        # Row `capacity` of each storage buffer is a scratch row absorbing the
        # staged-buffer entries of positions that hit.
        self.__storage = {
            name: make_ref(
                jnp.empty(
                    (self.__capacity + 1,) + self.__elem_shapes[name],
                    self.__dtypes[name],
                )
            )
            for name in data
        }
        self.__slot_keys = make_ref(jnp.full((self.__capacity,), -1, jnp.int32))
        self.__slot_steps = make_ref(jnp.zeros((self.__capacity,), self.__step_dtype))
        self.__step = make_ref(jnp.zeros((), self.__step_dtype))
        self.__hit_count = make_ref(jnp.zeros((), self.__step_dtype))
        self.__miss_count = make_ref(jnp.zeros((), self.__step_dtype))

    @staticmethod
    def __resolve_device(device) -> "jax.Device":
        if device is None:
            return jax.devices()[0]
        if isinstance(device, jax.Device):
            return device
        platform, _, index = str(device).partition(":")
        devices = jax.devices("gpu" if platform == "cuda" else platform)
        return devices[int(index)] if index else devices[0]

    @property
    def capacity(self) -> int:
        return self.__capacity

    @property
    def device(self) -> "jax.Device":
        return self.__device

    @property
    def hits(self) -> int:
        """Requested rows served from the cache so far (duplicates included)."""
        return int(self.__hit_count[...])

    @property
    def misses(self) -> int:
        """Requested rows staged from the host so far."""
        return int(self.__miss_count[...])

    def __len__(self) -> int:
        return self.__num_rows

    def __getitem__(self, index) -> "dict[str, jax.Array]":
        return self.get(index)

    def get(self, index) -> "dict[str, jax.Array]":
        """
        The rows of the attached arrays at ``index`` as jax arrays on the cache's
        device. ``index`` may be a single (negative-allowed) integer, in which case
        the arrays are the plain rows, or a flat sequence/array of integers or a
        slice, in which case the rows are concatenated along a new leading axis
        (equivalent to ``data[key][index]``). Jit-traceable; under jit,
        out-of-range indices are only detected at run time (by the miss callback).
        """
        if isinstance(index, slice):
            index = np.arange(self.__num_rows)[index]
        indices = jnp.asarray(index)
        if indices.size == 0:
            indices = indices.astype(jnp.int32)
        if not jnp.issubdtype(indices.dtype, jnp.integer):
            raise TypeError(f"Indices must be integers, got dtype {indices.dtype}.")
        if indices.ndim > 1:
            raise ValueError(
                f"Indices must be a scalar or a flat sequence, got shape "
                f"{indices.shape}."
            )
        single = indices.ndim == 0
        flat = jnp.atleast_1d(indices)
        try:
            concrete = np.asarray(flat)
        except jax.errors.TracerArrayConversionError:
            concrete = None
        if concrete is not None and concrete.size:
            if int(concrete.min()) < -self.__num_rows or (
                int(concrete.max()) >= self.__num_rows
            ):
                raise IndexError(
                    f"Indices must lie in [{-self.__num_rows}, {self.__num_rows}), "
                    f"got values in "
                    f"[{int(concrete.min())}, {int(concrete.max())}]."
                )
            distinct = np.unique(concrete % self.__num_rows).size
            if distinct > self.__capacity:
                raise ValueError(
                    f"A single lookup requests {distinct} distinct rows, which "
                    f"exceeds the cache capacity of {self.__capacity}."
                )
        with jax.default_device(self.__device):
            flat = jnp.where(flat < 0, flat + self.__num_rows, flat).astype(jnp.int32)
            result = self.__lookup(flat)
        if single:
            result = {name: value[0] for name, value in result.items()}
        return result

    def __lookup(self, flat: "jax.Array") -> "dict[str, jax.Array]":
        n = flat.shape[0]
        slot_keys = self.__slot_keys[...]
        slot_steps = self.__slot_steps[...]
        match = flat[:, None] == slot_keys[None, :]
        hit_slots = jnp.where(
            match.any(axis=1), match.argmax(axis=1).astype(jnp.int32), -1
        )
        num_missing = jnp.sum(hit_slots < 0)

        # Staging bucket sizes 0 (all hit), ..., n/4, n/2, n; a lookup uses the
        # smallest bucket fitting its miss count, so the host transfer is less than
        # twice the number of misses instead of always request-sized. Buckets below
        # 16 rows are not worth their branches; n itself must always be a bucket,
        # as searchsorted+switch would otherwise silently clamp to a too-small one.
        sizes = {0, n} if n else {0}
        size = n // 2
        while size >= 16:
            sizes.add(size)
            size //= 2
        sizes = sorted(sizes)

        def on_hit():
            return (
                hit_slots,
                jnp.full((n,), self.__capacity, jnp.int32),
                slot_keys,
                jnp.full((), n, self.__step_dtype),
                jnp.zeros((), self.__step_dtype),
                *(
                    jnp.zeros((n,) + self.__elem_shapes[name], self.__dtypes[name])
                    for name in self.__storage
                ),
            )

        def make_miss_branch(size: int):
            result_shapes = (
                jax.ShapeDtypeStruct((n,), jnp.int32),  # serving slots
                jax.ShapeDtypeStruct((size,), jnp.int32),  # staging scatter targets
                jax.ShapeDtypeStruct((self.__capacity,), jnp.int32),  # new slot keys
                jax.ShapeDtypeStruct((), self.__step_dtype),  # hits
                jax.ShapeDtypeStruct((), self.__step_dtype),  # misses
            ) + tuple(
                jax.ShapeDtypeStruct((size,) + self.__elem_shapes[name], dtype)
                for name, dtype in self.__dtypes.items()
            )

            def branch():
                out = jax.pure_callback(
                    functools.partial(self.__serve_missing, size),
                    result_shapes,
                    flat,
                    hit_slots,
                    slot_keys,
                    slot_steps,
                )
                slots, scatter_slots, new_slot_keys, hits, misses = out[:5]
                # Only the branch's own bucket size crosses the host boundary; the
                # padding to the request size happens on the device, so all branches
                # agree on their output shapes without inflating the transfer.
                pad = n - size
                scatter_slots = jnp.pad(
                    scatter_slots, (0, pad), constant_values=self.__capacity
                )
                rows = tuple(
                    jnp.pad(row, [(0, pad)] + [(0, 0)] * (row.ndim - 1))
                    for row in out[5:]
                )
                return (slots, scatter_slots, new_slot_keys, hits, misses, *rows)

            return branch

        branches = [
            on_hit if size == 0 else make_miss_branch(size) for size in sizes
        ]
        bucket = jnp.searchsorted(
            jnp.asarray(sizes, jnp.int32), num_missing.astype(jnp.int32)
        )
        slots, scatter_slots, new_slot_keys, hits, misses, *staged = jax.lax.switch(
            bucket, branches
        )
        # The storage updates stay outside the switch: jax cannot partial-evaluate a
        # cond whose branches carry state effects, so a lookup with ref writes inside
        # the branches cannot be differentiated (see the module docstring).
        self.__slot_keys[...] = new_slot_keys
        for storage, rows in zip(self.__storage.values(), staged):
            storage[scatter_slots] = rows
        step = self.__step[...] + 1
        self.__step[...] = step
        self.__hit_count[...] = self.__hit_count[...] + hits
        self.__miss_count[...] = self.__miss_count[...] + misses
        self.__slot_steps[slots] = jnp.full((n,), step, self.__step_dtype)
        return {name: storage[slots] for name, storage in self.__storage.items()}

    def __serve_missing(self, size, flat, hit_slots, slot_keys, slot_steps):
        """
        Host side of a lookup with misses: assigns free/LRU slots to the missed
        rows (slots serving this lookup are pinned) and gathers the missed rows
        from the source arrays into staging buffers of ``size`` rows (the bucket
        picked by the caller, at least the number of misses). Returns the serving
        slot per position, the staged rows' scatter targets (the scratch row for
        unused staging entries), the updated slot keys, the hit and miss counts,
        and the staging buffers.
        """
        flat = np.asarray(flat)
        slots = np.array(hit_slots, np.int32)
        slot_keys = np.array(slot_keys, np.int32)
        slot_steps = np.asarray(slot_steps)
        if flat.size and (int(flat.min()) < 0 or int(flat.max()) >= self.__num_rows):
            raise IndexError(
                f"Indices must lie in [{-self.__num_rows}, {self.__num_rows}), got "
                f"out-of-range values."
            )
        distinct = len(set(flat.tolist()))
        if distinct > self.__capacity:
            raise ValueError(
                f"A single lookup requests {distinct} distinct rows, which "
                f"exceeds the cache capacity of {self.__capacity}."
            )
        scatter_slots = np.full(size, self.__capacity, np.int32)
        pinned = {int(slot) for slot in slots if slot >= 0}
        eviction_order = sorted(
            (slot for slot in range(self.__capacity) if slot not in pinned),
            key=lambda slot: (slot_keys[slot] != -1, slot_steps[slot], slot),
        )
        victims = iter(eviction_order)
        assigned: dict[int, int] = {}
        hits = misses = 0
        miss_rows: list[int] = []
        for position, (row, slot) in enumerate(zip(flat.tolist(), slots.tolist())):
            if slot >= 0:
                hits += 1
                continue
            previous = assigned.get(row)
            if previous is not None:
                slots[position] = previous
                hits += 1
                continue
            victim = next(victims)
            assigned[row] = victim
            slots[position] = victim
            scatter_slots[len(miss_rows)] = victim
            slot_keys[victim] = row
            miss_rows.append(row)
            misses += 1
        staged = []
        for name, array in self.__data.items():
            buffer = np.empty(
                (size,) + self.__elem_shapes[name], np.dtype(self.__dtypes[name])
            )
            buffer[: len(miss_rows)] = array[miss_rows]
            staged.append(buffer)
        step_dtype = np.dtype(self.__step_dtype)
        return (
            slots,
            scatter_slots,
            slot_keys,
            np.asarray(hits, step_dtype),
            np.asarray(misses, step_dtype),
            *staged,
        )

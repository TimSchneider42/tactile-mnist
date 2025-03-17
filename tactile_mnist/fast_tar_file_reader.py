from __future__ import annotations

import hashlib
import io
import pickle
import struct
import threading
from collections import defaultdict
from pathlib import Path
from tarfile import TarFile, TarInfo
from typing import IO

import filelock

from .constants import CACHE_BASE_DIR


class ThreadSafeFileReader(io.BufferedIOBase):
    # WARNING: This class is leaking memory as the file descriptors of threads are never removed.
    def __init__(self, path: Path, in_memory: bool = False):
        if in_memory:
            with path.open("rb") as f:
                self._data = f.read()
            self._file_descriptors = defaultdict(lambda: io.BytesIO(self._data))
        else:
            self._file_descriptors = defaultdict(lambda: path.open("rb"))
            self._data = None

    def __getattribute__(self, item):
        if item in ["_file_descriptors", "_data", "close"]:
            return object.__getattribute__(self, item)
        return self._file_descriptors[threading.get_ident()].__getattribute__(item)

    def close(self):
        for fd in self._file_descriptors.values():
            fd.close()
        if self._data is not None:
            del self._data


class FastTarFileReader(TarFile):
    def __init__(self, path: Path, in_memory: bool = False, store_index: bool = True):
        """
        :param path:        Path to the tar file.
        :param in_memory:   Keep the entire tar file in memory while its open.
        :param store_index: Store an index file for faster access in the future.
        """
        fileobj = ThreadSafeFileReader(path, in_memory=in_memory)
        super().__init__(str(path), "r", fileobj)
        self.__members = None
        self.__store_index = store_index

    def __enter__(self):
        super().__enter__()
        self.__members = None
        if not self.index_file.exists():
            if self.__store_index:
                CACHE_BASE_DIR.mkdir(parents=True, exist_ok=True)
                with filelock.FileLock(self.index_lock_file):
                    if not self.index_file.exists():
                        self.__members = {m.name: m for m in self.getmembers()}
                        with self.index_tmp_file.open("wb") as f:
                            pickle.dump(self.getmembers(), f)
                        self.index_tmp_file.rename(self.index_file)
        if self.index_file.exists() and self.__members is None:
            with self.index_file.open("rb") as f:
                members = pickle.load(f)
            self.__members = {m.name: m for m in members}
        if self.__members is None:
            self.__members = {m.name: m for m in self.getmembers()}
        return self

    def close(self):
        super().close()
        self.fileobj.close()

    def extractfile(self, member: str | TarInfo) -> IO[bytes]:
        if isinstance(member, str):
            tarinfo = self.__members[member]
        else:
            tarinfo = member
        return super().extractfile(tarinfo)

    def __str__(self):
        return f"FastTarFileReader({self.name})"

    @property
    def path(self):
        return Path(self.name)

    @property
    def index_file(self):
        path = self.path.resolve()
        mtime = path.stat().st_mtime
        filename_hash = hashlib.sha256((str(path)).encode()).hexdigest()[:16]
        mtime_hash = hashlib.sha256(bytearray(struct.pack("f", mtime))).hexdigest()[:16]
        filename = f"{path.name}_{filename_hash}_{mtime_hash}"
        return CACHE_BASE_DIR / f"{filename}.index"

    @property
    def index_lock_file(self):
        return self.index_file.parent / f"{self.index_file.name}.lock"

    @property
    def index_tmp_file(self):
        return self.index_file.parent / f"{self.index_file.name}.tmp"

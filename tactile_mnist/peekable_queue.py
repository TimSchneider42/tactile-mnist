from __future__ import annotations

from queue import Empty, Queue
from time import time


class PeekableQueue(Queue):
    def __init__(self, maxsize: int | None = None):
        super().__init__(maxsize=maxsize)

    def peek(self, block=True, timeout=None):
        with self.not_empty:
            if not block:
                if not self._qsize():
                    raise Empty
            elif timeout is None:
                while not self._qsize():
                    self.not_empty.wait()
            elif timeout < 0:
                raise ValueError("'timeout' must be a non-negative number")
            else:
                endtime = time() + timeout
                while not self._qsize():
                    remaining = endtime - time()
                    if remaining <= 0.0:
                        raise Empty
                    self.not_empty.wait(remaining)
            return self.queue[0]

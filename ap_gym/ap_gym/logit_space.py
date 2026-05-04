import gymnasium as gym


class LogitSpace(gym.spaces.Box):
    @classmethod
    def from_box(cls, box: gym.spaces.Box):
        return cls(
            box.low,
            box.high,
            box.shape,
            box.dtype,
            box.np_random,
        )

    def __repr__(self) -> str:
        return (
            f"LogitSpace({self.low_repr}, {self.high_repr}, {self.shape}, {self.dtype})"
        )


@gym.vector.utils.batch_space.register(LogitSpace)
def _batch_space_logit_space(space: LogitSpace, n: int = 1):
    return LogitSpace.from_box(gym.vector.utils.space_utils._batch_space_box(space, n))

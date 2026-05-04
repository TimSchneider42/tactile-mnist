from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, Literal

import gymnasium as gym
import numpy as np

from .loss_fn import ZeroLossFn, LossFn
from .types import (
    ObsType,
    ActType,
    PredType,
    PredTargetType,
    WrapperObsType,
    WrapperActType,
    WrapperPredType,
    WrapperPredTargetType,
    FullActType,
)


class NoActivePerceptionEnvError(ValueError):
    pass


class ActivePerceptionActionSpace(gym.spaces.Dict, Generic[ActType, PredType]):
    def __init__(
        self,
        inner_action_space: gym.Space[ActType],
        prediction_space: gym.Space[PredType],
        seed: dict | int | np.random.Generator | None = None,
    ):
        super().__init__(
            {"action": inner_action_space, "prediction": prediction_space}, seed=seed
        )

    @property
    def inner_action_space(self) -> gym.Space[ActType]:
        return self["action"]

    @property
    def prediction_space(self) -> gym.Space[PredType]:
        return self["prediction"]

    @property
    def as_dict(self):
        return gym.spaces.Dict(
            {"action": self.inner_action_space, "prediction": self.prediction_space},
            seed=self._np_random,
        )

    @staticmethod
    def from_dict(
        d: gym.spaces.Dict[Literal["action", "prediction"], Any],
    ) -> ActivePerceptionActionSpace:
        return ActivePerceptionActionSpace(
            d["action"], d["prediction"], seed=d._np_random
        )


@gym.vector.utils.batch_space.register(ActivePerceptionActionSpace)
def _batch_space_active_perception_action_space(
    space: ActivePerceptionActionSpace, n: int = 1
):
    return ActivePerceptionActionSpace.from_dict(
        gym.vector.utils.batch_space(space.as_dict, n)
    )


class BaseActivePerceptionEnv(
    gym.Env[ObsType, FullActType[ActType, PredType]],
    Generic[ObsType, ActType, PredType, PredTargetType],
    ABC,
):
    # Set these in every subclass
    prediction_target_space: gym.Space[PredTargetType]
    action_space: ActivePerceptionActionSpace[ActType, PredType]
    loss_fn: LossFn[PredType, PredTargetType]

    @property
    def prediction_space(self) -> gym.Space[PredType]:
        return self.action_space["prediction"]

    @property
    def inner_action_space(self) -> gym.Space[ActType]:
        return self.action_space["action"]


class ActivePerceptionEnv(
    BaseActivePerceptionEnv[ObsType, ActType, PredType, PredTargetType],
    Generic[ObsType, ActType, PredType, PredTargetType],
    ABC,
):
    @abstractmethod
    def _step(
        self, action: ActType, prediction: PredType
    ) -> tuple[ObsType, float, bool, bool, dict[str, Any], PredTargetType]:
        pass

    def step(
        self, action: FullActType[ActType, PredType]
    ) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        obs, base_reward, terminated, truncated, info, prediction_target = self._step(
            action["action"], action["prediction"]
        )

        batch_shape = (self.num_envs,) if isinstance(self, gym.vector.VectorEnv) else ()
        prediction_loss = self.loss_fn(
            action["prediction"], prediction_target, batch_shape
        )

        info = {
            "base_reward": base_reward,
            "prediction": {
                "target": prediction_target,
                "loss": prediction_loss,
            },
        }

        return obs, base_reward - prediction_loss, terminated, truncated, info


class ActivePerceptionWrapper(
    gym.Wrapper[
        WrapperObsType,
        FullActType[WrapperActType, WrapperPredType],
        ObsType,
        FullActType[ActType, PredType],
    ],
    BaseActivePerceptionEnv[
        WrapperObsType, WrapperActType, WrapperPredType, WrapperPredTargetType
    ],
    Generic[
        WrapperObsType,
        WrapperActType,
        WrapperPredType,
        WrapperPredTargetType,
        ObsType,
        ActType,
        PredType,
        PredTargetType,
    ],
):
    env: BaseActivePerceptionEnv[ObsType, ActType, PredType, PredTargetType]

    def __init__(self, env: gym.Env[ObsType, FullActType[ActType, PredType]]):
        env = ensure_active_perception_env(env)
        self._action_space: ActivePerceptionActionSpace[ActType, PredType] | None
        self._prediction_target_space: (
            gym.spaces.Space[WrapperPredTargetType] | None
        ) = None
        self._loss_fn: LossFn[WrapperPredType, WrapperPredTargetType] | None = None
        super().__init__(env)

    @property
    def loss_fn(self) -> LossFn[WrapperPredType, WrapperPredTargetType]:
        if self._loss_fn is not None:
            return self._loss_fn
        return self.env.loss_fn

    @property
    def prediction_target_space(self) -> WrapperPredTargetType:
        if self._prediction_target_space is not None:
            return self._prediction_target_space
        return self.env.prediction_target_space


def find_loss_and_pred_space(
    env: gym.Env,
) -> tuple[LossFn[PredType, PredTargetType], gym.Space[PredTargetType]]:
    if isinstance(env, BaseActivePerceptionEnv):
        return env.loss_fn, env.prediction_target_space
    else:
        if isinstance(env, gym.Wrapper):
            return find_loss_and_pred_space(env.env)
        else:
            raise NoActivePerceptionEnvError(
                "The environment does not contain an ActivePerceptionEnv"
            )


class ActivePerceptionRestoreWrapper(
    gym.Wrapper[
        ObsType, FullActType[ActType, PredType], ObsType, FullActType[ActType, PredType]
    ],
    BaseActivePerceptionEnv[ObsType, ActType, PredType, PredTargetType],
    Generic[ObsType, ActType, PredType, PredTargetType],
):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.__loss_fn, self.__prediction_target_space = find_loss_and_pred_space(env)

    @property
    def loss_fn(self) -> LossFn[ActType, PredTargetType]:
        return self.__loss_fn

    @property
    def action_space(self) -> ActivePerceptionActionSpace[ActType, PredType]:
        return self.__action_space

    @property
    def prediction_target_space(self) -> PredTargetType:
        return self.__prediction_target_space

    def __getattr__(self, item):
        try:
            return super().__getattribute__(item)
        except AttributeError:
            return getattr(self.env, item)


class PseudoActivePerceptionWrapper(
    BaseActivePerceptionEnv[ObsType, ActType, tuple[()], tuple[()]],
    gym.Wrapper,
    Generic[ObsType, ActType],
):
    def __init__(self, env: gym.Env[ObsType, ActType]):
        gym.Wrapper.__init__(self, env)
        self.action_space = ActivePerceptionActionSpace(
            self.env.action_space, gym.spaces.Tuple(())
        )
        self.prediction_target_space = gym.spaces.Tuple(())
        self.loss_fn = ZeroLossFn()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any | None] = None
    ) -> tuple[ObsType, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        info["prediction"] = {
            "target": (),
        }
        return obs, info

    def step(
        self, action: FullActType[ActType, None]
    ) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action["action"])
        info.update(
            {
                "base_reward": reward,
                "prediction": {
                    "target": (),
                    "loss": np.zeros((), dtype=np.float32),
                },
            }
        )
        return obs, float(reward), terminated, truncated, info


def ensure_active_perception_env(env: gym.Env) -> BaseActivePerceptionEnv:
    if isinstance(env, BaseActivePerceptionEnv):
        return env
    try:
        return ActivePerceptionRestoreWrapper(env)
    except NoActivePerceptionEnvError:
        pass
    return PseudoActivePerceptionWrapper(env)

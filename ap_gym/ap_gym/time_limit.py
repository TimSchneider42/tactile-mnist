"""Variant of the TimeLimit wrapper with more flexibility."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import gymnasium as gym
import numpy as np

from ap_gym import idoc

if TYPE_CHECKING:
    from gymnasium.envs.registration import EnvSpec


class TimeLimit(gym.Wrapper, gym.utils.RecordConstructorArgs):
    """This wrapper will issue a `truncated` or `terminated` signal if a maximum number of timesteps is exceeded.

    Example:
       >>> import gymnasium as gym
       >>> import ap_gym
       >>> env = gym.make("CartPole-v1")
       >>> env = ap_gym.TimeLimit(env, max_episode_steps=200, issue_termination=True)

    #!AP_GYM_WRAPPER
    end_conditions:
        terminate:
        - the maximum number of steps (`max_episode_steps`) is reached.
    """

    def __init__(
        self,
        env: gym.Env,
        max_episode_steps: int,
        issue_termination: bool = False,
        observe_time_steps: bool | None = None,
    ):
        """
        Args:
            env:                The environment to apply the wrapper.
            max_episode_steps:  Maximum number of steps before truncation/termination.
            issue_termination:  If `True`, the terminate flag will be set after `max_episode_steps` steps, otherwise the
                                truncated flag will be set.
            observe_time_steps: If `True`, an additional observation will be added to the environment observation that
                                contains the number of steps elapsed, box normalized between -1 and 1. If `None`, the
                                observation will be added only if issue_termination is `True`.
        """
        gym.utils.RecordConstructorArgs.__init__(
            self,
            max_episode_steps=max_episode_steps,
            issue_termination=issue_termination,
            observe_time_steps=observe_time_steps,
        )
        gym.Wrapper.__init__(self, env)

        self._cached_env_spec: EnvSpec | None = None
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None
        self._issue_termination = issue_termination
        if observe_time_steps is None:
            observe_time_steps = issue_termination
        self._observe_time_steps = observe_time_steps

        if self._observe_time_steps:
            obs_space = self.observation_space
            time_obs_space = idoc(
                gym.spaces.Box(low=-1.0, high=1.0, shape=(), dtype=np.float32),
                f"represents the normalized current time step between 0 and `max_episode_steps` (default "
                f"{max_episode_steps}).",
            )

            if isinstance(obs_space, gym.spaces.Dict):
                obs_space = copy.deepcopy(obs_space)
                obs_space.spaces["time_step"] = time_obs_space
                self._obs_func = lambda obs: {
                    **obs,
                    "time_step": self._get_time_obs(),
                }
            elif isinstance(obs_space, gym.spaces.Tuple):
                obs_space = gym.spaces.Tuple(list(obs_space.spaces) + [time_obs_space])
                self._obs_func = lambda obs: (
                    *obs,
                    self._get_time_obs(),
                )
            elif isinstance(obs_space, gym.spaces.Box) and np.issubdtype(
                obs_space.dtype, np.floating
            ):
                obs_space = gym.spaces.Box(
                    low=np.concatenate(
                        [obs_space.low, np.array([-1.0], dtype=obs_space.dtype)]
                    ),
                    high=np.concatenate(
                        [obs_space.high, np.array([1.0], dtype=obs_space.dtype)]
                    ),
                    dtype=obs_space.dtype,
                )
                self._obs_func = lambda obs: np.concatenate(
                    [obs, [self._get_time_obs().astype(obs.dtype)]]
                )
            else:
                obs_space = gym.spaces.Dict(
                    {"observation": obs_space, "time_step": time_obs_space}
                )
                self._obs_func = lambda obs: {
                    "observation": obs,
                    "time_step": self._get_time_obs(),
                }
            self.observation_space = obs_space
        else:
            self._obs_func = lambda obs: obs

    def _get_time_obs(self):
        return np.array(
            2.0 * self._elapsed_steps / self._max_episode_steps - 1.0, dtype=np.float32
        )

    def step(self, action):
        """Steps through the environment and if the number of steps elapsed exceeds ``max_episode_steps`` then
        truncate/terminate.

        Args:
            action: The environment step action

        Returns:
            The environment step ``(observation, reward, terminated, truncated, info)`` with truncated or terminated set
            if the number of steps elapsed >= max episode steps

        """
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._elapsed_steps += 1

        if self._elapsed_steps >= self._max_episode_steps:
            if self._issue_termination:
                terminated = True
            else:
                truncated = True

        return self._obs_func(observation), reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Resets the environment with :param:`**kwargs` and sets the number of steps elapsed to zero.

        Args:
            **kwargs: The kwargs to reset the environment with

        Returns:
            The reset environment
        """
        self._elapsed_steps = 0
        observation, info = self.env.reset(**kwargs)
        return self._obs_func(observation), info

    @property
    def spec(self) -> EnvSpec | None:
        """Modifies the environment spec to include the `max_episode_steps=self._max_episode_steps`."""
        if self._cached_spec is not None:
            return self._cached_spec

        env_spec = self.env.spec
        if env_spec is not None:
            env_spec = copy.copy(env_spec)
            env_spec.max_episode_steps = self._max_episode_steps

        self._cached_spec = env_spec
        return env_spec

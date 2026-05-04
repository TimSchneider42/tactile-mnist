from __future__ import annotations

import copy
from typing import Any, Literal

import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import EnvSpec
from scipy.special import softmax

from ap_gym import (
    ActiveClassificationVectorEnv,
    ActivePerceptionVectorToSingleWrapper,
    idoc,
)
from .image import (
    ImagePerceptionModule,
    ImagePerceptionConfig,
)


class ImageClassificationVectorEnv(
    ActiveClassificationVectorEnv[
        dict[Literal["glimpse", "glimpse_pos", "time_step"], np.ndarray], np.ndarray
    ],
):
    r"""
    #!AP_GYM_BASE_ENV
    title: Image Classification Environments
    description: |
        In image classification environments, the agent has to classify an image by moving a small glimpse around the
        image. The glimpse is never large enough to see the entire image at once, so the agent has to move around to
        gather information.

        Consider the following example from the [CIFAR10](CIFAR10.md) environment:
        <p align="center"><img src="img/CIFAR10-v0.gif" alt="CIFAR10-v0" width="200px"/></p>

        Marked in blue is the agent's current glimpse.
        We mark the history of glimpses the agent has taken in a color scale ranging from red to green, red meaning that
        the agent predicted a probability of 0 for the correct class and green meaning that the agent predicted a
        probability of 1 for the correct class.

        All image classification environments in _ap_gym_ are instantiations of the
        `ap_gym.envs.image_classification.ImageClassificationVectorEnv` class and share the following properties:
    rewards:
    - 'A small action regularization equal to $10^{-3} \cdot{} \lVert \textit{action}\rVert$.'
    starting_state: The glimpse starts at a uniformly random position within the image.
    end_conditions:
        terminate:
        - the maximum number of steps (`image_perception_config.step_limit`) is reached.
    space_variables:
    - $K \in \mathbb{N}$ is the number of classes in the environment
    - $G \in \mathbb{N}$ is the glimpse size
    - $C \in \mathbb{N}$ is the number of image channels (1 for grayscale, 3 for RGB)
    """

    metadata: dict[str, Any] = {
        "render_modes": ["rgb_array"],
        "render_fps": 2,
        "autoreset_mode": gym.vector.AutoresetMode.NEXT_STEP,
    }

    def __init__(
        self,
        num_envs: int,
        image_perception_config: ImagePerceptionConfig,
        render_mode: Literal["rgb_array"] = "rgb_array",
    ):
        """

        :param num_envs:                Number of environments to create.
        :param image_perception_config: Configuration of the image perception environment. See the
                                        [ImagePerceptionConfig documentation](ImagePerceptionConfig.md) for details.
        :param render_mode:             Rendering mode (currently only `"rgb_array"` is supported).
        """

        self.__image_perception_module = ImagePerceptionModule(
            num_envs,
            image_perception_config,
        )
        if render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render mode: {render_mode}")
        self.__render_mode = render_mode
        super().__init__(
            num_envs,
            image_perception_config.dataset.num_classes,
            self.__image_perception_module.single_inner_action_space,
        )
        self.single_observation_space = gym.spaces.Dict(
            self.__image_perception_module.observation_space_dict
        )
        self.observation_space = gym.vector.utils.batch_space(
            self.single_observation_space, self.num_envs
        )
        self.__np_random = None
        self.__spec: EnvSpec | None = None
        idoc(
            self.single_prediction_space,
            {
                "text": "contains the logits of the agent's prediction w.r.t. the class label.",
                "var": {0: "K"},
            },
        )
        idoc(
            self.single_prediction_target_space,
            {"text": "represents the true class.", "var": "K"},
        )

    def reset(self, *, seed: int | None = None, options: dict[str, Any | None] = None):
        super().reset(seed=seed, options=options)
        return self.__image_perception_module.reset()

    def _step(self, action: np.ndarray, prediction: np.ndarray):
        prediction_quality = softmax(prediction, axis=-1)[
            np.arange(self.num_envs), self.__image_perception_module.current_labels
        ]
        obs, base_reward, terminated_arr, truncated_arr, info = (
            self.__image_perception_module.step(action, prediction_quality)
        )
        return (
            obs,
            base_reward,
            terminated_arr,
            truncated_arr,
            info,
            self.__image_perception_module.current_labels,
        )

    def render(self) -> np.ndarray | None:
        return self.__image_perception_module.render()

    def close(self):
        self.__image_perception_module.close()
        super().close()

    @property
    def render_mode(self) -> Literal["rgb_array"]:
        return self.__render_mode

    @property
    def _np_random(self):
        return self.__np_random

    @_np_random.setter
    def _np_random(self, np_random):
        self.__image_perception_module.seed(
            np_random.integers(0, 2**32 - 1, endpoint=True)
        )
        self.__np_random = np_random

    @property
    def spec(self) -> EnvSpec | None:
        return self.__spec

    @spec.setter
    def spec(self, spec: EnvSpec):
        spec = copy.copy(spec)
        spec.max_episode_steps = self.__image_perception_module.config.step_limit
        self.__spec = spec

    @property
    def config(self):
        return self.__image_perception_module.config

    @property
    def image_perception_module(self) -> ImagePerceptionModule:
        return self.__image_perception_module


def ImageClassificationEnv(
    image_perception_config: ImagePerceptionConfig,
    render_mode: Literal["rgb_array"] = "rgb_array",
):
    return ActivePerceptionVectorToSingleWrapper(
        ImageClassificationVectorEnv(
            1,
            image_perception_config,
            render_mode=render_mode,
        )
    )

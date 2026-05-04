from __future__ import annotations

import copy
from typing import Any, Literal

import gymnasium as gym
import numpy as np
from PIL import ImageDraw
from gymnasium.envs.registration import EnvSpec

from ap_gym import (
    ActivePerceptionVectorToSingleWrapper,
    ActiveRegressionVectorEnv,
    ImageSpace,
    idoc,
)
from .image import (
    ImagePerceptionModule,
    ImagePerceptionConfig,
)
from .style import COLOR_PRED


class ImageLocalizationVectorEnv(
    ActiveRegressionVectorEnv[
        dict[
            Literal["glimpse", "glimpse_pos", "time_step", "target_glimpse"], np.ndarray
        ],
        np.ndarray,
    ],
):
    r"""
    #!AP_GYM_BASE_ENV
    title: Image Localization Environments
    description: |
        In image localization environments, the agent has to localize a given part of the image by moving a small
        glimpse around. The glimpse is never large enough to see the entire image at once, so the agent has to move
        around to gather information. Unlike the [Image Classification Environments](ImageClassificationVectorEnv.md),
        this task is a regression task where the agent has to predict the coordinates of the target glimpse it is
        provided.

        Consider the following example from the [TinyImageNetLoc](TinyImageNetLoc.md) environment:
        <p align="center"><img src="img/TinyImageNetLoc-v0.gif" alt="TinyImageNetLoc-v0" width="200px"/></p>
        Marked in blue is the agent's current glimpse. The transparent purple box represents the target glimpse the
        agent has to predict the coordinates of and the opaque purple box is the agent's current prediction. We further
        mark the history of glimpses the agent has taken in a color scale ranging from red to green, red meaning that
        the prediction it took at this step was far from the target and green meaning that the prediction was close to
        the target.

        All image localization environments in _ap_gym_ are instantiations of the
        `ap_gym.envs.image_classification.ImageLocalizationVectorEnv` class and share the following properties:
    rewards:
    - 'A small action regularization equal to $10^{-3} \cdot{} \lVert \textit{action}\rVert$.'
    starting_state: The glimpse starts at a uniformly random position within the image.
    end_conditions:
        terminate:
        - the maximum number of steps (`image_perception_config.step_limit`) is reached.
    space_variables:
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
            2,
            self.__image_perception_module.single_inner_action_space,
            prediction_low=-1,
            prediction_high=1,
        )
        self.single_observation_space = gym.spaces.Dict(
            {
                **self.__image_perception_module.observation_space_dict,
                "target_glimpse": idoc(
                    ImageSpace(
                        image_perception_config.sensor_size[1],
                        image_perception_config.sensor_size[0],
                        image_perception_config.dataset[0][0].shape[-1],
                        dtype=np.float32,
                    ),
                    {
                        "text": "represents the target glimpse.",
                        "var": {0: "G", 1: "G", 2: "C"},
                    },
                ),
            }
        )
        self.observation_space = gym.vector.utils.batch_space(
            self.single_observation_space, self.num_envs
        )
        self.__current_prediction_target = None
        self.__prev_done = None
        self.__last_prediction = None
        self.__np_random = None
        self.__spec: EnvSpec | None = None
        idoc(
            self.single_prediction_space,
            "contains the coordinates of the agent's prediction w.r.t. the target glimpse.",
        )
        idoc(
            self.single_prediction_target_space,
            "contains the true coordinates of the target glimpse.",
        )

    def reset(self, *, seed: int | None = None, options: dict[str, Any | None] = None):
        super().reset(seed=seed, options=options)
        self.__last_prediction = None
        obs, info = self.__image_perception_module.reset()
        self.__current_prediction_target = (
            self.__image_perception_module.sample_unique_glimpse_positions().astype(
                np.float32
            )
        )
        self.__prev_done = np.zeros(self.num_envs, dtype=np.bool_)
        return (
            {
                **obs,
                "target_glimpse": self.__image_perception_module.get_glimpse(
                    self.__current_prediction_target
                ),
            },
            info,
        )

    def _step(self, action: np.ndarray, prediction: np.ndarray):
        prediction_target = self.__current_prediction_target.copy()
        if np.any(self.__prev_done):
            self.__current_prediction_target[self.__prev_done] = self.np_random.uniform(
                -1, 1, (np.sum(self.__prev_done), 2)
            ).astype(np.float32)
        prediction_quality = 1 - np.linalg.norm(
            prediction - self.__current_prediction_target, axis=-1
        ) / np.sqrt(4)
        self.__last_prediction = prediction
        (
            obs,
            base_reward,
            terminated_arr,
            truncated_arr,
            info,
        ) = self.__image_perception_module.step(action, prediction_quality)
        self.__prev_done = terminated_arr | truncated_arr
        return (
            {
                **obs,
                "target_glimpse": self.__image_perception_module.get_glimpse(
                    self.__current_prediction_target
                ),
            },
            base_reward,
            terminated_arr,
            truncated_arr,
            info,
            prediction_target,
        )

    def render(self) -> np.ndarray | None:
        imgs = self.__image_perception_module.render(return_pil_imgs=True)
        last_prediction = self.__last_prediction
        if last_prediction is None:
            last_prediction = [None] * self.num_envs

        glimpse_size = (
            self.__image_perception_module.effective_sensor_size
            * self.__image_perception_module.render_scaling
        )
        target_color = COLOR_PRED + (100,)

        for img, last_pred, target in zip(
            imgs, last_prediction, self.__current_prediction_target
        ):
            draw = ImageDraw.Draw(img, "RGBA")
            t_trans = self.__image_perception_module.to_render_coords(target)
            draw.rectangle(
                (tuple(t_trans - glimpse_size / 2), tuple(t_trans + glimpse_size / 2)),
                outline=target_color,
                width=self.__image_perception_module.glimpse_border_width,
            )
            if last_pred is not None:
                lp_trans = self.__image_perception_module.to_render_coords(last_pred)
                lp_coords = np.concatenate(
                    [lp_trans - glimpse_size / 2, lp_trans + glimpse_size / 2]
                )
                draw.rectangle(
                    tuple(lp_coords),
                    outline=COLOR_PRED,
                    width=self.__image_perception_module.glimpse_border_width,
                )
                draw.rectangle(
                    tuple(
                        lp_coords + self.__image_perception_module.glimpse_border_width
                    ),
                    outline=(0, 0, 0, 80),
                    width=self.__image_perception_module.glimpse_border_width,
                )

        return np.asarray(imgs)

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


def ImageLocalizationEnv(
    image_perception_config: ImagePerceptionConfig,
    render_mode: Literal["rgb_array"] = "rgb_array",
):
    return ActivePerceptionVectorToSingleWrapper(
        ImageLocalizationVectorEnv(
            1,
            image_perception_config,
            render_mode=render_mode,
        )
    )

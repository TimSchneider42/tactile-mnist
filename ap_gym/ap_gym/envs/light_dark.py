from __future__ import annotations

from collections import deque
from typing import Any, Literal

import gymnasium as gym
import numpy as np
from PIL import Image, ImageDraw

from ap_gym import ActiveRegressionEnv, idoc
from .style import COLOR_PRED, COLOR_AGENT, COLOR_OBS_PRIMARY, quality_color


class LightDarkEnv(ActiveRegressionEnv[np.ndarray, np.ndarray]):
    r"""
    #!AP_GYM_BASE_ENV
    title: LightDark
    description: |
        In the LightDark Environment, the agent must estimate its position based on noisy observations, where the noise
        level depends on the brightness of the surrounding area. The environment simulates an active regression task
        where the agent can move to areas with better visibility to improve its position estimation.

        This environment is useful for testing active regression models, where the agent must strategically explore its
        environment to obtain more reliable observations before making predictions.

        The visualization shown above has to be interpreted as follows:

        - **Blue dot**: Agent's current position.
        - **Green transparent circle**: Observation uncertainty (higher in dark regions).
        - **Green dot**: Noisy observed position.
        - **Purple dot**: Agent's last prediction.
        - **Light blue dot**: Agent's previous position (this is what the agent's prediction tries to approximate).
        - **White background**: Bright regions with low uncertainty.
        - **Dark background**: Dark regions with high uncertainty.
    rewards:
    - 'A small action regularization equal to $10^{-3} \cdot{} \lVert \textit{action}\rVert$.'
    - 'A constant reward of $0.1$ to ensure that the reward stays positive and the agent does not learn to terminate the
    episode on purpose.'
    starting_state: 'The agent''s initial position is uniformly randomly sampled from the range $[-1, 1]^2$.'
    end_conditions:
        terminate:
        - The agent moves out of bounds.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(self, render_mode: Literal["rgb_array"] = "rgb_array"):
        """

        :param render_mode: Rendering mode (currently only `"rgb_array"` is supported).
        """
        super().__init__(
            2,
            idoc(
                gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
                "describes the agent's relative movement. The value is first projected into the unit circle and then "
                "scaled by 0.15. If the agent moves outside the valid region ($[-1, 1]^2$), the episode is terminated.",
            ),
            prediction_low=-1,
            prediction_high=1,
        )
        idoc(self.prediction_target_space, "represents the true position of the agent.")
        idoc(self.prediction_space, "represents the predicted position of the agent.")

        if render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Invalid render mode: {render_mode}")
        self.__render_mode = render_mode
        self.__pos = self.__last_obs = self.__last_pos = self.__last_pred = None
        self.__light_pos = np.array([0, -0.7], dtype=np.float32)
        self.__light_height = 0.2

        res = 500
        coords_x, coords_y = np.meshgrid(
            np.linspace(-1, 1, res), np.linspace(-1, 1, res), indexing="ij"
        )
        brightness = self.__compute_brightness(np.stack([coords_y, coords_x], axis=-1))
        ambient_light = 0.1
        self.__base_image = np.broadcast_to(
            ((brightness * 0.9 + ambient_light) * 255).astype(np.uint8)[..., None],
            (res, res, 3),
        )

        self.observation_space = gym.spaces.Dict(
            {
                "noisy_position": idoc(
                    gym.spaces.Box(low=-2, high=2, shape=(2,), dtype=np.float32),
                    "contains a noisy estimate of the agent's position. The level of noise depends on the "
                    "brightness of the area the agent is in. A brighter area results in a lower noise level, while a "
                    "darker area results in a higher noise level.",
                )
            }
        )

        self.__trajectory = deque()

    def __compute_brightness(self, pos: np.ndarray) -> np.ndarray:
        dist_squared = (
            np.sum((pos - self.__light_pos) ** 2, axis=-1) + self.__light_height**2
        )
        return self.__light_height**2 / dist_squared

    def __get_obs(self):
        self.__last_obs = self.__pos + self.np_random.normal(size=2).astype(
            np.float32
        ) * self.__get_std_dev(self.__pos)
        self.__last_obs = np.clip(self.__last_obs, -2, 2)
        return {"noisy_position": self.__last_obs}

    def __get_std_dev(self, pos: np.ndarray):
        return (1 - self.__compute_brightness(pos)) * 0.3

    def reset(self, *, seed: int | None = None, options: dict[str, Any | None] = None):
        super().reset(seed=seed, options=options)
        self.__pos = self.np_random.uniform(
            -np.ones(2),
            np.ones(2),
            size=2,
        ).astype(np.float32)
        self.__trajectory.clear()
        self.__last_pred = self.__last_pos = None
        return self.__get_obs(), {}

    def _step(self, action: np.ndarray, prediction: np.ndarray):
        if np.any(np.isnan(action)):
            raise ValueError("NaN values detected in action.")
        if np.any(np.isnan(prediction)):
            raise ValueError("NaN values detected in prediction.")

        self.__last_pred = prediction
        self.__last_pos = self.__pos.copy()

        # The 0.1 is to ensure that the agent does not simply learn to terminate the episode early by moving out of
        # bounds
        base_reward = 0.1 - 1e-3 * np.sum(action**2, axis=-1)

        magnitude = np.linalg.norm(action)
        if magnitude > 1:
            action = action / magnitude

        self.__pos += action * 0.15
        terminated = False
        if np.any(np.abs(self.__pos) >= 1):
            terminated = True
        self.__pos = np.clip(self.__pos, -1, 1)

        prediction_quality = np.maximum(
            1 - np.linalg.norm(prediction - self.__last_pos) / 0.5, 0
        )
        self.__trajectory.append((self.__last_pos, prediction_quality))
        return self.__get_obs(), base_reward, terminated, False, {}, self.__last_pos

    def render(self):
        img = Image.fromarray(self.__base_image)  # Convert base image to PIL format
        draw = ImageDraw.Draw(img, mode="RGBA")

        dot_radius = 0.01 * img.size[0]

        pos = (self.__pos + 1) / 2 * np.array(img.size[::-1])

        std_radius = self.__get_std_dev(self.__pos) / 2 * np.array(img.size[::-1])
        draw.ellipse(
            [
                tuple(pos - std_radius),
                tuple(pos + std_radius),
            ],
            fill=COLOR_OBS_PRIMARY + (30,),
            outline=None,
        )

        traj_hist = list(self.__trajectory)
        for (pos_a, qual_a), (pos_b, qual_b) in zip(
            traj_hist[:-1], list(traj_hist)[1:]
        ):
            pos_a = (pos_a + 1) / 2 * np.array(img.size[::-1])
            pos_b = (pos_b + 1) / 2 * np.array(img.size[::-1])
            draw.line(
                (
                    pos_a[0],
                    pos_a[1],
                    pos_b[0],
                    pos_b[1],
                ),
                width=2,
                fill=quality_color(qual_b),
            )

        last_obs = (self.__last_obs + 1) / 2 * np.array(img.size[::-1])
        draw.line(
            (
                tuple(pos),
                tuple(last_obs),
            ),
            fill=COLOR_OBS_PRIMARY + (80,),
        )
        draw.ellipse(
            [
                tuple(last_obs - dot_radius),
                tuple(last_obs + dot_radius),
            ],
            fill=COLOR_OBS_PRIMARY + (100,),
            outline=None,
        )

        if self.__last_pred is not None:
            last_pred = (self.__last_pred + 1) / 2 * np.array(img.size[::-1])
            last_pos = (self.__last_pos + 1) / 2 * np.array(img.size[::-1])

            draw.line(
                (
                    tuple(last_pos),
                    tuple(last_pred),
                ),
                fill=COLOR_PRED + (80,),
            )

            draw.ellipse(
                [
                    tuple(last_pred - dot_radius),
                    tuple(last_pred + dot_radius),
                ],
                fill=COLOR_PRED + (100,),
                outline=None,
            )

            draw.ellipse(
                [
                    tuple(last_pos - dot_radius),
                    tuple(last_pos + dot_radius),
                ],
                fill=COLOR_AGENT + (100,),
                outline=None,
            )

        draw.ellipse(
            [
                tuple(pos - dot_radius),
                tuple(pos + dot_radius),
            ],
            fill=COLOR_AGENT,
            outline=None,
        )

        return np.array(img)

    @property
    def render_mode(self) -> Literal["rgb_array"]:
        return self.__render_mode

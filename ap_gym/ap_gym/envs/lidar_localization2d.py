from __future__ import annotations

from collections import deque
from typing import Any, Literal

import PIL.Image
import PIL.ImageDraw
import gymnasium as gym
import numpy as np
import shapely

from ap_gym import ActiveRegressionEnv, ImageSpace, idoc
from .dataset import DataLoader, DatasetIterator
from .floor_map import FloorMapDataset
from .style import (
    COLOR_AGENT,
    COLOR_PRED,
    COLOR_OBS_PRIMARY,
    COLOR_OBS_SECONDARY,
    quality_color,
)


class LIDARLocalization2DEnv(ActiveRegressionEnv[dict[str, np.ndarray], np.ndarray]):
    r"""
    #!AP_GYM_BASE_ENV
    title: 2D LIDAR Localization Environments
    description: |
        In the 2D LIDAR localization environments, the agent is placed in a random location in a 2D map and must predict
        its own position. As shown in the visualization below, the map is represented by a 2D bitmap, where passable
        pixels are white and obstacles are black. The size of the map varies between 21x21 pixels and 32x32 pixels,
        depending on the environment.

        To allow the agent to perform self-localization, it receives two types of information per step: distance
        readings from an 8-beam LIDAR sensor and odometry data. The LIDAR sensor emits beams in 8 directions, and the
        agent receives the distance to the nearest obstacle in each direction. However, the range of the LIDAR sensor is
        limited, so the agent might receive no information sometimes. The odometry data is the agent's relative movement
        from its starting position, which is exact in these environments.

        There are two types of environments in this category, with _static_ and _dynamic_ maps. In case of a static map,
        the agent is continuously placed in the same map but in a random position. Thus, it can learn the layout of the
        map over time and use this information to localize itself.

        In the dynamic map environments, the map is randomized at the beginning of each episode. The maps are
        procedurally generated, so the probability of the agent receiving the same map multiple times is very low. To
        make the task of self-localization possible, the agent is provided with the map of the environment as input.

        Furthermore, we currently provide two types of maps: _maze_ and _room_. In maze maps, the agent faces narrow
        corridors and many turns, while the room maps feature larger open spaces.

        Examples of each combination are shown here:

        <table align="center" style="border-collapse: collapse; border: none;">
            <tr style="border: none;">
                <td align="center" style="border: none; padding: 10px;">
                    &nbsp;
                </td>
                <td align="center" style="border: none; padding: 10px;">
                     <b>Static Map</b>
                </td>
                <td align="center" style="border: none; padding: 10px;">
                     <b>Dynamic Map</b>
                </td>
            </tr>
            <tr style="border: none;">
                <td align="center" style="border: none; padding: 10px;">
                    <b>Rooms</b>
                </td>
                <td align="center" style="border: none; padding: 10px;">
                    <img src="img/LIDARLocRoomsStatic-v0.gif" alt="LIDARLocRoomsStatic-v0" width="150px"/><br/>
                    <a href="LIDARLocRoomsStatic.md">
                        LIDARLocRoomsStatic-v0
                    </a>
                </td>
                <td align="center" style="border: none; padding: 10px;">
                    <img src="img/LIDARLocRooms-v0.gif" alt="LIDARLocRooms-v0" width="150px"/><br/>
                    <a href="LIDARLocRooms.md">
                        LIDARLocRooms-v0
                    </a>
                </td>
            </tr>
            <tr style="border: none;">
                <td align="center" style="border: none; padding: 10px;">
                    <b>Maze</b>
                </td>
                <td align="center" style="border: none; padding: 10px;">
                    <img src="img/LIDARLocMazeStatic-v0.gif" alt="LIDARLocMazeStatic-v0" width="150px"/><br/>
                    <a href="LIDARLocMazeStatic.md">
                        LIDARLocMazeStatic-v0
                    </a>
                </td>
                <td align="center" style="border: none; padding: 10px;">
                    <img src="img/LIDARLocMaze-v0.gif" alt="LIDARLocMaze-v0" width="150px"/><br/>
                    <a href="LIDARLocMaze.md">
                        LIDARLocMaze-v0
                    </a>
                </td>
            </tr>
        </table>

        In this visualization, the agent moves through the room while using its LIDAR sensors, represented by the green
        beams extending outward. The grayed out regions indicate areas that the agent has not yet observed and a purple
        dot shows the agent's last prediction. To illustrate the agent’s localization accuracy, we visualize its past
        predictions with a color gradient ranging from red to green along its trajectory. A red trail means that the
        agent's predicted position is far from the true position, whereas a green trail indicates a good estimate.


        All 2D LIDAR localization environments in ap_gym are implemented as instances of the
        `ap_gym.envs.lidar_localization2d.LIDARLocalization2DEnv` class and share the following properties:
    rewards:
    - 'A small action regularization equal to $10^{-3} \cdot{} \lVert \textit{action}\rVert$.'
    starting_state: The agent begins at a uniformly random, valid location within the environment.
    space_variables:
    - $M\in \mathbb{N}$ is the side length of the map in pixels
    """

    metadata = {
        "render_modes": ["rgb_array"],
        "render_fps": 4,
    }

    def __init__(
        self,
        dataset: FloorMapDataset,
        render_mode: Literal["rgb_array"] = "rgb_array",
        static_map: bool = False,
        lidar_beam_count: int = 8,
        lidar_range: float = 5,
        static_map_index: int = 0,
        prefetch: bool = True,
        prefetch_buffer_size: int = 128,
    ):
        """
        :param dataset:                 Dataset used for the environment containing maps.
        :param render_mode:             Rendering mode. Currently, only "rgb_array" is supported.
        :param static_map:              Whether the environment uses a single fixed map (`True`) or samples different
                                        maps per episode (`False`).
        :param lidar_beam_count:        Number of beams emitted by the LIDAR sensor.
        :param lidar_range:             Maximum range (distance) for LIDAR sensor readings.
        :param static_map_index:        Index of the map used when `static_map=True`. Ignored otherwise.
        :param prefetch:                Whether maps should be prefetched asynchronously for efficiency.
        :param prefetch_buffer_size:    Buffer size for prefetching maps from the dataset.
        """
        super().__init__(
            2,
            idoc(
                gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
                "describes the agent's relative movement in pixels. The value is projected into the unit circle before "
                "being added to the position",
            ),
            prediction_low=-1,
            prediction_high=1,
        )
        if render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Invalid render mode: {render_mode}")
        self.__render_mode = render_mode

        self.__static_map = static_map
        self.__dataset = dataset
        self.__prefetch = prefetch
        self.__prefetch_buffer_size = prefetch_buffer_size

        self.__map = None
        self.__map_shapely = None
        self.__map_obs = None
        self.__pos = None
        self.__last_pos = None
        self.__last_pred = None
        self.__initial_pos = None
        self.__trajectory = deque()
        self.__observation_map = None
        self.__data_loader = None
        self.__np_random = None
        self.__map_idx = None

        self.__dataset.load()
        if self.__static_map:
            self.__set_map(self.__dataset[static_map_index], static_map_index)

        self.__lidar_range = lidar_range
        lidar_angles = np.linspace(
            -np.pi, np.pi, lidar_beam_count, dtype=np.float32, endpoint=False
        )
        lidar_directions_unscaled = np.stack(
            [np.cos(lidar_angles), np.sin(lidar_angles)], axis=-1
        )
        self.__lidar_directions = lidar_directions_unscaled * self.__lidar_range
        self.__scan_points = (
            np.arange(0, self.__lidar_range, 0.05)[None, :, None]
            * lidar_directions_unscaled[:, None]
        )

        observation_dict = {
            "lidar": idoc(
                gym.spaces.Box(
                    low=0, high=1, shape=(lidar_beam_count,), dtype=np.float32
                ),
                "contains distances measured by the LIDAR sensor.",
            ),
            "odometry": idoc(
                gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
                "represents the agent's normalized relative displacement from its starting position.",
            ),
        }

        map_space = idoc(
            ImageSpace(width=dataset.map_width, height=dataset.map_height, channels=1),
            {
                "text": "contains a grayscale image representation of the environment.",
                "var": {0: "M", 1: "M"},
            },
        )

        fake_entries = {}
        if not static_map:
            observation_dict["map"] = map_space
        else:
            # This is just to simplify the auto-generated documentation
            fake_entries["map"] = map_space

        self.observation_space = idoc(
            gym.spaces.Dict(observation_dict),
            {
                "extra_entries": fake_entries,
                "extra_text": 'The `"map"` key is only included in the non-static environments.',
            },
        )

        idoc(
            self.prediction_space,
            "contains the predicted normalized agent position.",
        )
        idoc(
            self.prediction_target_space,
            "contains the actual normalized agent position.",
        )

    def __get_obs(self) -> dict[str, np.ndarray]:
        distances, contact_coords = self.__lidar_scan(
            self.__pos, self.__pos + self.__lidar_directions
        )
        self.__last_lidar_readings = distances

        valid_contact_coords = contact_coords[np.all(contact_coords >= 0, axis=-1)]
        self.__observation_map[
            valid_contact_coords[:, 1], valid_contact_coords[:, 0]
        ] = True

        # Not 100% exact but good enough for visualization
        not_occluded = (
            np.linalg.norm(self.__scan_points, axis=-1) <= distances[..., None]
        )
        scan_points_world = self.__pos[None, None] + self.__scan_points
        scan_point_coords = np.floor(scan_points_world).astype(np.int_)
        in_bounds = np.all(
            (scan_point_coords >= 0) & (scan_point_coords < self.__map.shape), axis=-1
        )
        scan_point_coords = scan_point_coords[not_occluded & in_bounds]
        self.__observation_map[scan_point_coords[..., 1], scan_point_coords[..., 0]] = (
            True
        )

        odometry = self.__pos - self.__initial_pos
        odometry_max_value = np.array(
            [self.__dataset.map_width, self.__dataset.map_height], dtype=np.float32
        )
        odometry_min_value = -odometry_max_value
        odometry_norm = (odometry - odometry_min_value) / (
            odometry_max_value - odometry_min_value
        ) * 2 - 1
        obs = {
            "lidar": np.clip(distances / self.__lidar_range, -1, 1),
            "odometry": odometry_norm,
        }
        if not self.__static_map:
            obs["map"] = self.__map_obs
        return obs

    def __set_map(self, new_map: np.ndarray, map_idx: int):
        assert new_map.shape == (self.__dataset.map_height, self.__dataset.map_width)
        self.__map = new_map
        coords = np.meshgrid(
            np.arange(self.__map.shape[1]), np.arange(self.__map.shape[0])
        )
        occupied_coords = np.stack(
            [coords[0][self.__map], coords[1][self.__map]], axis=-1
        )
        self.__map_shapely = shapely.union_all(
            [shapely.box(*c, *(c + 1)) for c in occupied_coords]
        )
        self.__map_idx = map_idx

    def reset(self, *, seed: int | None = None, options: dict[str, Any | None] = None):
        super().reset(seed=seed, options=options)
        self.__last_lidar_readings = None

        if not self.__static_map:
            self.__set_map(*next(self.__data_loader))
            self.__map_obs = self.__map[..., None].astype(np.float32) / 255

        self.__observation_map = np.zeros_like(self.__map, dtype=np.bool_)

        valid_starting_coords = np.where(self.__map == 0)
        idx = self.np_random.integers(0, len(valid_starting_coords[0]))
        self.__pos = self.__initial_pos = (
            np.array(
                [valid_starting_coords[1][idx], valid_starting_coords[0][idx]],
                dtype=np.float32,
            )
            + 0.5
        )
        self.__trajectory.clear()
        self.__last_pred = self.__last_pos = None

        return self.__get_obs(), {"map_idx": self.__map_idx}

    def _step(self, action: np.ndarray, prediction: np.ndarray):
        if np.any(np.isnan(action)):
            raise ValueError("NaN values detected in action.")
        if np.any(np.isnan(prediction)):
            raise ValueError("NaN values detected in prediction.")

        map_size = np.array(
            [self.__map.shape[1], self.__map.shape[0]], dtype=np.float32
        )
        self.__last_pred = (prediction + 1) / 2 * map_size
        self.__last_pos = self.__pos.copy()

        # The 1 is to ensure that the agent does not simply learn to terminate the episode early by moving out of bounds
        base_reward = 0.1 - 1e-3 * np.sum(action**2, axis=-1)

        magnitude = np.linalg.norm(action)
        if magnitude > 1:
            action = action / magnitude

        target_pos = self.__pos + action
        direction = target_pos - self.__pos
        total_dist = np.linalg.norm(direction)
        if total_dist > 0:
            direction /= total_dist
            dist_to_wall, _ = self.__lidar_scan(self.__pos, target_pos[None])
            dist_to_wall = dist_to_wall[0]
            self.__pos += direction * dist_to_wall

            # Make agent slide along wall
            remaining_dist = total_dist - dist_to_wall
            if remaining_dist > 1e-5:
                remaining_vec = direction * remaining_dist
                remaining_vec = remaining_vec[remaining_vec > 1e-5]
                if len(remaining_vec) > 0:
                    direction_candidates = np.eye(2, dtype=np.float32) * remaining_vec
                    target_pos_candidates = self.__pos + direction_candidates
                    dist_to_wall_candidates, _ = self.__lidar_scan(
                        self.__pos, target_pos_candidates
                    )
                    if dist_to_wall_candidates[0] > 0 or len(direction_candidates) == 1:
                        idx = 0
                    else:
                        idx = 1
                    self.__pos += (
                        direction_candidates[idx]
                        / np.linalg.norm(direction_candidates[idx])
                        * dist_to_wall_candidates[idx]
                    )

        terminated = False
        pos_min = np.zeros(2, dtype=np.float32)
        pos_max = np.array([self.__map.shape[1], self.__map.shape[0]], dtype=np.float32)
        if np.any(self.__pos < pos_min) or np.any(self.__pos >= pos_max):
            terminated = True
        self.__pos = np.clip(
            self.__pos,
            pos_min,
            pos_max,
        )

        normalized_last_pos = self.__last_pos / map_size * 2 - 1

        prediction_quality = 1 - np.linalg.norm(prediction - normalized_last_pos) / 0.25
        self.__trajectory.append((self.__last_pos, np.minimum(prediction_quality, 1)))

        return (
            self.__get_obs(),
            base_reward,
            terminated,
            False,
            {"map_idx": self.__map_idx},
            normalized_last_pos,
        )

    def render(self):
        width = 500
        scale = width / self.__map.shape[1]
        alpha = 0.25 + 0.75 * self.__observation_map.astype(np.float32)
        base_img = (
            PIL.Image.fromarray(
                (alpha * (~self.__map).astype(np.float32) + (1.0 - alpha) * 0.5) * 255
            )
            .resize(
                (
                    int(round(self.__map.shape[1] * scale)),
                    int(round(self.__map.shape[0] * scale)),
                ),
                resample=PIL.Image.NEAREST,
            )
            .convert("RGB")
        )

        draw = PIL.ImageDraw.Draw(base_img, mode="RGBA")
        contact_marker_radius = 0.2
        agent_radius = 0.2

        traj_hist = list(self.__trajectory)
        for (pos_a, qual_a), (pos_b, qual_b) in zip(
            traj_hist[:-1], list(traj_hist)[1:]
        ):
            draw.line(
                (
                    pos_a[0] * scale,
                    pos_a[1] * scale,
                    pos_b[0] * scale,
                    pos_b[1] * scale,
                ),
                width=2,
                fill=quality_color(qual_b),
            )

        if self.__last_lidar_readings is not None:
            directions_norm = self.__lidar_directions / np.linalg.norm(
                self.__lidar_directions, axis=-1, keepdims=True
            )
            for dist, direction in zip(self.__last_lidar_readings, directions_norm):
                draw.line(
                    (
                        self.__pos[0] * scale,
                        self.__pos[1] * scale,
                        (self.__pos + direction * dist)[0] * scale,
                        (self.__pos + direction * dist)[1] * scale,
                    ),
                    width=2,
                    fill=COLOR_OBS_PRIMARY,
                )
                contact_point = self.__pos + direction * dist
                draw.ellipse(
                    (
                        (contact_point[0] - contact_marker_radius) * scale,
                        (contact_point[1] - contact_marker_radius) * scale,
                        (contact_point[0] + contact_marker_radius) * scale,
                        (contact_point[1] + contact_marker_radius) * scale,
                    ),
                    fill=COLOR_OBS_SECONDARY,
                )
        if self.__last_pred is not None:
            draw.line(
                (
                    self.__last_pos[0] * scale,
                    self.__last_pos[1] * scale,
                    self.__last_pred[0] * scale,
                    self.__last_pred[1] * scale,
                ),
                fill=COLOR_PRED + (80,),
            )

            draw.ellipse(
                (
                    (self.__last_pred[0] - agent_radius) * scale,
                    (self.__last_pred[1] - agent_radius) * scale,
                    (self.__last_pred[0] + agent_radius) * scale,
                    (self.__last_pred[1] + agent_radius) * scale,
                ),
                fill=COLOR_PRED,
            )

            draw.ellipse(
                (
                    (self.__last_pos[0] - agent_radius) * scale,
                    (self.__last_pos[1] - agent_radius) * scale,
                    (self.__last_pos[0] + agent_radius) * scale,
                    (self.__last_pos[1] + agent_radius) * scale,
                ),
                fill=COLOR_AGENT + (100,),
            )

        draw.ellipse(
            (
                (self.__pos[0] - agent_radius) * scale,
                (self.__pos[1] - agent_radius) * scale,
                (self.__pos[0] + agent_radius) * scale,
                (self.__pos[1] + agent_radius) * scale,
            ),
            fill=COLOR_AGENT,
        )

        return np.array(base_img)

    def __lidar_scan(self, pos: np.ndarray, target_pos: np.ndarray, eps=1e-3):
        output_distances = np.empty(target_pos.shape[0], dtype=np.float32)
        output_contact_coords = np.empty((target_pos.shape[0], 2), dtype=np.int32)
        for i, target in enumerate(target_pos):
            line = shapely.LineString([pos, target])
            intersections = line.intersection(self.__map_shapely)
            if (
                isinstance(intersections, shapely.LineString)
                and not intersections.is_empty
            ):
                contact_point = np.array(
                    [intersections.xy[0][0], intersections.xy[1][0]]
                )
                output_distances[i] = np.maximum(
                    np.linalg.norm(contact_point - pos) - eps, 0
                )
            elif isinstance(intersections, shapely.Point):
                contact_point = pos
                output_distances[i] = 0
            elif isinstance(intersections, shapely.MultiPoint) or isinstance(
                intersections, shapely.MultiLineString
            ):
                intersection_points = np.array(
                    [[p.xy[0][0], p.xy[1][0]] for p in intersections.geoms],
                    dtype=np.float32,
                )
                distances = np.linalg.norm(intersection_points - pos, axis=-1)
                min_idx = np.argmin(distances)
                contact_point = intersection_points[min_idx]
                output_distances[i] = np.maximum(distances[min_idx] - eps, 0)
            else:
                output_distances[i] = np.linalg.norm(target - pos)
                contact_point = target
            if output_distances[i] < np.linalg.norm(target - pos):
                coords = np.floor(contact_point)
                exact = np.abs(coords - contact_point) < 1e-5
                coords[exact & (target < pos)] -= 1
                output_contact_coords[i] = coords
            else:
                output_contact_coords[i] = -1
        return output_distances, output_contact_coords

    def close(self):
        if self.__data_loader is not None:
            self.__data_loader.close()
        super().close()

    @property
    def _np_random(self):
        return self.__np_random

    @_np_random.setter
    def _np_random(self, np_random):
        if not self.__static_map:
            self.__data_loader = DataLoader(
                DatasetIterator(
                    self.__dataset, np_random.integers(0, 2**32, endpoint=True)
                ),
                self.__prefetch,
                self.__prefetch_buffer_size,
            )
        self.__np_random = np_random

    @property
    def render_mode(self) -> Literal["rgb_array"]:
        return self.__render_mode

    @property
    def dataset(self):
        return self.__dataset

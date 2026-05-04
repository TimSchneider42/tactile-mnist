# 2D LIDAR Localization Environments

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


## Properties

<table>
  <tr>
    <td><strong>Action Space</strong></td>
    <td><code>Box(-1.0, 1.0, (2,), float32)</code></td>
  </tr>
  <tr>
    <td><strong>Prediction Space</strong></td>
    <td><code>Box(-inf, inf, (2,), float32)</code></td>
  </tr>
  <tr>
    <td><strong>Prediction Target Space</strong></td>
    <td><code>Box(-inf, inf, (2,), float32)</code></td>
  </tr>
  <tr>
    <td><strong>Observation Space</strong></td>
    <td><code>Dict({</code><br/><code>&nbsp;&nbsp;"lidar"    : Box(0.0, 1.0, (8,), float32)</code><br/><code>&nbsp;&nbsp;"map"      : Box(0.0, 1.0, (M, M, 1), float32)</code><br/><code>&nbsp;&nbsp;"odometry" : Box(-1.0, 1.0, (2,), float32)</code><br/><code>&nbsp;&nbsp;"time_step": Box(-1.0, 1.0, (), float32)</code><br/><code>})
The `"map"` key is only included in the static environments.</code></td>
  </tr>
  <tr>
    <td><strong>Loss Function</strong></td>
    <td><code>ap_gym.MSELossFn()</code></td>
  </tr>
</table>


 where $M\in \mathbb{N}$ is the side length of the map in pixels.

## Action Space

The action is a `np.ndarray[float32]` $\in [-1, 1]^{2}$ that describes the agent's relative movement in pixels. The value is projected into the unit circle before being added to the position

## Prediction Space

The prediction is a `np.ndarray[float32]` $\in \mathbb{R}^{2}$ that contains the predicted normalized agent position.

## Prediction Target Space

The prediction target is a `np.ndarray[float32]` $\in \mathbb{R}^{2}$ that contains the actual normalized agent position.

## Observation Space

The observation is a dictionary with the following keys:

| Key       | Type         | Description                                                                                                                      |
|-----------|--------------|----------------------------------------------------------------------------------------------------------------------------------|
| lidar     | `np.ndarray` | `np.ndarray[float32]` $\in [0, 1]^{8}$ that contains distances measured by the LIDAR sensor.                                     |
| map       | `np.ndarray` | `np.ndarray[float32]` $\in [0, 1]^{M \times M \times 1}$ that contains a grayscale image representation of the environment.      |
| odometry  | `np.ndarray` | `np.ndarray[float32]` $\in [-1, 1]^{2}$ that represents the agent's normalized relative displacement from its starting position. |
| time_step | `float32`    | `float32` $\in [-1, 1]$ that represents the normalized current time step between 0 and `max_episode_steps` (default 100).        |
The `"map"` key is only included in the static environments.

## Rewards

The reward at each timestep is  the sum of:
- A small action regularization equal to $10^{-3} \cdot{} \lVert \textit{action}\rVert$.
- The negative mean squared error between the agent's prediction and the target.

## Starting State

The agent begins at a uniformly random, valid location within the environment.

## Episode End

The episode ends with the terminate flag set if the maximum number of steps (`max_episode_steps`) is reached.

## Arguments

| Name                   | Type                                                                | Default       | Description                                                                                               |
|------------------------|---------------------------------------------------------------------|---------------|-----------------------------------------------------------------------------------------------------------|
| `dataset`              | `<class 'ap_gym.envs.floor_map.floor_map_dataset.FloorMapDataset'>` |               | Dataset used for the environment containing maps.                                                         |
| `render_mode`          | `typing.Literal['rgb_array']`                                       | `'rgb_array'` | Rendering mode. Currently, only "rgb_array" is supported.                                                 |
| `static_map`           | `<class 'bool'>`                                                    | `False`       | Whether the environment uses a single fixed map (`True`) or samples different maps per episode (`False`). |
| `lidar_beam_count`     | `<class 'int'>`                                                     | `8`           | Number of beams emitted by the LIDAR sensor.                                                              |
| `lidar_range`          | `<class 'float'>`                                                   | `5`           | Maximum range (distance) for LIDAR sensor readings.                                                       |
| `static_map_index`     | `<class 'int'>`                                                     | `0`           | Index of the map used when `static_map=True`. Ignored otherwise.                                          |
| `prefetch`             | `<class 'bool'>`                                                    | `True`        | Whether maps should be prefetched asynchronously for efficiency.                                          |
| `prefetch_buffer_size` | `<class 'int'>`                                                     | `128`         | Buffer size for prefetching maps from the dataset.                                                        |

## Overview of Implemented Environments

| Environment ID                                   | Map type | Static/dynamic | Map size | Map description                                            |
|--------------------------------------------------|----------|----------------|----------|------------------------------------------------------------|
| [LIDARLocMaze-v0](LIDARLocMaze.md)               | Maze     | Dynamic        | 21x21    | Dynamic maze environment with different maps per episode.  |
| [LIDARLocMazeStatic-v0](LIDARLocMazeStatic.md)   | Maze     | Static         | 21x21    | Maze with static map.                                      |
| [LIDARLocRooms-v0](LIDARLocRooms.md)             | Rooms    | Dynamic        | 32x32    | Dynamic rooms environment with different maps per episode. |
| [LIDARLocRoomsStatic-v0](LIDARLocRoomsStatic.md) | Rooms    | Static         | 32x32    | Rooms with static map.                                     |

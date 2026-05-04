# LIDARLocMazeStatic

<p align="center"><img src="img/LIDARLocMazeStatic-v0.gif" alt="LIDARLocMazeStatic-v0" width="200px"/></p>

 This environment is part of the 2D LIDAR Localization Environments. Refer to the [2D LIDAR Localization Environments overview](LIDARLocalization2DEnv.md) for a general description of these environments.

|                     |                       |
|---------------------|-----------------------|
| **Environment ID**  | LIDARLocMazeStatic-v0 |
| **Map type**        | Maze                  |
| **Static/dynamic**  | Static                |
| **Map size**        | 21x21                 |
| **Map description** | Maze with static map. |

## Description

In the LIDARLocMazeStatic environment, the agent faces a map with narrow corridors. Hence, it will always receive information from its LIDAR sensors, but many regions of the maze look alike. The agent must navigate around the map to gather information and localize itself. In this variant, the map stays sconstant, meaning that the agent can memorize the layout of the maze over the course of the training.

## Example Usage

```python

env = ap_gym.make("LIDARLocMazeStatic-v0")

# Or for the vectorized version with 4 environments:
envs = ap_gym.make_vec("LIDARLocMazeStatic-v0", num_envs=4)
```

## Version History

- `v0`: Initial version

# LIDARLocMaze

<p align="center"><img src="img/LIDARLocMaze-v0.gif" alt="LIDARLocMaze-v0" width="200px"/></p>

 This environment is part of the 2D LIDAR Localization Environments. Refer to the [2D LIDAR Localization Environments overview](LIDARLocalization2DEnv.md) for a general description of these environments.

|                     |                                                           |
|---------------------|-----------------------------------------------------------|
| **Environment ID**  | LIDARLocMaze-v0                                           |
| **Map type**        | Maze                                                      |
| **Static/dynamic**  | Dynamic                                                   |
| **Map size**        | 21x21                                                     |
| **Map description** | Dynamic maze environment with different maps per episode. |

## Description

In the LIDARLocMaze environment, the agent faces a map with narrow corridors. Hence, it will always receive information from its LIDAR sensors, but many regions of the maze look alike. The agent must navigate around the map to gather information and localize itself. In this variant, the maze layout changes every episode, meaning that the agent has to learn to process the map it is provided as additional input.

## Example Usage

```python

env = ap_gym.make("LIDARLocMaze-v0")

# Or for the vectorized version with 4 environments:
envs = ap_gym.make_vec("LIDARLocMaze-v0", num_envs=4)
```

## Version History

- `v0`: Initial version

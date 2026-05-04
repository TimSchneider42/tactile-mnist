# LIDARLocRoomsStatic

<p align="center"><img src="img/LIDARLocRoomsStatic-v0.gif" alt="LIDARLocRoomsStatic-v0" width="200px"/></p>

 This environment is part of the 2D LIDAR Localization Environments. Refer to the [2D LIDAR Localization Environments overview](LIDARLocalization2DEnv.md) for a general description of these environments.

|                     |                        |
|---------------------|------------------------|
| **Environment ID**  | LIDARLocRoomsStatic-v0 |
| **Map type**        | Rooms                  |
| **Static/dynamic**  | Static                 |
| **Map size**        | 32x32                  |
| **Map description** | Rooms with static map. |

## Description

In the LIDARLocRooms environment, the agent faces a map with wide open areas. Hence, often it might not receive any information from its LIDAR sensors if it is in the middle of a large room. The agent must, thus, navigate around the map to gather information and localize itself. In this variant, the map stays constant, meaning that the agent can memorize the layout of the rooms over the course of the training.

## Example Usage

```python

env = ap_gym.make("LIDARLocRoomsStatic-v0")

# Or for the vectorized version with 4 environments:
envs = ap_gym.make_vec("LIDARLocRoomsStatic-v0", num_envs=4)
```

## Version History

- `v0`: Initial version

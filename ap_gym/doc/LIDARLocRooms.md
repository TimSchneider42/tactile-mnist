# LIDARLocRooms

<p align="center"><img src="img/LIDARLocRooms-v0.gif" alt="LIDARLocRooms-v0" width="200px"/></p>

 This environment is part of the 2D LIDAR Localization Environments. Refer to the [2D LIDAR Localization Environments overview](LIDARLocalization2DEnv.md) for a general description of these environments.

|                     |                                                            |
|---------------------|------------------------------------------------------------|
| **Environment ID**  | LIDARLocRooms-v0                                           |
| **Map type**        | Rooms                                                      |
| **Static/dynamic**  | Dynamic                                                    |
| **Map size**        | 32x32                                                      |
| **Map description** | Dynamic rooms environment with different maps per episode. |

## Description

In the LIDARLocRooms environment, the agent faces a map with wide open areas. Hence, often it might not receive any information from its LIDAR sensors if it is in the middle of a large room. The agent must, thus, navigate around the map to gather information and localize itself. In this variant, the room layout changes every episode, meaning that the agent has to learn to process the map it is provided as additional input.

## Example Usage

```python

env = ap_gym.make("LIDARLocRooms-v0")

# Or for the vectorized version with 4 environments:
envs = ap_gym.make_vec("LIDARLocRooms-v0", num_envs=4)
```

## Version History

- `v0`: Initial version

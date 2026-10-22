# Starstruck

<p align="center"><img src="img/env/Starstruck-v0.webp" alt="Starstruck-v0" width="200px"/></p>

This environment is part of the tactile classification environments.
Refer to the [tactile classification environments overview](TactileClassificationEnv.md) for a general description of these environments.

|                              |                                                               |
|------------------------------|---------------------------------------------------------------|
| **Environment ID**           | Starstruck-v0                                                 |
| **Dataset**                  | [Starstruck](datasets.md#starstruck) (procedurally generated) |
| **Number of classes**        | 3                                                             |
| **Step limit**               | 32                                                            |
| **Sensor rotation**          | disabled                                                      |
| **Object pose perturbation** | disabled                                                      |

## Description

In the _Starstruck_ environment, the agent must count the number of stars in a scene cluttered with other objects.
Since all stars look the same, distinguishing stars from other objects is rather straightforward.
Instead, the main challenge posed in this environment is to learn an effective search strategy to systematically cover as much space as possible.

The scenes are not stored anywhere but generated procedurally on the fly: each of the 300,000 scenes is derived deterministically from its index with a stable pseudo random number generator (see the [dataset description](datasets.md#starstruck)).
Hence, the agent cannot simply memorize the object arrangements and has to learn a search strategy that generalizes.
For the same reason, there is no separate test split.

## Example Usage

```python
import ap_gym

env = ap_gym.make("Starstruck-v0")

# Or for the vectorized version with 4 environments:
envs = ap_gym.make_vec("Starstruck-v0", num_envs=4)
```

## Version History

- `v0`: Initial release. Initially shipped with a [static version](https://huggingface.co/datasets/TimSchneider42/tactile-mnist-starstruck) of the dataset (3,300 pre-computed scenes), which was replaced by the procedurally generated [Starstruck](datasets.md#starstruck) dataset.

## Variants

| Environment ID         | Description                                                                                                   | Preview                                                                                     |
|------------------------|---------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| Starstruck-DR-v0       | Same as Starstruck-v0 but with [domain randomization](TactilePerceptionConfig.md#sensor-noise) enabled.       | <img src="img/env/Starstruck-DR-v0.webp" alt="Starstruck-DR-v0" width="200px"/>             |
| Starstruck-Depth-v0    | Uses a depth image instead of rendering tactile images.                                                       | <img src="img/env/Starstruck-Depth-v0.webp" alt="Starstruck-Depth-v0" width="200px"/>       |
| Starstruck-Depth-DR-v0 | Same as Starstruck-Depth-v0 but with [domain randomization](TactilePerceptionConfig.md#sensor-noise) enabled. | <img src="img/env/Starstruck-Depth-DR-v0.webp" alt="Starstruck-Depth-DR-v0" width="200px"/> |

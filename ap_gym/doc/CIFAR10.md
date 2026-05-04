# CIFAR10

<p align="center"><img src="img/CIFAR10-v0.gif" alt="CIFAR10-v0" width="200px"/></p>

 This environment is part of the Image Classification Environments. Refer to the [Image Classification Environments overview](ImageClassificationVectorEnv.md) for a general description of these environments.

|                       |                                                                                         |
|-----------------------|-----------------------------------------------------------------------------------------|
| **Environment ID**    | CIFAR10-v0                                                                              |
| **Image type**        | RGB                                                                                     |
| **# data points**     | 50000                                                                                   |
| **Image size**        | 32x32                                                                                   |
| **Glimpse size**      | 5x5                                                                                     |
| **Step limit**        | 16                                                                                      |
| **# classes**         | 10                                                                                      |
| **Image description** | Natural images from the [CIFAR10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html). |

## Description

In the CIFAR10 environment, the agent's objective is to classify natural images into 10 classes. The agent has limited visibility, represented by a small movable glimpse that captures partial views of the image. It must strategically explore different regions of the image to gather enough information for accurate classification.

## Example Usage

```python

env = ap_gym.make("CIFAR10-v0")

# Or for the vectorized version with 4 environments:
envs = ap_gym.make_vec("CIFAR10-v0", num_envs=4)
```

## Version History

- `v0`: Initial version

## Variants

| Environment ID   | Description                                                 | Preview                                                                    |
|------------------|-------------------------------------------------------------|----------------------------------------------------------------------------|
| CIFAR10-train-v0 | Uses the train split of CIFAR10 instead of the train split. | <img src="img/CIFAR10-train-v0.gif" alt="CIFAR10-train-v0" width="200px"/> |
| CIFAR10-test-v0  | Uses the test split of CIFAR10 instead of the train split.  | <img src="img/CIFAR10-test-v0.gif" alt="CIFAR10-test-v0" width="200px"/>   |

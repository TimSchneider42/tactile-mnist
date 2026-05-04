# MNIST

<p align="center"><img src="img/MNIST-v0.gif" alt="MNIST-v0" width="200px"/></p>

 This environment is part of the Image Classification Environments. Refer to the [Image Classification Environments overview](ImageClassificationVectorEnv.md) for a general description of these environments.

|                       |                                                                                 |
|-----------------------|---------------------------------------------------------------------------------|
| **Environment ID**    | MNIST-v0                                                                        |
| **Image type**        | Grayscale                                                                       |
| **# data points**     | 60000                                                                           |
| **Image size**        | 28x28                                                                           |
| **Glimpse size**      | 5x5                                                                             |
| **Step limit**        | 16                                                                              |
| **# classes**         | 10                                                                              |
| **Image description** | Handwritten digits from the [MNIST dataset](http://yann.lecun.com/exdb/mnist/). |

## Description

In the MNIST environment, the agent's objective is to classify images of handwritten digits (0-9). The agent has limited visibility, represented by a small movable glimpse that captures partial views of the image. It must strategically explore different regions of the image to gather enough information for accurate classification.

## Example Usage

```python

env = ap_gym.make("MNIST-v0")

# Or for the vectorized version with 4 environments:
envs = ap_gym.make_vec("MNIST-v0", num_envs=4)
```

## Version History

- `v0`: Initial version

## Variants

| Environment ID | Description                                               | Preview                                                                |
|----------------|-----------------------------------------------------------|------------------------------------------------------------------------|
| MNIST-train-v0 | Uses the train split of MNIST instead of the train split. | <img src="img/MNIST-train-v0.gif" alt="MNIST-train-v0" width="200px"/> |
| MNIST-test-v0  | Uses the test split of MNIST instead of the train split.  | <img src="img/MNIST-test-v0.gif" alt="MNIST-test-v0" width="200px"/>   |

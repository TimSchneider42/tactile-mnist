# TinyImageNet

<p align="center"><img src="img/TinyImageNet-v0.gif" alt="TinyImageNet-v0" width="200px"/></p>

 This environment is part of the Image Classification Environments. Refer to the [Image Classification Environments overview](ImageClassificationVectorEnv.md) for a general description of these environments.

|                       |                                                                                                         |
|-----------------------|---------------------------------------------------------------------------------------------------------|
| **Environment ID**    | TinyImageNet-v0                                                                                         |
| **Image type**        | RGB                                                                                                     |
| **# data points**     | 100000                                                                                                  |
| **Image size**        | 64x64                                                                                                   |
| **Glimpse size**      | 10x10                                                                                                   |
| **Step limit**        | 16                                                                                                      |
| **# classes**         | 200                                                                                                     |
| **Image description** | Natural images from the [Tiny ImageNet dataset](https://huggingface.co/datasets/zh-plus/tiny-imagenet). |

## Description

In the TinyImageNet environment, the agent's objective is to classify natural images into 200 classes. The agent has limited visibility, represented by a small movable glimpse that captures partial views of the image. It must strategically explore different regions of the image to gather enough information for accurate classification.

Compared to the CIFAR10 environment, the TinyImageNet dataset contains more classes and higher resolution images. Also, the glimpse size is larger to account for the higher image resolution. Consequently, this environment introduces additional complexity compared to CIFAR10.

## Example Usage

```python

env = ap_gym.make("TinyImageNet-v0")

# Or for the vectorized version with 4 environments:
envs = ap_gym.make_vec("TinyImageNet-v0", num_envs=4)
```

## Version History

- `v0`: Initial version

## Variants

| Environment ID        | Description                                                      | Preview                                                                              |
|-----------------------|------------------------------------------------------------------|--------------------------------------------------------------------------------------|
| TinyImageNet-train-v0 | Uses the train split of TinyImageNet instead of the train split. | <img src="img/TinyImageNet-train-v0.gif" alt="TinyImageNet-train-v0" width="200px"/> |
| TinyImageNet-test-v0  | Uses the test split of TinyImageNet instead of the train split.  | <img src="img/TinyImageNet-test-v0.gif" alt="TinyImageNet-test-v0" width="200px"/>   |

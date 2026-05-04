# TinyImageNetLoc

<p align="center"><img src="img/TinyImageNetLoc-v0.gif" alt="TinyImageNetLoc-v0" width="200px"/></p>

 This environment is part of the Image Localization Environments. Refer to the [Image Localization Environments overview](ImageLocalizationVectorEnv.md) for a general description of these environments.

|                       |                                                                                                         |
|-----------------------|---------------------------------------------------------------------------------------------------------|
| **Environment ID**    | TinyImageNetLoc-v0                                                                                      |
| **Image type**        | RGB                                                                                                     |
| **# data points**     | 100000                                                                                                  |
| **Image size**        | 64x64                                                                                                   |
| **Glimpse size**      | 10x10                                                                                                   |
| **Step limit**        | 16                                                                                                      |
| **Image description** | Natural images from the [Tiny ImageNet dataset](https://huggingface.co/datasets/zh-plus/tiny-imagenet). |

## Description

In the TinyImageNetLoc environment, the agent's objective is to localize a given glimpse in a natural image. The agent has limited visibility, represented by a small movable glimpse that captures partial views of the image. It must strategically explore different regions of the image to gather enough information for accurate classification.

Compared to the CIFAR10Loc environment, the TinyImageNetLoc dataset contains higher resolution images from more diverse classes. Also, the glimpse size is larger to account for the higher image resolution. Consequently, this environment introduces additional complexity compared to CIFAR10Loc.

## Example Usage

```python

env = ap_gym.make("TinyImageNetLoc-v0")

# Or for the vectorized version with 4 environments:
envs = ap_gym.make_vec("TinyImageNetLoc-v0", num_envs=4)
```

## Version History

- `v0`: Initial version

## Variants

| Environment ID           | Description                                                         | Preview                                                                                    |
|--------------------------|---------------------------------------------------------------------|--------------------------------------------------------------------------------------------|
| TinyImageNetLoc-train-v0 | Uses the train split of TinyImageNetLoc instead of the train split. | <img src="img/TinyImageNetLoc-train-v0.gif" alt="TinyImageNetLoc-train-v0" width="200px"/> |
| TinyImageNetLoc-test-v0  | Uses the test split of TinyImageNetLoc instead of the train split.  | <img src="img/TinyImageNetLoc-test-v0.gif" alt="TinyImageNetLoc-test-v0" width="200px"/>   |

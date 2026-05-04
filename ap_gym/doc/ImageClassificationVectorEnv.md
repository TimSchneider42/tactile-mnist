# Image Classification Environments

In image classification environments, the agent has to classify an image by moving a small glimpse around the
image. The glimpse is never large enough to see the entire image at once, so the agent has to move around to
gather information.

Consider the following example from the [CIFAR10](CIFAR10.md) environment:
<p align="center"><img src="img/CIFAR10-v0.gif" alt="CIFAR10-v0" width="200px"/></p>

Marked in blue is the agent's current glimpse.
We mark the history of glimpses the agent has taken in a color scale ranging from red to green, red meaning that
the agent predicted a probability of 0 for the correct class and green meaning that the agent predicted a
probability of 1 for the correct class.

All image classification environments in _ap_gym_ are instantiations of the
`ap_gym.envs.image_classification.ImageClassificationVectorEnv` class and share the following properties:


## Properties

<table>
  <tr>
    <td><strong>Action Space</strong></td>
    <td><code>Box(-1.0, 1.0, (2,), float32)</code></td>
  </tr>
  <tr>
    <td><strong>Prediction Space</strong></td>
    <td><code>Box(-inf, inf, (K,), float32)</code></td>
  </tr>
  <tr>
    <td><strong>Prediction Target Space</strong></td>
    <td><code>Discrete(K)</code></td>
  </tr>
  <tr>
    <td><strong>Observation Space</strong></td>
    <td><code>Dict({</code><br/><code>&nbsp;&nbsp;"glimpse"    : Box(0.0, 1.0, (G, G, C), float32)</code><br/><code>&nbsp;&nbsp;"glimpse_pos": Box(-1.0, 1.0, (2,), float32)</code><br/><code>&nbsp;&nbsp;"time_step"  : Box(-1.0, 1.0, (), float32)</code><br/><code>})</code></td>
  </tr>
  <tr>
    <td><strong>Loss Function</strong></td>
    <td><code>ap_gym.CrossEntropyLossFn()</code></td>
  </tr>
</table>


 where $K \in \mathbb{N}$ is the number of classes in the environment, $G \in \mathbb{N}$ is the glimpse size, and $C \in \mathbb{N}$ is the number of image channels (1 for grayscale, 3 for RGB).

## Action Space

The action is a `np.ndarray[float32]` $\in [-1, 1]^{2}$ that describes the relative movement of the glimpse sensor. The value is first projected into the unit circle and then scaled by `image_perception_config.max_step_length`, which is 0.2 (10% of the image) by default.

## Prediction Space

The prediction is a `np.ndarray[float32]` $\in \mathbb{R}^{K}$ that contains the logits of the agent's prediction w.r.t. the class label.

## Prediction Target Space

The prediction target is a scalar integer in $\{0, \dots{}, K\}$ that represents the true class.

## Observation Space

The observation is a dictionary with the following keys:

| Key         | Type         | Description                                                                                                                  |
|-------------|--------------|------------------------------------------------------------------------------------------------------------------------------|
| glimpse     | `np.ndarray` | `np.ndarray[float32]` $\in [0, 1]^{G \times G \times C}$ that represents a glimpse of the image.                             |
| glimpse_pos | `np.ndarray` | `np.ndarray[float32]` $\in [-1, 1]^{2}$ that contains the normalized position of the glimpse within the image.               |
| time_step   | `float32`    | `float32` $\in [-1, 1]$ that represents the normalized current time step between 0 and `image_perception_config.step_limit`. |

## Rewards

The reward at each timestep is  the sum of:
- A small action regularization equal to $10^{-3} \cdot{} \lVert \textit{action}\rVert$.
- The negative cross entropy between the agent's prediction and the target.

## Starting State

The glimpse starts at a uniformly random position within the image.

## Episode End

The episode ends with the terminate flag set if the maximum number of steps (`image_perception_config.step_limit`) is reached.

## Arguments

| Name                      | Type                    | Default       | Description                                                                                                                             |
|---------------------------|-------------------------|---------------|-----------------------------------------------------------------------------------------------------------------------------------------|
| `image_perception_config` | `ImagePerceptionConfig` |               | Configuration of the image perception environment. See the [ImagePerceptionConfig documentation](ImagePerceptionConfig.md) for details. |
| `render_mode`             | `Literal['rgb_array']`  | `'rgb_array'` | Rendering mode (currently only `"rgb_array"` is supported).                                                                             |

## Overview of Implemented Environments

| Environment ID                     | Image type | # data points | Image size | Glimpse size | Step limit | # classes | Image description                                                                                       |
|------------------------------------|------------|---------------|------------|--------------|------------|-----------|---------------------------------------------------------------------------------------------------------|
| [CIFAR10-v0](CIFAR10.md)           | RGB        | 50000         | 32x32      | 5x5          | 16         | 10        | Natural images from the [CIFAR10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html).                 |
| [CircleSquare-v0](CircleSquare.md) | Grayscale  | 1568          | 28x28      | 5x5          | 16         | 2         | An image containing either a circle or square.                                                          |
| [MNIST-v0](MNIST.md)               | Grayscale  | 60000         | 28x28      | 5x5          | 16         | 10        | Handwritten digits from the [MNIST dataset](http://yann.lecun.com/exdb/mnist/).                         |
| [TinyImageNet-v0](TinyImageNet.md) | RGB        | 100000        | 64x64      | 10x10        | 16         | 200       | Natural images from the [Tiny ImageNet dataset](https://huggingface.co/datasets/zh-plus/tiny-imagenet). |

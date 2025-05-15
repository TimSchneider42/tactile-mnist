# The Tactile MNIST Benchmark

<table style="border-collapse: collapse; border: none;">
    <tr style="border: none;">
        <td align="center" style="border: none; padding: 10px;">
            <img src="doc/img/env/TactileMNIST-v0.gif" alt="TactileMNIST-v0" width="180px"/><br/>
            <a href="doc/env/TactileMNIST.md">
                TactileMNIST-v0
            </a>
        </td>
        <td align="center" style="border: none; padding: 10px;">
            <img src="doc/img/env/Starstruck-v0.gif" alt="Starstruck-v0" width="180px"/><br/>
            <a href="doc/env/Starstruck.md">
                Starstruck-v0
            </a>
        </td>
        <td align="center" style="border: none; padding: 10px;">
            <img src="doc/img/env/Toolbox-v0.gif" alt="Toolbox-v0" width="180px"/><br/>
            <a href="doc/env/Toolbox.md">
                Toolbox-v0
            </a>
        </td>
        <td align="center" style="border: none; padding: 10px;">
            <img src="doc/img/env/TactileMNISTVolume-v0.gif" alt="TactileMNISTVolume-v0" width="180px"/><br/>
            <a href="doc/env/TactileMNISTVolume.md">
                TactileMNISTVolume-v0
            </a>
        </td>
    </tr>
</table>

Tactile MNIST is a benchmark for learning-based active perception algorithms.
It introduces four simulated tactile perception tasks, ranging from classification and counting to pose and volume estimation.
Each task comes with a unique set of challenges and, thus, Tactile MNIST requires adaptive algorithms and clever exploration strategies.
The aim of Tactile MNIST is to provide an extensible framework for a fair comparison of active tactile perception methods.

Tactile MNIST implements each task as an [ap_gym](https://github.com/TimSchneider42/active-perception-gym) environment and is, thus, very straightforward to set up and use.
In each task, the agent controls a single simulated [GelSight Mini](https://www.gelsight.com/gelsightmini/) above a platform with some task-specific objects.
The agent's objective is to make a prediction about some property of the objects it is exploring, such as their class, count, or pose.

In addition to the simulated benchmark tasks, this package provides access to a large dataset of real tactile images collected from 3D printed MNIST digits and a couple of synthetic datasets.

Further details can be found on our [project page](https://sites.google.com/robot-learning.de/tactile-mnist/), which also links to the paper.

## Installation

Install Tactile MNIST via `pip`:

```bash
pip install tactile-mnist[OPTIONS]
```

where OPTIONS can be any number of the following (comma separated):

- `examples`: installs dependencies for the examples.
- `torch`, `jax`, and `jax-cpu`: installs dependencies for the interactive Tactile MNIST environment with PyTorch, JAX, or JAX without CUDA support, respectively. Without any of those options, the interactive environment cannot be used but the static datasets will still work.

If you encounter problems during the installation or execution, check our [troubleshooting](#troubleshooting) section.

## Contents

This package provides [ap_gym](https://github.com/TimSchneider42/active-perception-gym) environments for four simulated [active tactile classification benchmark tasks](#simulated-active-tactile-perception-benchmark) and access to the [Tactile MNIST datasets](#datasets).
The ap_gym environments can be used to train and evaluate agents on active tactile perception problems on simulated data and are further described in the [Benchmark section](#simulated-active-tactile-perception-benchmark).
In the Tactile MNIST datasets, you find two datasets of 3D CAD models, _MNIST 3D_ and _Starstruck_, and several datasets of simulated and real tactile images.
This package provides an easy way of loading and working with these datasets, as further described in the [Datasets section](#datasets).

## Simulated Active Tactile Perception Benchmark

This package provides [ap_gym](https://github.com/TimSchneider42/active-perception-gym) environments for four active tactile perception environments: [TactileMNIST](doc/TactileMNIST.md), [Starstruck](doc/Starstruck.md), [Toolbox](doc/Toolbox.md), and [TactileMNISTVolume](doc/TactileMNISTVolume.md).
In all environments, the agent must solve a perception problem by actively controlling a [GelSight Mini](https://www.gelsight.com/gelsightmini/) tactile sensor in a simulated environment.

The _TactileMNIST_ environment challenges the agent to find and classify a 3D MNIST model as quickly as possible.
Aside of finding the object, the main challenge in the TactileMNIST environment is to learn contour following strategies to efficiently classify it once found.

In the _Starstruck_ environment, the agent must count the number of stars in a scene cluttered with other objects.
Since all stars look the same, distinguishing stars from other objects is rather straightforward.
Instead, the main challenge posed in this environment is to learn an effective search strategy to systematically cover as much space as possible.

The _Toolbox_ environment challenges the agent to locate a wrench positioned randomly on a platform and estimate its precise 2D position and 1D orientation.
Unlike the previous classification tasks, Toolbox is poses a regression problem that requires combining multiple touch observations to resolve ambiguities inherent in the wrench’s shape.
For example, touching the handle may reveal lateral placement but not longitudinal position or orientation, making it critical for the agent to explore strategically and seek out one of the wrench’s ends to accurately determine its pose.
Overall, the Toolbox tests the agent’s ability to both find and precisely localize an object through sequential tactile exploration.

Finally, the _TactileMNISTVolume_ environment poses another regression problem.
Here, the agent must determine the exact volume of the 3D MNIST model it is given.
Thus, unlike in the TactileMNIST environment, where a couple of touches might already be sufficient for classification, in TactileMNISTVolume, the agent has to make sure to explore the entire object.

A detailed description of the environments can be found [here](doc/TactilePerceptionEnv.md).

## Datasets

Aside of the simulated benchmark tasks, this package provides access to two classes of static datasets: 3D mesh datasets and touch datasets.
Below is an overview of the datasets provided in this package:

- **[3D Mesh Datasets](doc/datasets.md#3d-mesh-datasets)**:
    1. **MNIST 3D**: a dataset of 3D models generated from a [high-resolution version of the MNIST dataset](https://arxiv.org/abs/2011.07946).
    2. **Starstruck**: a dataset in which the number of stars in a scene have to be counted (3 classes, 1 - 3 stars per scene).
- **[Touch Datasets](doc/datasets.md#touch-datasets)**
    1. **Synthetic Tactile MNIST**: a dataset of synthetic tactile images generated from the _MNIST 3D_ dataset with the [Taxim simulator](https://arxiv.org/abs/2109.04027).
    2. **Real Tactile MNIST**: a dataset of real tactile images of 3D printed _MNIST 3D_ digits collected with a Franka robot.
    3. **Synthetic Tactile Starstruck**: a dataset of synthetic tactile images generated from the _Starstruck_ dataset with the Taxim simulator.

A detailed description of the datasets can be found in the [Dataset documentation](doc/datasets.md).

## Troubleshooting

### OpenGL `ctypes.ArgumentError`

If you are seeing the following error:
```
ctypes.ArgumentError: ("argument 2: TypeError: No array-type handler for type _ctypes.type (value: <cparam 'P' (0x75561c1c8420)>) registered", (1, <cparam 'P' (0x75561c1c8420)>))
```
then you are likely using Python version 3.12 or higher.

In that case, `PyOpenGL` has to be updated manually to version 3.1.9 or higher:
```bash
pip install --upgrade PyOpenGL>=3.1.9
```
You will see an error message warning of a dependency conflict with `pyrender`, but you can ignore it.

### Torch-Scatter Undefined Symbol Error
If you are seeing errors as such

```
OSError: [...]/torch_scatter/_version_cpu.so: undefined symbol: _ZN3c1017RegisterOperatorsD1Ev
```

then there are CUDA version incompatibilities between `torch_scatter`, PyTorch, and `nvcc`.

If you are seeing errors as such

```
RuntimeError: Not compiled with CUDA support
```

then `nvcc` was not found when installing `torch_scatter`.

In both cases, follow the instructions in the [official torch_scatter repository](https://github.com/rusty1s/pytorch_scatter) to install compatible PyTorch and `torch_scatter` versions.


## License

The project is licensed under the MIT license.

## Contributing

If you wish to contribute to this project, you are welcome to create a pull request.
Please run the [pre-commit](https://pre-commit.com/) hooks before submitting your pull request.
To install the pre-commit hooks, run:

1. [Install pre-commit](https://pre-commit.com/#install)
2. Install the Git hooks by running `pre-commit install` or, alternatively, run `pre-commit run --all-files manually.

# The Tactile MNIST Benchmark

This repository contains code to access the Tactile MNIST simulated benchmark tasks and datasets, as further described on our [project page](https://sites.google.com/robot-learning.de/tactile-mnist/), which also links to the paper.

## Installation

This package can be installed using pip:

```bash
pip install tactile-mnist[OPTIONS]
```

where OPTIONS can be any number of the following (comma separated):

- `examples`: installs dependencies for the examples.
- `torch`, `jax`, and `jax-cpu`: installs dependencies for the interactive Tactile MNIST environment with PyTorch, JAX, or JAX without CUDA support, respectively. Without any of those options, the interactive environment cannot be used but the static datasets will still work.

### Troubleshooting

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

## Contents

This package provides [ap_gym](https://github.com/TimSchneider42/active-perception-gym) environments for two simulated [active tactile classification benchmark tasks](#simulated-active-tactile-perception-benchmark) and access to the [Tactile MNIST datasets](#datasets).
The ap_gym environments can be used to train and evaluate agents on active tactile perception problems on simulated data and are further described in the [Benchmark section](#simulated-active-tactile-perception-benchmark).
In the Tactile MNIST datasets, you find two datasets of 3D CAD models, _MNIST 3D_ and _Starstruck_, and several datasets of simulated and real tactile images.
This package provides an easy way of loading and working with these datasets, as further described in the [Datasets section](#datasets).

## Simulated Active Tactile Perception Benchmark

This package provides [ap_gym](https://github.com/TimSchneider42/active-perception-gym) environments for two active tactile perception environments: [TactileMNIST](doc/TactileMNIST.md) and [Starstruck](doc/Starstruck.md).
In both environments, the agent must solve a perception problem by actively controlling a [GelSight Mini](https://www.gelsight.com/gelsightmini/) tactile sensor in a simulated environment.

<div align="center">
    <table style="border-collapse: collapse; border: none;">
        <tr style="border: none;">
            <td align="center" style="border: none; padding: 10px;">
                <img src="img/env/TactileMNIST-v0.gif" alt="TactileMNIST-v0" width="200px"/><br/>
                <a href="/TactileMNIST.md">
                    TactileMNIST-v0
                </a>
            </td>
            <td align="center" style="border: none; padding: 10px;">
                <img src="img/env/Starstruck-v0.gif" alt="Starstruck-v0" width="200px"/><br/>
                <a href="Starstruck.md">
                    Starstruck-v0
                </a>
            </td>
        </tr>
    </table>
</div>

The _TactileMNIST_ environment challenges the agent to find and classify a 3D MNIST model as quickly as possible.
Aside of finding the object, the main challenge in the TactileMNIST environment is to learn contour following strategies to efficiently classify it once found.

In the _Starstruck_ environment, the agent must count the number of stars in a scene cluttered with other objects.
Since all stars look the same, distinguishing stars from the other objects is rather straightforward.
Instead, the main challenge posed in this environment is to learn an effective search strategy to systematically cover as much space as possible.

Thus, although both environments might seem visually similar, they pose significantly different challenges to the agent.

A detailed description of the environments can be found [here](doc/TactileClassificationEnv.md).

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

## License

The project is licensed under the MIT license.

## Contributing

If you wish to contribute to this project, you are welcome to create a pull request.
Please run the [pre-commit](https://pre-commit.com/) hooks before submitting your pull request.
To install the pre-commit hooks, run:

1. [Install pre-commit](https://pre-commit.com/#install)
2. Install the Git hooks by running `pre-commit install` or, alternatively, run `pre-commit run --all-files manually.

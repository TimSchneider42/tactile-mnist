# Baseline Data

This directory contains the learning curves of the baselines evaluated on the environments of this package.
They are the data behind the plots on the [project page](https://timschneider42.github.io/tactile-mnist/).
Every method was trained for 5M environment steps on 5 seeds per environment.

## Layout

```
baseline-data/<environment>/<method>/<metric>.csv
baseline-data/<environment>/<method>/<metric>_detailed.csv
```

`<environment>` is the environment name without version suffix.
`<method>` is one of
- `crossq`: [CrossQ](https://arxiv.org/abs/1902.05605),
- `sac`: [Soft Actor-Critic](https://arxiv.org/abs/1801.01290),
- `rand_act`: random actions; only the predictor is trained.

## File formats

`<metric>.csv` holds the curve aggregated over seeds:

```
x,y_mean,y_std
```

`<metric>_detailed.csv` holds the curve of each seed:

```
x,seed_0,seed_1,seed_2,seed_3,seed_4
```

`x` is the number of training environment steps (summed over all parallel environments).
Each seed's curve is smoothed with a Gaussian filter (standard deviation of 3% of the training length) and resampled at 50 evenly spaced points.
`y_mean` and `y_std` are the mean and (population) standard deviation over seeds at each point.
`nan` in a `_detailed.csv` marks points a run did not reach.

## Metrics

Metrics prefixed with `train_` are measured on the training episodes.
Metrics prefixed with `eval_` are measured on separate evaluation episodes of the test split (`<environment>-test-v0`), for environments that have one; these environments are trained on `<environment>-train-v0`.
All values are per-episode quantities averaged over the episodes completed around each point.

| Metric               | Meaning                                                                                                           |
|----------------------|-------------------------------------------------------------------------------------------------------------------|
| `*_ret`              | Episodic return.                                                                                                  |
| `*_loss`             | Prediction loss summed over the episode.                                                                          |
| `*_mean_loss`        | Prediction loss averaged over the steps of the episode.                                                           |
| `*_avg`              | Classification: accuracy averaged over the steps of the episode. Regression: Euclidean distance averaged likewise. |
| `*_final`            | Classification: accuracy at the last step of the episode. Regression: Euclidean distance at the last step.         |
| `train_avg_lin`, `train_final_lin` | Toolbox only: linear (position) error of the pose estimate, averaged over the episode / at its last step.      |
| `train_avg_ang`, `train_final_ang` | Toolbox only: angular error of the pose estimate, averaged over the episode / at its last step.                |
| `eval_acc_vec`       | TactileMNIST only: accuracy at each step of the episode (`x` is the step index), at the end of training.          |
| `eval_clp_vec`       | TactileMNIST only: probability assigned to the correct label at each step of the episode, at the end of training. |

`eval_acc_vec` and `eval_clp_vec` come without a `_detailed.csv`.

## Environments

| Directory                   | Task           | Evaluation split |
|-----------------------------|----------------|------------------|
| `TactileMNIST`              | Classification | yes              |
| `Starstruck`                | Classification | yes              |
| `PrimitiveGeometries`       | Classification | yes              |
| `TactileMNISTVolume`        | Regression     | yes              |
| `ABCVolume`                 | Regression     | yes              |
| `ABCCenterOfMass`           | Regression     | yes              |
| `PrimitiveGeometriesVolume` | Regression     | yes              |
| `Toolbox`                   | Pose estimation | no              |

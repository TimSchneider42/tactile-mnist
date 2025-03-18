# Datasets

This package provides the following datasets:

- **[3D Mesh Datasets](#3d-mesh-datasets)**:
    1. **MNIST 3D**: a dataset of 3D models generated from a [high-resolution version of the MNIST dataset](https://arxiv.org/abs/2011.07946).
    2. **Starstruck**: a dataset in which the number of stars in a scene have to be counted (3 classes, 1 - 3 stars per scene).
- **[Touch Datasets](#touch-datasets)**
    1. **Synthetic Tactile MNIST**: a dataset of synthetic tactile images generated from the _MNIST 3D_ dataset with the [Taxim simulator](https://arxiv.org/abs/2109.04027).
    2. **Real Tactile MNIST**: a dataset of real tactile images of 3D printed _MNIST 3D_ digits collected with a Franka robot.
    3. **Synthetic Tactile Starstruck**: a dataset of synthetic tactile images generated from the _Starstruck_ dataset with the [Taxim simulator](https://arxiv.org/abs/2109.04027).

All data can be found [https://archimedes.ias.informatik.tu-darmstadt.de/s/EiFPmyqa34DLF8S](here), though this package will download and cache the required files automatically when needed.

## 3D Mesh Datasets

The _MNIST 3D_ and _Starstruck_ datasets can be accessed by creating an instance of `MeshDataset`:

```python
from tactile_mnist import MeshDataset, get_resource

mnist_3d_dataset = MeshDataset.load(get_resource("remote:mnist3d-v0/train"))
starstruck_dataset = MeshDataset.load(get_resource("remote:starstruck-v0/train"))
```

Next to the `train` split, the `test`, `holdout`, `printed_train`, and `printed_test` splits are also available for _MNIST 3D_.
The latter two contain meshes of the digits that were 3D printed and used to collect the _Real Tactile MNIST_ dataset.
A detailed list of available datasets can be found in the [Available 3D Mesh Datasets](#available-3d-mesh-datasets) section.

`MeshDataset` is indexable and loads meshes lazily, so it does not require much memory.
For example, to get the first data point in the dataset:

```python
data_point = mnist_3d_dataset[0]
```

Each data point has the following fields:

- `mesh`: a `trimesh.Trimesh` object containing the mesh of the digit.
- `metadata`: metadata of the data point
    - `id`: the ID of the data point (from the original high resolution MNIST dataset).
    - `label`: the label of the data point (0-9).

`MeshDataset` is also iterable, so you can use it in a for loop:

```python
for data_point in mnist_3d_dataset:
    print(f"Datapoint {data_point.metadata.id}")
    data_point.mesh.show()  # data_point.mesh is a trimesh.Trimesh object
```

If you wish to view a dataset, you can take a look at the [example/view_mesh_dataset.py](example/view_mesh_dataset.py) script.
Check out the [Advanced Dataset Usage](#advanced-dataset-usage) section for a comprehensive overview of the features of `MeshDataset`.

### Available 3D Mesh Datasets

Currently, the following 3D mesh datasets are available:

#### mnist3d-v0

<p align="center">
  <img src="img/tactile_mnist_0.png" alt="MNIST 3D models" width="24%"/>
  <img src="img/tactile_mnist_1.png" alt="MNIST 3D models" width="24%"/>
  <img src="img/tactile_mnist_2.png" alt="MNIST 3D models" width="24%"/>
  <img src="img/tactile_mnist_3.png" alt="MNIST 3D models" width="24%"/>
</p>

3D models generated from a [high-resolution version of the MNIST dataset](https://arxiv.org/abs/2011.07946).

| Split           | Description                                                                                                                                                                                   |
|-----------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `train`         | Training split of *mnist3d*.                                                                                                                                                                  |
| `test`          | Test split of *mnist3d*.                                                                                                                                                                      |
| `printed_train` | Training split of the digits that were 3D printed and used to collect the *Real Tactile MNIST* dataset. Corresponds to the touch data in *tactile_mnist-real-[seq/single]-t256-...-v0/train*. |
| `printed_test`  | Test split of the digits that were 3D printed and used to collect the *Real Tactile MNIST* dataset. Corresponds to the touch data in *tactile_mnist-real-[seq/single]-t256-...-v0/test*.      |
| `holdout`       | Holdout split of *mnist3d*.                                                                                                                                                                   |

#### starstruck-v0

<p align="center">
  <img src="img/starstruck_0.png" alt="Starstruck models" width="24%"/>
  <img src="img/starstruck_1.png" alt="Starstruck models" width="24%"/>
  <img src="img/starstruck_2.png" alt="Starstruck models" width="24%"/>
  <img src="img/starstruck_3.png" alt="Starstruck models" width="24%"/>
</p>

A dataset in which the number of stars in a scene must be counted (3 classes, 1–3 stars per scene).

| Split   | Description                        |
|---------|------------------------------------|
| `train` | Training split of *starstruck-v0*. |
| `test`  | Test split of *starstruck-v0*.     |

To access the datasets, you can use the `MeshDataset` class:

```python
from tactile_mnist import MeshDataset, get_resource

dataset = MeshDataset.load(get_resource("remote:DATASET_NAME/SPLIT_NAME"))
# e.g. dataset = MeshDataset.load(get_resource("remote:mnist3d-v0/train"))
```

## Touch datasets

To access the touch datasets, you can use the `TouchDataset` class:

```python
from tactile_mnist import TouchDataset, get_resource

dataset = TouchDataset(
    get_resource("remote:tactile_mnist-real-single-t256-320x240-v0/train")
)

with dataset as loaded_dataset:
    # Do something with the dataset
    data_point = loaded_dataset[0]
```

The context manager (`with dataset as loaded_dataset`) loads the dataset's metadata, while the memory intensive images in the dataset are by default loaded lazily once they are indexed.
If you wish to trade memory efficiency for performance, you can instruct `TouchDataset` to load the full data points into memory instead:

```python
dataset = TouchDataset(
    get_resource("remote:tactile_mnist-real-single-t256-320x240-v0/train"),
    mode="in_memory",
)
```

Loading full data points into memory requires much more space, but can yield substantial speed-ups, especially if you are using HDDs or plan to request data points more than once.

Note, that some datasets are stored in chunks to save memory.
In this case, use `TouchDataset.open_all` to load all chunks of a dataset:

```python
from tactile_mnist import TouchDataset, get_resource

dataset_chunks = TouchDataset.open_all(
    get_resource("remote:tactile_mnist-syn-single-t32-320x240-v0/train")
)

for dataset_chunk in dataset_chunks:
    with dataset_chunk as loaded_dataset_chunk:
        # Do something with the dataset
        data_point = loaded_dataset_chunk[0]
```

`TouchDataset.open_all` does not yet load the dataset into memory, so it is safe to use with large datasets.
Only when you use the `with` statement, the dataset is loaded into memory.
Also note, that `TouchDataset.open_all` can also be safely used with non-chunked datasets, as it will simply return a list with a single dataset.

See [Available Touch Datasets](#available-touch-datasets) for a list of available touch datasets.
For a more complete example of how to use the touch datasets, refer to [example/view_touch_dataset.py](example/view_touch_dataset.py).

### Datapoint Structure

Tactile MNIST contains two types of touch datasets: _image sequence_ (_seq_) datasets and _single image_ (_single_) datasets.
In _image sequence_ datasets, each data point is a short video sequence of the sensor being pressed in the object, while _single image_ data points contain just a single snapshot of a touch.
Depending on the type of dataset (_seq_ or _single_), the data points have different fields:

**_seq_ datasets**: each data point is a sequence of tactile images, from the moment the robot starts pressing down on
the digit until it has retracted its end effector again.

- `sensor_image_seq`: `np.ndarray` containing a sequence of tactile images (shape N x 240 x 320 x 3, where N is the
  sequence length).
- `metadata`: metadata of the data point
    - `label`: the label of the data point (0-9).
    - `pos_in_cell`: the intended 2D position of the touch in the cell frame (in meters).
    - `object_id`: the ID of the data point (from the original high resolution MNIST dataset).
    - `round_id`: the ID of the round in which the data point was collected.
    - `touch_no`: the ID of the touch in the round.
    - `info`: additional information about the data point (e.g. the ID of the gel used to collect it).
    - `touch_start_time_rel`: time stamp of the first registered contact with the object (in seconds).
    - `touch_end_time_rel`: time stamp of the last registered contact with the object (in seconds).
    - `time_stamp_rel_seq`: time stamps of the individual frames in the sequence (in seconds).
    - `gel_position_cell_frame_seq`: full actual 3D position of the gel in the cell frame for each frame in the sequence (in
      meters).
    - `gel_orientation_cell_frame_seq`: full actual 3D orientation of the gel in the cell frame for each frame in the
      sequence (as a quaternion; x, y, z, w).

**_single_ datasets**: each data point is a single tactile image, extracted from the corresponding _seq_ dataset by `touch_dataset/touch_dataset_to_single.py`.

- `sensor_image`: `np.ndarray` containing a single tactile image (shape 240 x 320 x 3).
- `metadata`: metadata of the data point
    - `label`: the label of the data point (0-9).
    - `pos_in_cell`: the intended 2D position of the touch in the cell frame (in meters).
    - `object_id`: the ID of the data point (from the original high resolution MNIST dataset).
    - `round_id`: the ID of the round in which the data point was collected.
    - `touch_no`: the ID of the touch in the round.
    - `info`: additional information about the data point (e.g. the ID of the gel used to collect it).
    - `gel_position_cell_frame`: full actual 3D position of the gel in the cell frame (in meters).
    - `gel_orientation_cell_frame`: full actual 3D orientation of the gel in the cell frame (as a quaternion; x, y, z, w).

Here, a round is defined as a sequence of touches on the same object, during which the object might move due to the touches but is otherwise not externally influenced.
A touch is a single press with the robot's end effector on the object.

The coordinate frame of each cell is in its center, with the x-axis pointing to the right of the robot, the y-axis pointing away from the robot, and the z-axis pointing up, orthogonal to the cell surface.

### Using Tactile MNIST datasets

For training, it might be necessary to group the touches by round.
To do this, you can use the `BaseTouchDataset.rounds` property, which returns a dictionary mapping from round IDs to a dataset of touches in that round.

```python
with dataset as loaded_dataset:
    for round_id, round_data in loaded_dataset.rounds:
        # round_data is a touch dataset

        # Iterate over the touches in the round
        for touch in round_data:
            # Do something with the touch
            pass
```

We also provide a `TouchDatasetRoundIterator` class, which can be used to iterate over the rounds in multiple datasets.

```python
from tactile_mnist import TouchDatasetRoundIterator, TouchDataset, get_resource

dataset_chunks = TouchDataset.open_all(
    get_resource("remote:tactile_mnist-real-single-t256-320x240-v0/train")
)
for round_data in TouchDatasetRoundIterator(
        dataset_chunks, dataset_prefetch_count=1, shuffle=True
):
    # round_data is a touch dataset
    # Iterate over the touches in the round
    for touch in round_data:
        # Do something with the touch
        pass
```

`TouchDatasetRoundIterator` takes care of loading and unloading datasets and supports both shuffling and thread-based prefetching of datasets.
A full example of how to use `TouchDatasetRoundIterator` can be found in [example/touch_dataset_round_iterator.py](example/touch_dataset_round_iterator.py).

### Available Touch Datasets

This package provides three classes of touch datasets: [Real Tactile MNIST](#real-tactile-mnist), [Synthetic Tactile MNIST](#synthetic-tactile-mnist), and [Synthetic Tactile Starstruck](#synthetic-tactile-starstruck).
Each class contains multiple datasets and each dataset has a training (`train`) and test (`test`) split.

| Name                                           | 3D Model Dataset | Type     | # Rounds         | # Touches / Round | Sensor Resolution | Description                                                                                                                                                                                                    | Preview                                                                                                                                    |
|------------------------------------------------|------------------|----------|------------------|-------------------|-------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| `tactile_mnist-real-seq-t256-320x240-v0`       | `mnist3d-v0`     | _seq_    | 500 / 100        | 256               | 320 x 240         | Real tactile images of 3D printed _MNIST 3D_ digits collected with a Franka robot. The `train` and `test` splits of this dataset corresponds to the `printed_train` and `printed_test` splits of _mnist3d-v0_. | <img src="img/tactile_mnist-real-seq-t256-320x240-v0.gif" alt="tactile_mnist-real-seq-t256-320x240-v0 preview" width="200px">              |
| `tactile_mnist-real-single-t256-320x240-v0`    | `mnist3d-v0`     | _single_ | 500 / 100        | 256               | 320 x 240         | _Single_ version of `tactile_mnist-real-seq-t256-320x240-v0`                                                                                                                                                   | <img src="img/tactile_mnist-real-single-t256-320x240-v0.jpeg" alt="tactile_mnist-real-single-t256-320x240-v0 preview" width="200px">       |
| `tactile_mnist-real-single-t256-64x64-v0`      | `mnist3d-v0`     | _single_ | 500 / 100        | 256               | 64 x 64           | `tactile_mnist-real-single-t256-320x240-v0` scaled to a 64x64 resolution.                                                                                                                                      | <img src="img/tactile_mnist-real-single-t256-64x64-v0.jpeg" alt="tactile_mnist-real-single-t256-64x64-v0 preview" width="200px">           |
| `tactile_mnist-syn-single-t32-320x240-v0`      | `mnist3d-v0`     | _single_ | 193,280 / 16,000 | 32                | 320 x 240         | Synthetic tactile images generated from the _MNIST 3D_ dataset with the Taxim simulator.                                                                                                                       | <img src="img/tactile_mnist-syn-single-t32-320x240-v0.jpeg" alt="tactile_mnist-syn-single-t32-320x240-v0 preview" width="200px">           |
| `tactile_mnist-syn-single-t32-64x64-v0`        | `mnist3d-v0`     | _single_ | 193,280 / 16,000 | 32                | 64 x 64           | `tactile_mnist-syn-single-t32-320x240-v0` scaled to a 64x64 resolution.                                                                                                                                        | <img src="img/tactile_mnist-syn-single-t32-64x64-v0.jpeg" alt="tactile_mnist-syn-single-t32-64x64-v0 preview" width="200px">               |
| `tactile_starstruck-syn-single-t32-320x240-v0` | `starstruck-v0`  | _single_ | 16,000 / 1,600   | 32                | 320 x 240         | Synthetic tactile images generated from the _Starstruck_ dataset with the Taxim simulator.                                                                                                                     | <img src="img/tactile_starstruck-syn-single-t32-320x240-v0.jpeg" alt="tactile_starstruck-syn-single-t32-320x240-v0 preview" width="200px"> |
| `tactile_starstruck-syn-single-t32-64x64-v0`   | `starstruck-v0`  | _single_ | 16,000 / 1,600   | 32                | 64 x 64           | `tactile_starstruck-syn-single-t32-320x240-v0` scaled to a 64x64 resolution.                                                                                                                                   | <img src="img/tactile_starstruck-syn-single-t32-64x64-v0.jpeg" alt="tactile_starstruck-syn-single-t32-64x64-v0 preview" width="200px">     |

To access the datasets, you can use the `TouchDataset` class:

```python
from tactile_mnist import TouchDataset, get_resource

dataset_chunks = TouchDataset.open_all(
    get_resource("remote:DATASET_NAME/SPLIT_NAME")
)
```

For details about the data collection procedure see the [Data Collection](#data-collection) section.

### Tactile Data Collection

In the following, we provide a brief overview of the data collection procedure for the _Real Tactile MNIST_ and _Synthetic Tactile MNIST_ datasets.
For more details, check out our paper linked in our [project page](https://sites.google.com/robot-learning.de/tactile-mnist/).

#### Real Tactile Data Collection

Using a Franka robot, we collected 153,600 real tactile touches of 3D printed digits from the _MNIST 3D_ dataset.
For each touch, the robot's end effector was pressed down on the digit, and, once the measured force exceeded 5N, retracted again.
Touch positions were sampled uniformly from 2D cell coordinates, while the orientation of the sensor was kept fixed, parallel to the cell surface.
After collecting half of the total data points, the gel was replaced with a new one.

<img src="img/tactile_mnist_collection.jpeg" alt="Collection of the Real Tactile MNIST dataset" width="100%"/>

#### Synthetic Data Generation

Using the [Taxim simulator](https://arxiv.org/abs/2109.04027), we generate synthetic touches from the _MNIST 3D_ and _Starstruck_ datasets.
Touch positions were sampled uniformly from 2D cell coordinates, while the orientation of the sensor was kept fixed, parallel to the cell surface.

## Advanced Dataset Usage

`MeshDataset` and `LoadedTouchDataset` inherit from `Dataset`, which provides some useful functionality for working
with the datasets.

### Indexing

`Dataset` is indexable, so you can use the `[]` operator to get a data point from the dataset:

```python
data_point = dataset[0]
```

You can also use slicing to get a subset of the dataset:

```python
subset = dataset[0:10]
```

This will return a new `Dataset` object containing the subset of the original dataset.

Finally, you can use a list of indices to get a subset of the dataset:

```python
subset = dataset[[0, 1, 8, 9]]
```

Crucially, no datapoint will be loaded when using slices or index lists to create subsets, making these operations very
efficient.

```python
subset = dataset[0:10]  # No data points are loaded here
data_point = subset[0]  # Only the first data point is loaded here
```

The length of the dataset can be obtained using the `len` function:

```python
length = len(dataset)
```

### Iteration

Dataset is iterable, so you can use it in a for loop:

```python
for data_point in dataset:
# Do something with the data point
```

### Concatenating Datasets

Two or more datasets can be concatenated using `Dataset.concatenate` or the `+` operator:

```python
concatenated_dataset = Dataset.concatenate(dataset_1, dataset_2, dataset_3)

# Or (equivalent but less efficient):
concatenated_dataset = dataset_1 + dataset_2 + dataset_3
```

### Viewing datapoint metadata without loading them

You can view the metadata of data points without loading them fully using the `Dataset.partial` property:

```python
dp_metadata = dataset.partial[0]

# Equivalent to, but much more efficient than:
dp_metadata = dataset[0].metadata
```

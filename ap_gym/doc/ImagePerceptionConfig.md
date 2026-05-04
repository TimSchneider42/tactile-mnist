# ImagePerceptionConfig

The `ap_gym.envs.image.ImagePerceptionConfig` class is a configuration class for image perception environments, such as
`ap_gym.envs.ImageClassification` and `ap_gym.envs.ImageLocalization`.
It holds the following parameters:

| Parameter                  | Type                                           | Default  | Description                                                                                                                                              |
|----------------------------|------------------------------------------------|----------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| `dataset`                  | `ap_gym.envs.image.ImageClassificationDataset` |          | Dataset to use. Implemented types of datasets are `ap_gym.envs.image.CircleSquareDataset` and `ap_gym.envs.image.HuggingfaceImageClassificationDataset`. |
| `sensor_size`              | `tuple[int, int]`                              | `(5, 5)` | Size of the glimpse sensor in pixels.                                                                                                                    |
| `sensor_scale`             | `float`                                        | `1.0`    | Relation of glimpse pixels to image pixels. A value of 2 means that glimpse pixels are twice as large as image pixels.                                   |
| `max_step_length`          | `float \| Sequence[float]`                     | `0.2`    | Maximum normalized sensor movement per step relative to the total image size. Can be a single float or a sequence of floats for per-axis movement.       |
| `step_limit`               | `int`                                          | `16`     | Maximum steps per episode. After this number of steps, the terminate flag will be set.                                                                   |
| `display_visitation`       | `bool`                                         | `True`   | Visualize glimpse visitation history during rendering.                                                                                                   |
| `render_unvisited_opacity` | `float`                                        | `0.0`    | Opacity of the overlay used for unvisited areas in the image.                                                                                            |
| `render_visited_opacity`   | `float`                                        | `0.3`    | Opacity of the overlay used for visited areas in the image.                                                                                              |
| `prefetch_buffer_size`     | `int`                                          | `128`    | Size of the prefetching buffer when using prefetching.                                                                                                   |
| `prefetch`                 | `bool`                                         | `True`   | Whether to prefetch data points from the data set                                                                                                        |

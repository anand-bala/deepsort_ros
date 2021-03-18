# Deep SORT ROS

A ROS wrapper for the _Simple Online and Realtime Tracking with a Deep Association Metric_
(Deep SORT) algorithm.

- [Original Repo](https://github.com/nwojke/deep_sort)

## Design

It subscribes to a topic publishing
[`perception_interfaces/BoundingBoxes`](https://github.com/anand-bala/perception_ros2)
and outputs another stream of bounding boxes with tracking information added.

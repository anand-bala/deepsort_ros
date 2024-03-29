from collections import deque
from typing import Deque, Tuple

import numpy as np
import rclpy
from perception_interfaces.msg import BoundingBox, BoundingBoxes
from rclpy.node import Node

from deep_sort.detection import Detection
from deep_sort.nn_matching import NearestNeighborDistanceMetric
from deep_sort.preprocessing import non_max_suppression
from deep_sort.track import Track
from deep_sort.tracker import Tracker


class TrackerNode(Node):
    def __init__(self):
        super().__init__("deepsort_tracker")
        self.get_logger().info("*****************************")
        self.get_logger().info(" * Deep SORT Node ")
        self.get_logger().info("*****************************")
        self.get_logger().info(" * namespace: {}".format(self.get_namespace()))
        self.get_logger().info(" * node name: {}".format(self.get_name()))

        self._bbox_sub = self.create_subscription(
            BoundingBoxes, "bounding_boxes", self.bbox_callback, 2
        )
        self._bbox_pub = self.create_publisher(BoundingBoxes, "~/tracked_objects", 10)
        # TODO: Do I need to subscribe to any image?
        self.declare_parameters(
            namespace=self.get_namespace(),
            parameters=[
                ("max_cosine_distance", 0.2),
                ("nn_budget", 100),
                ("min_confidence", 0.8),
                ("nms_max_overlap", 1.0),
            ],
        )

        max_cosine_distance = self.get_parameter("max_cosine_distance")
        nn_budget = self.get_parameter("nn_budget")
        metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self._tracker = Tracker(metric)
        self.get_logger().info(" * Tracker ready for Bounding Boxes")

    def bbox_callback(self, msg: BoundingBoxes) -> None:
        def bbox_to_tlwh(bbox: BoundingBox) -> Tuple[float, float, float, float]:
            left = bbox.xmin
            top = bbox.ymin
            width = abs(bbox.xmax - bbox.xmin)
            height = abs(bbox.ymax - bbox.ymin)
            return left, top, width, height

        detections = [
            Detection(
                tlwh=bbox_to_tlwh(box),
                confidence=box.probability,
                class_id=box.class_id,
                class_name=box.class_label,
                feature=None,
            )
            for box in msg.bounding_boxes
        ]
        received_time = self.get_clock().now()

        nms_max_overlap = self.get_parameter("nms_max_overlap")
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Update tracker.
        self._tracker.predict()
        self._tracker.update(detections)

        # Store results.
        results = deque()  # type: Deque[BoundingBox]
        track: Track
        for track in self._tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            # bbox = track.to_tlwh()  # type: np.ndarray
            bbox = track.to_tlbr()  # type: np.ndarray
            result = BoundingBox()
            result.xmin = bbox[0]
            result.ymin = bbox[1]
            result.xmax = bbox[2]
            result.ymax = bbox[3]
            result.class_id = track.get_class_id()
            result.class_label = track.get_class()
            result.probability = track.get_confidence()
            result.tracking_id = track.track_id
            results.append(result)

        result_msg = BoundingBoxes()
        result_msg.header = msg.header
        result_msg.header.stamp = received_time
        result_msg.height = msg.height
        result_msg.width = msg.width
        result_msg.bounding_boxes = results
        self._bbox_pub.publish(result_msg)


def main(args=None):
    rclpy.init(args=args)
    node = TrackerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

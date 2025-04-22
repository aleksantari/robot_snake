#!/usr/bin/env python3
"""
ROS 2 dummy scaffold for real‑time image segmentation
----------------------------------------------------
Subscribes to `/image_raw`, converts ROS 2 Image messages to `cv2` matrices
with `cv_bridge`, applies a *placeholder* segmentation (circular mask in
centre of frame), then publishes the result on `/image_segmented`.

Usage (after installing in a ROS 2 Python package):
    ros2 run snake_segmentation segmentation_node \
        --ros-args -r /image_raw:=/camera/image_raw
"""

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class SegmentationNode(Node):
    """Minimal image‑processing node for the snake‑robot vision pipeline."""

    def __init__(self) -> None:
        super().__init__("snake_segmentation_node")

        # Bridge between ROS 2 <‑> OpenCV
        self.bridge = CvBridge()

        # Subscribe to raw camera frames (remap at launch if topic differs)
        self.subscription = self.create_subscription(
            Image,            # message type
            "/image_raw",     # default input topic
            self.image_callback,
            10,               # QoS history depth
        )
        self.subscription  # prevent unused‑variable warning

        # Advertise processed image topic
        self.publisher = self.create_publisher(Image, "/image_segmented", 10)

        self.get_logger().info("Snake segmentation node initialised")

    # ──────────────────────────────────────────────────────────────────────
    def image_callback(self, msg: Image) -> None:  # noqa: D401 (imperative)
        """Convert incoming frame, apply dummy segmentation, republish."""
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:  # pylint: disable=broad-except
            self.get_logger().error(f"imgmsg_to_cv2 failed: {e}")
            return

        # ── Dummy segmentation: keep a central circular ROI only ─────────
        height, width = frame.shape[:2]
        mask = cv2.circle(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
            center=(width // 2, height // 2),
            radius=min(width, height) // 4,
            color=255,
            thickness=-1,
        )
        segmented = cv2.bitwise_and(frame, frame, mask=mask)
        # ─────────────────────────────────────────────────────────────────

        out_msg = self.bridge.cv2_to_imgmsg(segmented, encoding="bgr8")
        out_msg.header = msg.header  # preserve timestamp & frame_id
        self.publisher.publish(out_msg)


# ──────────────────────────────────────────────────────────────────────────

def main(args=None):  # noqa: D401
    rclpy.init(args=args)
    node = SegmentationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

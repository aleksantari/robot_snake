#!/usr/bin/env python3
"""
ROS 2 node: real-time instance-segmentation with Ultralytics YOLO v11
===================================================================
Replaces the previous dummy mask with a production-ready pipeline that:
  • Subscribes to `/image_raw` (sensor_msgs/msg/Image)
  • Runs a YOLO v11-Seg model on the Apple-silicon GPU (or CPU/CUDA)
  • Publishes a colour-masked image on `/image_segmented`

Launch example (inside your ROS 2 workspace):
    ros2 run snake_segmentation segmentation_node \
        --ros-args \
        -p model_path:=yolo11n-seg.pt \
        -p device:=mps \
        -r /image_raw:=/camera/image_raw

Model weights should live in:
  install/snake_segmentation/share/snake_segmentation/models/
or you can pass an absolute path via `model_path`.
"""

from __future__ import annotations

import os
import cv2
import numpy as np
import rclpy
import torch
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from ultralytics import YOLO
from ament_index_python.packages import get_package_share_directory


class YoloSegNode(Node):
    """ROS 2 node running YOLO v11 instance-segmentation in real time."""

    def __init__(self) -> None:
        super().__init__("snake_segmentation_node")

        # -------- Parameters (declare → read once) ----------------------
        self.declare_parameter("model_path", "yolo11n-seg.pt")
        self.declare_parameter("device", "mps")   # cpu|cuda|mps
        self.declare_parameter("img_size", 640)   # inference res
        self.declare_parameter("conf", 0.25)      # confidence threshold

        # read parameters
        model_name = self.get_parameter("model_path").value
        device_str = self.get_parameter("device").value
        self.img_size = self.get_parameter("img_size").value
        self.conf     = float(self.get_parameter("conf").value)

        # -------- Resolve model path -----------------------------------
        # If absolute, use it; otherwise assume it's in share/.../models/
        if os.path.isabs(model_name):
            model_path = model_name
        else:
            share_dir = get_package_share_directory('snake_segmentation')
            model_path = os.path.join(share_dir, 'models', model_name)

        if not os.path.exists(model_path):
            self.get_logger().error(f"Model file not found: {model_path}")
            raise FileNotFoundError(f"Cannot locate model at '{model_path}'")

        # -------- Load model & handle device ----------------------------
        try:
            _ = torch.device(device_str)
        except Exception as e:
            self.get_logger().warning(
                f"Device '{device_str}' unavailable ({e}); falling back to CPU"
            )
            device_str = "cpu"

        self.model = YOLO(model_path)
        self.model.fuse()
        self.device = device_str
        self.get_logger().info(f"Loaded model '{model_path}' on '{self.device}'")

        # -------- ROS bridges -------------------------------------------
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image, "/image_raw", self.image_callback, 10
        )
        self.publisher    = self.create_publisher(
            Image, "/image_segmented", 10
        )

    def image_callback(self, msg: Image) -> None:
        """Run YOLO segmentation and republish colour-masked image."""
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge failed: {e}")
            return

        h, w = frame.shape[:2]
        try:
            results = self.model.predict(
                source=frame,
                device=self.device,
                imgsz=self.img_size,
                conf=self.conf,
                retina_masks=True,
                verbose=False,
            )
        except Exception as e:
            self.get_logger().error(f"YOLO inference failed: {e}")
            return

        pred  = results[0]
        masks = pred.masks

        if masks is None or masks.data.shape[0] == 0:
            segmented = np.zeros_like(frame)
        else:
            areas = masks.data.sum(dim=(1, 2)).cpu().numpy()
            idx   = int(areas.argmax())
            mask  = masks.data[idx].cpu().numpy()
            mask_resized = cv2.resize(mask, (w, h),
                                      interpolation=cv2.INTER_NEAREST)
            mask_bin      = (mask_resized > 0.5).astype(np.uint8) * 255
            segmented     = cv2.bitwise_and(frame, frame, mask=mask_bin)

        out_msg = self.bridge.cv2_to_imgmsg(segmented, encoding="bgr8")
        out_msg.header = msg.header
        self.publisher.publish(out_msg)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = YoloSegNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

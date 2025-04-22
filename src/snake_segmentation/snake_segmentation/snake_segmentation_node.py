#!/usr/bin/env python3
"""
ROS 2 node: real‑time instance‑segmentation with Ultralytics YOLO v11
===================================================================
Replaces the previous dummy mask with a production‑ready pipeline that:
  • Subscribes to `/image_raw` (sensor_msgs/msg/Image)
  • Runs a YOLO v11‑Seg model on the Apple‑silicon GPU (or CPU/CUDA)
  • Publishes a colour‑masked image on `/image_segmented`

Launch example (inside your ROS 2 workspace):
    ros2 run snake_segmentation segmentation_node \
        --ros-args \
        -p model_path:=yolov11n-seg.pt \
        -p device:=mps \
        -r /image_raw:=/camera/image_raw

Model weights live in any path you pass via the `model_path` parameter.
Drop the file next to this script or keep it in ~/.cache/ultralytics.
"""

from __future__ import annotations

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

# Third‑party: Ultralytics/PyTorch
from ultralytics import YOLO
import torch


class YoloSegNode(Node):
    """ROS 2 node running YOLO v11 instance‑segmentation in real time."""

    # ───────────────────────────────────────── Node Init ──────────────────
    def __init__(self) -> None:  # noqa: D401
        super().__init__("yolo_segmentation_node")

        # -------- Parameters (declare → read once) ----------------------
        self.declare_parameter("model_path", "yolov11n-seg.pt")
        self.declare_parameter("device", "mps")                # cpu|cuda|mps
        self.declare_parameter("img_size", 640)                 # inference res
        self.declare_parameter("conf", 0.25)                    # conf threshold

        model_path = self.get_parameter("model_path").value
        device_str = self.get_parameter("device").value
        self.img_size = self.get_parameter("img_size").value
        self.conf = float(self.get_parameter("conf").value)

        # -------- Load model -------------------------------------------
        try:
            torch_device = torch.device(device_str)
        except Exception as e:  # pylint: disable=broad-except
            self.get_logger().warning(
                f"Device '{device_str}' unavailable ({e}); falling back to CPU"
            )
            torch_device = torch.device("cpu")
            device_str = "cpu"

        self.model = YOLO(model_path)
        self.model.fuse()
        self.device = device_str
        self.get_logger().info(f"Loaded {model_path} on {self.device}")

        # -------- ROS bridges -----------------------------------------
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image, "/image_raw", self.image_callback, 10
        )
        self.publisher = self.create_publisher(Image, "/image_segmented", 10)

    # ─────────────────────────────────────── Callback ────────────────────
    def image_callback(self, msg: Image) -> None:  # noqa: D401
        """Run YOLO segmentation and republish colour‑masked image."""
        # Convert to OpenCV (BGR8 guaranteed)
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:  # pylint: disable=broad-except
            self.get_logger().error(f"cv_bridge conversion failed: {e}")
            return

        h, w = frame.shape[:2]

        # Inference (Ultralytics handles resize/letterbox internally)
        try:
            results = self.model.predict(
                source=frame,
                device=self.device,
                imgsz=self.img_size,
                conf=self.conf,
                retina_masks=True,
                verbose=False,
            )
        except Exception as e:  # pylint: disable=broad-except
            self.get_logger().error(f"YOLO inference failed: {e}")
            return

        pred = results[0]
        masks = pred.masks
        if masks is None or masks.data.shape[0] == 0:
            # No detections — output an all‑black image to keep timing stable
            segmented = np.zeros_like(frame)
        else:
            # For simplicity pick the *largest* instance mask
            areas = masks.data.sum(dim=(1, 2)).cpu().numpy()
            idx = int(areas.argmax())
            mask = masks.data[idx].cpu().numpy()  # float32 0‑1, (H, W)
            mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            mask_bin = (mask_resized > 0.5).astype(np.uint8) * 255
            segmented = cv2.bitwise_and(frame, frame, mask=mask_bin)

        out_msg = self.bridge.cv2_to_imgmsg(segmented, encoding="bgr8")
        out_msg.header = msg.header  # keep sync with other streams
        self.publisher.publish(out_msg)


# ────────────────────────────────────────── main ─────────────────────────

def main(args: list[str] | None = None) -> None:  # noqa: D401
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

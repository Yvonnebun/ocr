from pathlib import Path
import os
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2

from inference.pipeline_floorplan import FloorplanPipelinePredictor


def main() -> None:
    image_path = "/app/test.png"
    wall_a_weights = "/app/teacher2_wall_mixed.pt"
    wall_b_weights = "/app/teacher2_wall_mixed.pt"
    room_weights = "/app/baseline.pt"
    window_weights = "/app/baseline.pt"
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    device = os.getenv("YOLO_DEVICE", "cpu")
    predictor = FloorplanPipelinePredictor(
        wall_a_weights=wall_a_weights,
        wall_b_weights=wall_b_weights,
        room_weights=room_weights,
        window_weights=window_weights,
        device=device,
    )
    bundle = predictor.predict_bundle(image_bgr)

    wall_det_count = len(bundle["wall"]["result"]["detections"])
    room_det_count = len(bundle["room"]["result"]["detections"])
    print(f"Wall detections: {wall_det_count}")
    print(f"Room detections: {room_det_count}")
    print(f"Window count: {bundle['window']['count']}")
    print(
        "Image sizes (orig -> infer): "
        f"{bundle['image']['orig_width']}x{bundle['image']['orig_height']} -> "
        f"{bundle['image']['infer_width']}x{bundle['image']['infer_height']} "
        f"(scale={bundle['image']['scale_factor']:.4f})"
    )
    print(f"Gate status: {bundle['gate']['status']} ({bundle['gate']['reason']})")
    print(f"Total ms: {bundle['total_ms']:.2f}")


if __name__ == "__main__":
    main()

from pathlib import Path
import argparse
import json
import os
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2

import config
from inference.pipeline_floorplan import FloorplanPipelinePredictor


def main() -> None:
    parser = argparse.ArgumentParser(description="Run floorplan wall/room predictors on a single image.")
    parser.add_argument("image_path", help="Path to input image.")
    parser.add_argument(
        "--wall-a-weights",
        default=config.FLOORPLAN_WALL_A_WEIGHTS or "/app/teacher2_wall_mixed.pt",
    )
    parser.add_argument(
        "--wall-b-weights",
        default=config.FLOORPLAN_WALL_B_WEIGHTS or "/app/teacher2_wall_mixed.pt",
    )
    parser.add_argument(
        "--room-weights",
        default=config.FLOORPLAN_ROOM_WEIGHTS or "/app/baseline.pt",
    )
    parser.add_argument(
        "--window-weights",
        default=config.FLOORPLAN_WINDOW_WEIGHTS or "/app/baseline.pt",
    )
    parser.add_argument(
        "--polys-out",
        default=None,
        help="Optional path to write the raw bundle JSON (including polygons).",
    )
    args = parser.parse_args()

    image_path = args.image_path
    wall_a_weights = args.wall_a_weights
    wall_b_weights = args.wall_b_weights
    room_weights = args.room_weights
    window_weights = args.window_weights
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
    wall_poly_count = len(bundle["wall"]["result"].get("polygons", []))
    room_poly_count = len(bundle["room"]["result"].get("polygons", []))
    print(f"Wall detections: {wall_det_count}")
    print(f"Room detections: {room_det_count}")
    print(f"Wall polygons: {wall_poly_count}")
    print(f"Room polygons: {room_poly_count}")
    print(f"Window count: {bundle['window']['count']}")
    print(
        "Image sizes (orig -> infer): "
        f"{bundle['image']['orig_width']}x{bundle['image']['orig_height']} -> "
        f"{bundle['image']['infer_width']}x{bundle['image']['infer_height']} "
        f"(scale={bundle['image']['scale_factor']:.4f})"
    )
    print(f"Gate status: {bundle['gate']['status']} ({bundle['gate']['reason']})")
    print(f"Total ms: {bundle['total_ms']:.2f}")

    if args.polys_out:
        output_path = Path(args.polys_out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote bundle JSON to {output_path}")


if __name__ == "__main__":
    main()

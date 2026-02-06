from pathlib import Path
import argparse
import json
import os
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np

import config
from inference.pipeline_floorplan import FloorplanPipelinePredictor


def _color_for_class(class_id: int | None) -> tuple[int, int, int]:
    palette = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (128, 0, 255),
        (255, 128, 0),
        (0, 128, 255),
    ]
    if class_id is None or class_id < 0:
        return (200, 200, 200)
    return palette[class_id % len(palette)]



def _scale_polygon_points(points: list, scale: float) -> list:
    if not points or scale == 1.0:
        return points
    coords = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    coords = coords / scale
    return coords.tolist()


def _extract_points_and_class(poly: object) -> tuple[list, int | None]:
    if isinstance(poly, dict):
        return poly.get("points") or [], poly.get("class_id")
    if isinstance(poly, list):
        if poly and isinstance(poly[0], (int, float)):
            return poly, None
        return poly, None
    return [], None


def _annotate_polygons(image_bgr: cv2.Mat, polygons: list[object], scale: float = 1.0) -> cv2.Mat:
    annotated = image_bgr.copy()
    for poly in polygons:
        points, class_id = _extract_points_and_class(poly)
        points = _scale_polygon_points(points, scale)

        if len(points) < 3:
            continue
        coords = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        if coords.shape[0] < 3:
            continue
        pts = coords.astype(np.int32).reshape((-1, 1, 2))

        color = _color_for_class(class_id)

        cv2.polylines(annotated, [pts], isClosed=True, color=color, thickness=2)
    return annotated



def _annotate_rings(image_bgr: cv2.Mat, rings: list[dict], scale: float = 1.0) -> cv2.Mat:
    annotated = image_bgr.copy()
    for ring in rings:
        exterior = _scale_polygon_points(ring.get("exterior") or [], scale)
        if len(exterior) >= 6:
            ext = np.asarray(exterior, dtype=np.float32).reshape(-1, 2).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated, [ext], isClosed=True, color=(255, 0, 0), thickness=2)
        for hole in ring.get("holes", []) or []:
            hole_pts = _scale_polygon_points(hole, scale)
            if len(hole_pts) < 6:
                continue
            hp = np.asarray(hole_pts, dtype=np.float32).reshape(-1, 2).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated, [hp], isClosed=True, color=(0, 255, 255), thickness=2)
    return annotated


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
    parser.add_argument(
        "--merge-walls",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Whether to merge wall_a/wall_b polygons before downstream processing.",
    )
    parser.add_argument(
        "--annotated-out",
        default=None,
        help="Optional path to write an annotated image with polygon overlays.",
    )

    parser.add_argument(
        "--draw-holes",
        action="store_true",
        help="When available, draw merged wall rings (exterior + holes) instead of exteriors only.",
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
    bundle = predictor.predict_bundle(image_bgr, merge_walls=args.merge_walls)

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

    if args.annotated_out:

        wall_result = bundle.get("wall", {}).get("result", {})
        if args.merge_walls:
            wall_polys = wall_result.get("polygons", []) or []
            wall_rings = wall_result.get("polygons_rings", []) or []

        else:
            wall_raw = bundle.get("wall_raw", {})
            wall_polys = (wall_raw.get("wall_a", {}).get("polygons", []) or []) + (
                wall_raw.get("wall_b", {}).get("polygons", []) or []
            )

            wall_rings = []
        room_polys = bundle.get("room", {}).get("result", {}).get("polygons", []) or []
        scale_factor = float(bundle.get("image", {}).get("scale_factor", 1.0) or 1.0)
        if args.draw_holes and wall_rings:
            annotated = _annotate_rings(image_bgr, wall_rings, scale=scale_factor)
            annotated = _annotate_polygons(annotated, room_polys, scale=scale_factor)
        else:
            annotated = _annotate_polygons(image_bgr, wall_polys + room_polys, scale=scale_factor)

        output_path = Path(args.annotated_out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), annotated)
        print(f"Wrote annotated image to {output_path}")


if __name__ == "__main__":
    main()

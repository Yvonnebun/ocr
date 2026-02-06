from __future__ import annotations

import time
from typing import List, Optional

import numpy as np

from .contracts import BundleResultTD, ModelResultTD
from .gate import apply_oom_gate
from .postprocess_hooks import merge_wall_predictions, polygons_to_mask, process_room_polygons, rings_to_mask
from .yolo_ultralytics import UltralyticsYoloPredictor


def _infer_space_sanity_stats(wall_result: dict, infer_w: int, infer_h: int) -> dict:
    max_x = 0.0
    max_y = 0.0

    def _consume_points(points):
        nonlocal max_x, max_y
        if not points:
            return
        arr = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        if arr.size == 0:
            return
        max_x = max(max_x, float(np.max(arr[:, 0])))
        max_y = max(max_y, float(np.max(arr[:, 1])))

    for poly in wall_result.get("polygons", []) or []:
        if isinstance(poly, dict):
            _consume_points(poly.get("points", []))
        else:
            _consume_points(poly)

    for ring in wall_result.get("polygons_rings", []) or []:
        _consume_points(ring.get("exterior", []))
        for hole in ring.get("holes", []) or []:
            _consume_points(hole)

    eps = 1e-3
    return {
        "max_x": max_x,
        "max_y": max_y,
        "infer_width": int(infer_w),
        "infer_height": int(infer_h),
        "within_infer_space": bool(max_x <= (infer_w - 1 + eps) and max_y <= (infer_h - 1 + eps)),
    }


def _empty_model_result(width: int, height: int) -> ModelResultTD:
    return {
        "model_name": "skipped",
        "model_version": "n/a",
        "image": {"width": int(width), "height": int(height)},
        "detections": [],
        "polygons": [],
        "meta": {},
    }


class FloorplanPipelinePredictor:
    def __init__(
        self,
        wall_a_weights: str,
        wall_b_weights: str,
        room_weights: str,
        window_weights: str,
        device: str = "auto",
        imgsz: int = 1024,
        half: bool = True,
        max_pixels: int = 8_000_000,
        max_side: int = 4096,
        gate_action: str = "downscale",
    ) -> None:
        self.wall_a_weights = wall_a_weights
        self.wall_b_weights = wall_b_weights
        self.room_weights = room_weights
        self.window_weights = window_weights
        self.device = device
        self.imgsz = imgsz
        self.half = half
        self.wall_a = None
        self.wall_b = None
        self.room = None
        self.window = None
        self.max_pixels = max_pixels
        self.max_side = max_side
        self.gate_action = gate_action

    def _ensure_models(self) -> None:
        if self.wall_a is None:
            self.wall_a = UltralyticsYoloPredictor(
                self.wall_a_weights, device=self.device, imgsz=self.imgsz, half=self.half
            )
        if self.wall_b is None:
            self.wall_b = UltralyticsYoloPredictor(
                self.wall_b_weights, device=self.device, imgsz=self.imgsz, half=self.half
            )
        if self.room is None:
            self.room = UltralyticsYoloPredictor(
                self.room_weights, device=self.device, imgsz=self.imgsz, half=self.half
            )
        if self.window is None:
            self.window = UltralyticsYoloPredictor(
                self.window_weights, device=self.device, imgsz=self.imgsz, half=self.half
            )

    def predict_bundle(
        self,
        image_bgr: np.ndarray,
        image_id: Optional[object] = None,
        conf: float = 0.25,
        iou: float = 0.7,
        max_det: int = 300,
        classes: Optional[List[int]] = None,
        merge_walls: bool = True,
    ) -> BundleResultTD:
        t0 = time.perf_counter()
        gated_image, gate_info = apply_oom_gate(
            image_bgr,
            max_pixels=self.max_pixels,
            max_side=self.max_side,
            action=self.gate_action,
        )
        orig_w = gate_info["orig_width"]
        orig_h = gate_info["orig_height"]
        infer_w = gate_info["infer_width"]
        infer_h = gate_info["infer_height"]
        scale_factor = gate_info["scale_factor"]
        gate = {
            "status": gate_info["status"],
            "reason": gate_info["reason"],
            "limits": gate_info["limits"],
        }

        if gated_image is None:
            empty_result = _empty_model_result(infer_w, infer_h)
            t_end = time.perf_counter()
            return {
                "model_bundle": "floorplan_bundle",
                "image": {
                    "image_id": image_id,
                    "orig_width": orig_w,
                    "orig_height": orig_h,
                    "infer_width": infer_w,
                    "infer_height": infer_h,
                    "scale_factor": scale_factor,
                },
                "gate": gate,
                "wall": {"source_models": ["wall_a", "wall_b"], "merged": False, "result": empty_result},
                "room": {"source_models": ["room"], "postprocessed": False, "result": empty_result},
                "window": {"source_models": ["window"], "count": 0},
                "total_ms": (t_end - t0) * 1000.0,
            }

        self._ensure_models()

        wall_a_result = self.wall_a.predict(gated_image, conf=conf, iou=iou, max_det=max_det, classes=classes)
        wall_b_result = self.wall_b.predict(gated_image, conf=conf, iou=iou, max_det=max_det, classes=classes)
        if merge_walls:
            wall_result = merge_wall_predictions(wall_a_result, wall_b_result, image_shape=(infer_h, infer_w))
            wall_merged = True
        else:
            wall_result = dict(wall_a_result)
            wall_result["meta"] = dict(wall_a_result.get("meta", {}))
            wall_result["meta"]["merged"] = False
            wall_merged = False
        wall_rings = wall_result.get("polygons_rings") if isinstance(wall_result, dict) else None
        if wall_rings:
            wall_mask = rings_to_mask(wall_rings, image_shape=(infer_h, infer_w))
        else:
            wall_mask = polygons_to_mask(wall_result.get("polygons", []) or [], image_shape=(infer_h, infer_w))


        wall_meta = dict(wall_result.get("meta", {}))
        wall_meta["scale_factor_definition"] = "infer_over_orig"
        wall_meta["infer_space_sanity"] = _infer_space_sanity_stats(wall_result, infer_w=infer_w, infer_h=infer_h)
        wall_result["meta"] = wall_meta

        room_input = gated_image.copy()
        if wall_mask.size > 0 and np.any(wall_mask > 0):
            room_input[wall_mask > 0] = (255, 255, 255)

        room_raw = self.room.predict(room_input, conf=conf, iou=iou, max_det=max_det, classes=classes)
        room_final = process_room_polygons(room_raw, wall_result, image_shape=(infer_h, infer_w))

        window_result = self.window.predict(gated_image, conf=conf, iou=iou, max_det=max_det, classes=classes)
        window_count = len(window_result.get("detections", []))

        t_end = time.perf_counter()
        return {
            "model_bundle": "floorplan_bundle",
            "image": {
                "image_id": image_id,
                "orig_width": orig_w,
                "orig_height": orig_h,
                "infer_width": infer_w,
                "infer_height": infer_h,
                "scale_factor": scale_factor,
            },
            "gate": gate,
            "wall": {
                "source_models": ["wall_a", "wall_b"],
                "merged": wall_merged,
                "result": wall_result,
            },
            "wall_raw": {"wall_a": wall_a_result, "wall_b": wall_b_result},
            "room": {
                "source_models": ["room"],
                "postprocessed": False,
                "result": room_final,
            },
            "window": {"source_models": ["window"], "count": int(window_count)},
            "total_ms": (t_end - t0) * 1000.0,
        }

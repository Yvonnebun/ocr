from __future__ import annotations

import time
from typing import List, Optional

import numpy as np

from .contracts import BundleResultTD, ModelResultTD
from .gate import apply_oom_gate
from .postprocess_hooks import merge_wall_predictions, process_room_polygons
from .yolo_ultralytics import UltralyticsYoloPredictor


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
        self.wall_a = UltralyticsYoloPredictor(wall_a_weights, device=device, imgsz=imgsz, half=half)
        self.wall_b = UltralyticsYoloPredictor(wall_b_weights, device=device, imgsz=imgsz, half=half)
        self.room = UltralyticsYoloPredictor(room_weights, device=device, imgsz=imgsz, half=half)
        self.window = UltralyticsYoloPredictor(window_weights, device=device, imgsz=imgsz, half=half)
        self.max_pixels = max_pixels
        self.max_side = max_side
        self.gate_action = gate_action

    def predict_bundle(
        self,
        image_bgr: np.ndarray,
        conf: float = 0.25,
        iou: float = 0.7,
        max_det: int = 300,
        classes: Optional[List[int]] = None,
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

        wall_a_result = self.wall_a.predict(gated_image, conf=conf, iou=iou, max_det=max_det, classes=classes)
        wall_b_result = self.wall_b.predict(gated_image, conf=conf, iou=iou, max_det=max_det, classes=classes)
        wall_merged = merge_wall_predictions(wall_a_result, wall_b_result, image_shape=(infer_h, infer_w))

        room_raw = self.room.predict(gated_image, conf=conf, iou=iou, max_det=max_det, classes=classes)
        room_final = process_room_polygons(room_raw, wall_merged, image_shape=(infer_h, infer_w))

        window_result = self.window.predict(gated_image, conf=conf, iou=iou, max_det=max_det, classes=classes)
        window_count = len(window_result.get("detections", []))

        t_end = time.perf_counter()
        return {
            "model_bundle": "floorplan_bundle",
            "image": {
                "orig_width": orig_w,
                "orig_height": orig_h,
                "infer_width": infer_w,
                "infer_height": infer_h,
                "scale_factor": scale_factor,
            },
            "gate": gate,
            "wall": {
                "source_models": ["wall_a", "wall_b"],
                "merged": False,
                "result": wall_merged,
            },
            "room": {
                "source_models": ["room"],
                "postprocessed": False,
                "result": room_final,
            },
            "window": {"source_models": ["window"], "count": int(window_count)},
            "total_ms": (t_end - t0) * 1000.0,
        }

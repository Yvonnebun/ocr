"""
Local inference wrapper for floorplan models (uses inference/ scripts).
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict

import cv2

import config
from inference.pipeline_floorplan import FloorplanPipelinePredictor
from run_logger import get_run_logger


def _require_weight(value: str, name: str) -> str:
    if not value:
        raise RuntimeError(f"{name} is not set for inference service")
    return value


@lru_cache(maxsize=1)
def _get_predictor() -> FloorplanPipelinePredictor:
    return FloorplanPipelinePredictor(
        wall_a_weights=_require_weight(config.INFERENCE_WALL_A_WEIGHTS, "INFERENCE_WALL_A_WEIGHTS"),
        wall_b_weights=_require_weight(config.INFERENCE_WALL_B_WEIGHTS, "INFERENCE_WALL_B_WEIGHTS"),
        room_weights=_require_weight(config.INFERENCE_ROOM_WEIGHTS, "INFERENCE_ROOM_WEIGHTS"),
        window_weights=_require_weight(config.INFERENCE_WINDOW_WEIGHTS, "INFERENCE_WINDOW_WEIGHTS"),
        device=config.INFERENCE_DEVICE,
        imgsz=config.INFERENCE_IMGSZ,
        half=config.INFERENCE_HALF,
        max_pixels=config.INFERENCE_MAX_PIXELS,
        max_side=config.INFERENCE_MAX_SIDE,
        gate_action=config.INFERENCE_GATE_ACTION,
    )


def infer_polygons(image_path: str) -> Dict[str, Any]:
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    run_logger = get_run_logger()
    if run_logger:
        run_logger.increment("inference_calls")
        run_logger.log_event("inference_call", {"image_path": image_path})

    predictor = _get_predictor()
    return predictor.predict_bundle(img_bgr)

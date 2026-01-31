"""
YOLO-based door detection for page selection.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict, Tuple

import config


def _yolo_available() -> bool:
    try:
        import ultralytics  # noqa: F401
    except Exception:
        return False
    return True


@lru_cache(maxsize=1)
def _load_model():
    from ultralytics import YOLO

    if not config.DOOR_MODEL_PATH:
        raise FileNotFoundError("DOOR_MODEL_PATH is not set")
    if not os.path.exists(config.DOOR_MODEL_PATH):
        raise FileNotFoundError(f"DOOR_MODEL_PATH not found: {config.DOOR_MODEL_PATH}")
    return YOLO(config.DOOR_MODEL_PATH)


def detect_door_count(image_path: str) -> Tuple[int, Dict[str, float]]:
    """
    Run YOLO model to detect doors on the page image.

    Returns:
        (door_count, stats)
    """
    if not config.DOOR_DETECT_ENABLED:
        return 0, {"enabled": 0.0}
    if not _yolo_available():
        return 0, {"enabled": 1.0, "error": 1.0, "reason": "ultralytics_missing"}

    if not config.DOOR_MODEL_PATH:
        return 0, {"enabled": 1.0, "error": 1.0, "reason": "model_path_missing"}
    if not os.path.exists(config.DOOR_MODEL_PATH):
        return 0, {"enabled": 1.0, "error": 1.0, "reason": "model_path_not_found"}

    model = _load_model()
    results = model.predict(
        image_path,
        conf=config.DOOR_CONF_THRESHOLD,
        iou=config.DOOR_IOU_THRESHOLD,
        verbose=False,
    )
    if not results:
        return 0, {"enabled": 1.0, "detections": 0.0}

    result = results[0]
    names = result.names if hasattr(result, "names") else {}
    target_names = set(config.DOOR_CLASS_NAMES)
    door_count = 0

    if result.boxes is None:
        return 0, {"enabled": 1.0, "detections": 0.0}

    for box in result.boxes:
        cls_id = int(box.cls[0].item()) if hasattr(box.cls[0], "item") else int(box.cls[0])
        cls_name = str(names.get(cls_id, "")).strip().lower()
        if cls_name in target_names:
            door_count += 1

    return door_count, {"enabled": 1.0, "detections": float(len(result.boxes))}

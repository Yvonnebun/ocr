from __future__ import annotations

from typing import List, Optional

import numpy as np

from .contracts import BundleResultTD
from .pipeline_floorplan import FloorplanPipelinePredictor


class WallRoomPipelinePredictor:
    def __init__(
        self,
        wall_a_weights: str,
        wall_b_weights: str,
        room_weights: str,
        window_weights: str,
        device: str = "cuda:0",
        imgsz: int = 1024,
        half: bool = True,
        max_pixels: int = 8_000_000,
        max_side: int = 4096,
        gate_action: str = "downscale",
    ) -> None:
        self._delegate = FloorplanPipelinePredictor(
            wall_a_weights=wall_a_weights,
            wall_b_weights=wall_b_weights,
            room_weights=room_weights,
            window_weights=window_weights,
            device=device,
            imgsz=imgsz,
            half=half,
            max_pixels=max_pixels,
            max_side=max_side,
            gate_action=gate_action,
        )

    def predict_bundle(
        self,
        image_bgr: np.ndarray,
        conf: float = 0.25,
        iou: float = 0.7,
        max_det: int = 300,
        classes: Optional[List[int]] = None,
    ) -> BundleResultTD:
        return self._delegate.predict_bundle(
            image_bgr, conf=conf, iou=iou, max_det=max_det, classes=classes
        )

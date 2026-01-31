from __future__ import annotations

from typing import List, Optional

import numpy as np

from .contracts import DetectionTD, ModelResultTD


class UltralyticsYoloPredictor:
    def __init__(self, weights_path: str, device: str = "cuda:0", imgsz: int = 1024, half: bool = True):
        from ultralytics import YOLO

        self.model = YOLO(weights_path)
        self.device = device
        self.imgsz = imgsz
        self.half = half
        self.model_name = getattr(self.model, "model", None).__class__.__name__ if hasattr(self.model, "model") else "yolo"
        self.model_version = getattr(self.model, "version", "unknown")

    def _extract_detections(self, result) -> List[DetectionTD]:
        detections: List[DetectionTD] = []
        if result.boxes is None:
            return detections
        names = result.names if hasattr(result, "names") else {}
        for box in result.boxes:
            cls_id = int(box.cls[0].item()) if hasattr(box.cls[0], "item") else int(box.cls[0])
            score = float(box.conf[0].item()) if hasattr(box.conf[0], "item") else float(box.conf[0])
            xyxy = box.xyxy[0].tolist() if hasattr(box.xyxy[0], "tolist") else list(box.xyxy[0])
            detections.append(
                {
                    "class_id": int(cls_id),
                    "class_name": str(names.get(cls_id, "")),
                    "score": float(score),
                    "bbox_xyxy": [float(v) for v in xyxy],
                }
            )
        return detections

    def predict(
        self,
        image_bgr: np.ndarray,
        conf: float = 0.25,
        iou: float = 0.7,
        max_det: int = 300,
        classes: Optional[List[int]] = None,
    ) -> ModelResultTD:
        h, w = image_bgr.shape[:2]
        results = self.model.predict(
            image_bgr,
            conf=conf,
            iou=iou,
            max_det=max_det,
            classes=classes,
            device=self.device,
            imgsz=self.imgsz,
            half=self.half,
            verbose=False,
        )
        if results:
            result = results[0]
            detections = self._extract_detections(result)
        else:
            detections = []
        return {
            "model_name": self.model_name,
            "model_version": str(self.model_version),
            "image": {"width": int(w), "height": int(h)},
            "detections": detections,
            "polygons": [],
            "meta": {},
        }

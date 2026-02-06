from __future__ import annotations

from typing import List, Optional

import numpy as np

from .contracts import DetectionTD, ModelResultTD


class UltralyticsYoloPredictor:
    def __init__(self, weights_path: str, device: str = "auto", imgsz: int = 1024, half: bool = True):
        from ultralytics import YOLO

        self.model = YOLO(weights_path)
        self.device = self._resolve_device(device)
        self.imgsz = imgsz
        self.half = half if self.device != "cpu" else False
        self.model_name = getattr(self.model, "model", None).__class__.__name__ if hasattr(self.model, "model") else "yolo"
        self.model_version = getattr(self.model, "version", "unknown")

    def _resolve_device(self, device: str) -> str:
        device_str = str(device) if device is not None else "auto"
        normalized = device_str.strip().lower()
        try:
            import torch
        except ImportError:
            return "cpu"

        if normalized in {"auto", ""}:
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                return "cuda:0"
            return "cpu"

        if normalized.startswith("cuda") or normalized.isdigit():
            if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
                return "cpu"

        return device_str

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

    def _extract_polygons(self, result) -> List[List[float]]:
        polygons: List[List[float]] = []
        masks = getattr(result, "masks", None)
        if masks is None:
            return polygons
        mask_polys = getattr(masks, "xy", None)
        if not mask_polys:
            return polygons
        for poly in mask_polys:
            try:
                coords = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
            except Exception:
                continue
            if coords.shape[0] < 3:
                continue
            polygons.append(coords.flatten().tolist())
        return polygons

    def predict(
        self,
        image_bgr: np.ndarray,
        conf: float = 0.25,
        iou: float = 0.7,
        max_det: int = 300,
        classes: Optional[List[int]] = None,
    ) -> ModelResultTD:
        h, w = image_bgr.shape[:2]
        runtime_device = self._resolve_device(self.device)
        runtime_half = self.half if runtime_device != "cpu" else False
        results = self.model.predict(
            image_bgr,
            conf=conf,
            iou=iou,
            max_det=max_det,
            classes=classes,
            device=runtime_device,
            imgsz=self.imgsz,
            half=runtime_half,
            verbose=False,
        )
        polygons: List[List[float]] = []
        if results:
            result = results[0]
            detections = self._extract_detections(result)
            polygons = self._extract_polygons(result)
        else:
            detections = []
        return {
            "model_name": self.model_name,
            "model_version": str(self.model_version),
            "image": {"width": int(w), "height": int(h)},
            "detections": detections,
            "polygons": polygons,
            "meta": {},
        }

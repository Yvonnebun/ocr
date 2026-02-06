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

    def _extract_polygons(self, result, normalized: bool = False) -> List[dict]:
        polygons: List[dict] = []
        masks = getattr(result, "masks", None)
        if masks is None:
            return polygons
        mask_polys = getattr(masks, "xyn" if normalized else "xy", None)
        if not mask_polys:
            return polygons
        boxes = getattr(result, "boxes", None)
        for i, poly in enumerate(mask_polys):
            try:
                coords = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
            except Exception:
                continue
            if coords.shape[0] < 3:
                continue
            class_id = None
            score = None
            if boxes is not None and i < len(boxes):
                box = boxes[i]
                if hasattr(box, "cls"):
                    class_id = int(box.cls[0].item()) if hasattr(box.cls[0], "item") else int(box.cls[0])
                if hasattr(box, "conf"):
                    score = float(box.conf[0].item()) if hasattr(box.conf[0], "item") else float(box.conf[0])
            polygons.append(
                {
                    "detection_index": int(i),
                    "class_id": class_id,
                    "score": score,
                    "points": coords.tolist(),
                }
            )
        return polygons


    def _polygons_to_rings(self, polygons: List[dict]) -> List[dict]:
        rings: List[dict] = []
        for poly in polygons:
            points = poly.get("points") or []
            if len(points) < 3:
                continue
            rings.append(
                {
                    "exterior": np.asarray(points, dtype=np.float32).reshape(-1, 2).flatten().tolist(),
                    "holes": [],
                    "detection_index": poly.get("detection_index"),
                    "class_id": poly.get("class_id"),
                    "score": poly.get("score"),
                }
            )
        return rings


    def predict(
        self,
        image_bgr: np.ndarray,
        conf: float = 0.25,
        iou: float = 0.7,
        max_det: int = 300,
        classes: Optional[List[int]] = None,
        polygon_mode: str = "xy",
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
        polygons: List[dict] = []

        polygons_rings: List[dict] = []

        polygons_mode = "none"
        if results:
            result = results[0]
            detections = self._extract_detections(result)
            if polygon_mode in {"xy", "xyn"}:
                polygons = self._extract_polygons(result, normalized=polygon_mode == "xyn")

                polygons_rings = self._polygons_to_rings(polygons)

                polygons_mode = polygon_mode
        else:
            detections = []
        return {
            "model_name": self.model_name,
            "model_version": str(self.model_version),
            "image": {"width": int(w), "height": int(h)},
            "detections": detections,
            "polygons": polygons,

            "polygons_rings": polygons_rings,

            "meta": {
                "boxes_n": int(len(result.boxes)) if results and getattr(result, "boxes", None) is not None else 0,
                "masks_n": int(len(result.masks.xy)) if results and getattr(result, "masks", None) is not None else 0,
                "polygons_mode": polygons_mode,
            },
        }

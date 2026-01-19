"""
Page-level keyword gate for floorplan detection.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import importlib
import importlib.util
import re

import cv2
import numpy as np

import config
from run_logger import get_run_logger


def _filter_kwargs(callable_obj, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    # codex update: filter unsupported kwargs for PaddleOCR init
    try:
        params = callable_obj.__init__.__code__.co_varnames
        return {k: v for k, v in kwargs.items() if k in params}
    except Exception:
        return kwargs


def _normalize_text(text: str) -> str:
    # codex update: normalize OCR/native text for keyword matching
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _keywords() -> List[str]:
    return [kw.lower() for kw in getattr(config, "PAGE_KEYWORDS", ["floor plan", "floorplan"])]


def native_text_has_kw(native_text: str) -> bool:
    # codex update: check keyword hit in native PDF text
    normalized = _normalize_text(native_text)
    return any(kw in normalized for kw in _keywords())


def init_ocr() -> Optional[Any]:
    # codex update: initialize PaddleOCR if available
    if importlib.util.find_spec("paddleocr") is None:
        return None
    paddle_module = importlib.import_module("paddleocr")
    PaddleOCR = getattr(paddle_module, "PaddleOCR", None)
    if PaddleOCR is None:
        return None
    init_kwargs = getattr(config, "GATE_OCR_PARAMS", {})
    init_kwargs = _filter_kwargs(PaddleOCR, dict(init_kwargs))
    try:
        return PaddleOCR(**init_kwargs)
    except Exception:
        return PaddleOCR(lang=getattr(config, "PAGE_KEYWORD_LANG", "en"))


def _resize_proportional(img_bgr: np.ndarray, limit: int) -> Tuple[np.ndarray, float]:
    h, w = img_bgr.shape[:2]
    max_side = max(h, w)
    if max_side <= limit:
        return img_bgr, 1.0
    ratio = limit / float(max_side)
    new_w, new_h = max(1, int(round(w * ratio))), max(1, int(round(h * ratio)))
    return cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA), ratio


def _preprocess_for_ocr(img_bgr: np.ndarray) -> np.ndarray:
    # codex update: light preprocess to stabilize OCR
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    gray = cv2.addWeighted(gray, 1.6, blur, -0.6, 0)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _ocr_items(page_bgr: np.ndarray, ocr_engine: Any) -> List[Dict[str, Any]]:
    if ocr_engine is None:
        return []
    img_rgb = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2RGB)
    # codex update: track PaddleOCR calls for gate OCR
    run_logger = get_run_logger()
    if run_logger:
        run_logger.increment("paddle_calls")
        run_logger.log_event("paddle_call", {"source": "page_gate"})
    pred = ocr_engine.predict(img_rgb) if hasattr(ocr_engine, "predict") else ocr_engine.ocr(img_rgb)
    if not pred:
        return []
    items: List[Dict[str, Any]] = []

    result = pred[0]
    if isinstance(result, dict) and "rec_texts" in result and "rec_boxes" in result:
        texts = result.get("rec_texts", [])
        boxes = result.get("rec_boxes", [])
        scores = result.get("rec_scores", [])
        count = min(len(texts), len(boxes), len(scores) if scores else len(texts))
        for i in range(count):
            text = _normalize_text(str(texts[i]))
            if not text:
                continue
            box = np.array(boxes[i], dtype=np.float32)
            if box.ndim == 1 and box.size >= 4:
                x0, y0, x1, y1 = map(float, box[:4])
            elif box.ndim == 2 and box.shape[1] >= 2:
                x0, y0 = float(np.min(box[:, 0])), float(np.min(box[:, 1]))
                x1, y1 = float(np.max(box[:, 0])), float(np.max(box[:, 1]))
            else:
                continue
            h = max(1.0, y1 - y0)
            items.append({"text": text, "bbox": [x0, y0, x1, y1], "h": h})
        return items

    if isinstance(pred, list) and pred and isinstance(pred[0], list):
        for item in pred[0]:
            try:
                pts4, tc = item
                text = ""
                if isinstance(tc, (list, tuple)) and tc:
                    text = str(tc[0])
                else:
                    text = str(tc)
                text = _normalize_text(text)
                if not text:
                    continue
                pts = np.array(pts4, dtype=np.float32)
                x0, y0 = float(np.min(pts[:, 0])), float(np.min(pts[:, 1]))
                x1, y1 = float(np.max(pts[:, 0])), float(np.max(pts[:, 1]))
                h = max(1.0, y1 - y0)
                items.append({"text": text, "bbox": [x0, y0, x1, y1], "h": h})
            except Exception:
                continue
    return items


def _kw_match_floor_plan(items: List[Dict[str, Any]]) -> bool:
    if not items:
        return False
    for item in items:
        text = item["text"]
        if "floorplan" in text or "floor plan" in text:
            return True
    return False


def page_has_floorplan_keyword(
    image_path: str,
    native_text_blocks: Optional[List[Dict[str, Any]]],
) -> Tuple[bool, str]:
    # codex update: page-level keyword gate using native text then OCR
    native_text = " ".join(block.get("text", "") for block in (native_text_blocks or []))
    if native_text_has_kw(native_text):
        return True, "native_text"

    if not getattr(config, "GATE_USE_OCR", True):
        return False, "native_text_miss"

    ocr_engine = init_ocr()
    if ocr_engine is None:
        return False, "ocr_unavailable"

    page_bgr = cv2.imread(image_path)
    if page_bgr is None:
        return False, "image_load_failed"

    page_small, _ = _resize_proportional(page_bgr, getattr(config, "GATE_MAX_SIDE", 3840))
    page_prep = _preprocess_for_ocr(page_small)
    items = _ocr_items(page_prep, ocr_engine)
    return _kw_match_floor_plan(items), "ocr"

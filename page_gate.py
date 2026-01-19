"""
Page-level keyword gate for floorplan detection.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import os
import re
import uuid

import cv2
import numpy as np

import config
from ocr_service import paddle_ocr


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


def _resize_proportional(img_bgr: np.ndarray, limit: int) -> Tuple[np.ndarray, float]:
    h, w = img_bgr.shape[:2]
    max_side = max(h, w)
    if max_side <= limit:
        return img_bgr, 1.0
    ratio = limit / float(max_side)
    new_w, new_h = max(1, int(round(w * ratio))), max(1, int(round(h * ratio)))
    return cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA), ratio


def _upscale_to_limit(
    img_bgr: np.ndarray,
    max_side: int,
    min_scale: float,
    max_scale: float,
) -> Tuple[np.ndarray, float]:
    h, w = img_bgr.shape[:2]
    scale_cap = max_side / float(max(h, w) + 1e-6)
    scale = min(max_scale, max(min_scale, scale_cap))
    if scale <= 1.01:
        return img_bgr, 1.0
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    out = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return out, float(scale)


def _preprocess_for_kw_ocr(img_bgr: np.ndarray, sharpen: bool = True) -> np.ndarray:
    # codex update: align keyword OCR preprocessing with floorplan script
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    if sharpen:
        blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
        gray = cv2.addWeighted(gray, 1.6, blur, -0.6, 0)

    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _prepare_gate_image(image_path: str) -> Optional[str]:
    # codex update: preprocess and save gate image for paddle-service
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return None

    h, w = img_bgr.shape[:2]
    strip_ratio = float(getattr(config, "KW_STRIP_Y0_RATIO", 0.80))
    if 0 < strip_ratio < 1:
        y0 = int(round(h * strip_ratio))
        if y0 < h:
            img_bgr = img_bgr[y0:h, :, :]

    img_bgr, _ = _resize_proportional(img_bgr, limit=int(getattr(config, "KW_OCR_MAX_SIDE", 3840)))
    img_bgr, _ = _upscale_to_limit(
        img_bgr,
        max_side=int(getattr(config, "KW_OCR_MAX_SIDE", 3840)),
        min_scale=float(getattr(config, "KW_OCR_MIN_UPSCALE", 2.0)),
        max_scale=float(getattr(config, "KW_OCR_MAX_UPSCALE", 4.0)),
    )
    img_bgr = _preprocess_for_kw_ocr(img_bgr, sharpen=True)

    out_dir = getattr(config, "GATE_PREPROCESS_DIR", os.path.join("output", "gate_preprocess"))
    os.makedirs(out_dir, exist_ok=True)
    out_name = f"gate_{uuid.uuid4().hex}.png"
    out_path = os.path.join(out_dir, out_name)
    cv2.imwrite(out_path, img_bgr)
    return out_path


def _ocr_items_from_service(image_path: str) -> List[Dict[str, float]]:
    # codex update: normalize paddle-service output into text items
    blocks = paddle_ocr(image_path)
    items: List[Dict[str, float]] = []
    for block in blocks:
        text = _normalize_text(block.get("text", ""))
        if not text:
            continue
        bbox = block.get("bbox", [])
        if not bbox or len(bbox) < 4:
            continue
        x0, y0, x1, y1 = map(float, bbox[:4])
        h = max(1.0, y1 - y0)
        cx, cy = (x0 + x1) * 0.5, (y0 + y1) * 0.5
        items.append({"text": text, "bbox": [x0, y0, x1, y1], "h": h, "cx": cx, "cy": cy})
    return items


def _kw_match_floor_plan(
    items: List[Dict[str, Any]],
    height_rel_tol: float = 0.25,
    line_y_tol: float = 0.6,
    gap_char_factor: float = 1.2,
) -> bool:
    # codex update: strict phrase matching aligned with floorplan script
    if not items:
        return False

    for item in items:
        text = item["text"]
        if "floorplan" in text:
            return True
        if "floor plan" in text:
            return True

    floors = [item for item in items if re.search(r"\bfloor\b", item["text"])]
    plans = [item for item in items if re.search(r"\bplan\b", item["text"])]

    if not floors or not plans:
        return False

    for floor_item in floors:
        fx0, _, fx1, _ = floor_item["bbox"]
        fh, fcy = floor_item["h"], floor_item["cy"]
        for plan_item in plans:
            px0, _, _, _ = plan_item["bbox"]
            ph, pcy = plan_item["h"], plan_item["cy"]

            mh = max(fh, ph)
            if abs(fh - ph) / (mh + 1e-6) > height_rel_tol:
                continue
            if abs(fcy - pcy) > line_y_tol * mh:
                continue
            if px0 < fx0:
                continue

            gap = px0 - fx1
            if gap <= gap_char_factor * mh:
                return True
    return False


def page_has_floorplan_keyword(
    image_path: str,
    native_text_blocks: Optional[List[Dict[str, Any]]],
) -> Tuple[bool, bool, str]:
    # codex update: page-level keyword gate using native text then OCR
    native_text = " ".join(block.get("text", "") for block in (native_text_blocks or []))
    if native_text_has_kw(native_text):
        return True, False, "native_text"

    if not getattr(config, "GATE_USE_OCR", True):
        return False, False, "native_text_miss"

    preprocessed_path = _prepare_gate_image(image_path)
    if preprocessed_path is None:
        return False, False, "image_load_failed"
    items = _ocr_items_from_service(preprocessed_path)
    if not getattr(config, "GATE_KEEP_PREPROCESS", False):
        try:
            os.remove(preprocessed_path)
        except OSError:
            pass
    hit = _kw_match_floor_plan(items)
    return hit, hit, "ocr"

"""
Page-level keyword gate for floorplan detection.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import re

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

    items = _ocr_items_from_service(image_path)
    hit = _kw_match_floor_plan(items)
    return hit, hit, "ocr"

"""
Page-level floorplan gate (v2-style):
- Step 1: native PDF text keyword hit (cheap) -> keep page, force_keep=False
- Step 2: OCR on FULL PAGE (no strip) using paddle_ocr service -> strict phrase match -> keep page, force_keep=True
- Step 3: Visual blueprint-like prefilter (edge density + color ratio) applied as a *secondary* filter
          - bypassed when force_keep=True (mirrors "force_keep bypass blueprint filter" behavior)
Returns:
    (keep: bool, force_keep: bool, reason: str)

Notes:
- This module does NOT do layout-based candidate crops (PrimaLayout). Instead it applies the blueprint visual
  filter at page-level, conservatively. This keeps the "spirit" of v2 while fitting the page-gate interface.
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


# ----------------------------
# Text normalization & keywords
# ----------------------------

def _normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _keywords() -> List[str]:
    # v2-style default: keep strict to avoid false positives
    return [kw.lower() for kw in getattr(config, "PAGE_KEYWORDS", ["floor plan", "floorplan"])]


def native_text_has_kw(native_text: str) -> bool:
    normalized = _normalize_text(native_text)
    # same behavior as v2: strict floorplan/floor plan
    return ("floorplan" in normalized) or ("floor plan" in normalized) or any(kw in normalized for kw in _keywords())


# ----------------------------
# Resize / preprocess helpers
# ----------------------------

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
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    if sharpen:
        blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
        gray = cv2.addWeighted(gray, 1.6, blur, -0.6, 0)

    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _prepare_gate_image_fullpage(image_path: str) -> Optional[str]:
    """
    v2-style: OCR on FULL PAGE (no strip).
    Writes a preprocessed temp image for paddle-service OCR.
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return None

    # Resize + controlled upscale (mirrors your earlier behavior, but without strip)
    max_side = int(getattr(config, "KW_OCR_MAX_SIDE", 3840))
    img_bgr, _ = _resize_proportional(img_bgr, limit=max_side)
    img_bgr, _ = _upscale_to_limit(
        img_bgr,
        max_side=max_side,
        min_scale=float(getattr(config, "KW_OCR_MIN_UPSCALE", 1.0)),
        max_scale=float(getattr(config, "KW_OCR_MAX_UPSCALE", 4.0)),
    )
    img_bgr = _preprocess_for_kw_ocr(img_bgr, sharpen=True)

    out_dir = getattr(config, "GATE_PREPROCESS_DIR", os.path.join("output", "gate_preprocess"))
    os.makedirs(out_dir, exist_ok=True)
    out_name = f"gate_full_{uuid.uuid4().hex}.png"
    out_path = os.path.join(out_dir, out_name)
    cv2.imwrite(out_path, img_bgr)
    return out_path


# ----------------------------
# OCR item normalization
# ----------------------------

def _ocr_items_from_service(image_path: str) -> List[Dict[str, float]]:
    """
    Normalize paddle-service output into items:
      {"text": str, "bbox":[x0,y0,x1,y1], "h":float, "cx":float, "cy":float}
    """
    blocks = paddle_ocr(image_path)
    items: List[Dict[str, float]] = []
    for block in blocks or []:
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
    """
    Strict phrase match (v2):
    - strong hit if any single item contains "floorplan" or "floor plan"
    - else: require "floor" and "plan" in nearby boxes (same line-ish, similar height, small gap)
    """
    if not items:
        return False

    for item in items:
        t = item["text"]
        if "floorplan" in t:
            return True
        if "floor plan" in t:
            return True

    floors = [it for it in items if re.search(r"\bfloor\b", it["text"])]
    plans = [it for it in items if re.search(r"\bplan\b", it["text"])]
    if not floors or not plans:
        return False

    for f in floors:
        fx0, _, fx1, _ = f["bbox"]
        fh, fcy = f["h"], f["cy"]
        for p in plans:
            px0, _, _, _ = p["bbox"]
            ph, pcy = p["h"], p["cy"]

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


# ----------------------------
# Visual blueprint-like prefilter (page-level)
# ----------------------------

def _edge_density(img_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return float(np.mean(edges > 0))


def _color_ratio(img_bgr: np.ndarray) -> float:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)
    mask = (s > 25) & (v > 25)
    denom = float(img_bgr.shape[0] * img_bgr.shape[1] + 1e-6)
    return float(np.count_nonzero(mask) / denom)


def _is_blueprint_like_page(img_bgr: np.ndarray) -> Tuple[bool, Dict[str, float]]:
    """
    Conservative page-level blueprint filter to approximate v2 crop filter.
    Bypassed by force_keep=True.

    Thresholds (configurable):
      - GATE_VIS_EDGE_DENSITY_MIN (default 0.010)
      - GATE_VIS_COLOR_MAX (default 0.30)
      - GATE_VIS_MAX_SIDE (default 2000) for feature computation downscale
    """
    max_side = int(getattr(config, "GATE_VIS_MAX_SIDE", 2000))
    small, _ = _resize_proportional(img_bgr, limit=max_side)

    ed = _edge_density(small)
    cr = _color_ratio(small)

    feats = {"page_edge_density": float(ed), "page_color_ratio": float(cr)}

    ed_min = float(getattr(config, "GATE_VIS_EDGE_DENSITY_MIN", 0.010))
    cr_max = float(getattr(config, "GATE_VIS_COLOR_MAX", 0.30))

    if cr > cr_max:
        return False, feats
    if ed < ed_min:
        return False, feats
    return True, feats


# ----------------------------
# Public API
# ----------------------------

def page_has_floorplan_keyword(
    image_path: str,
    native_text_blocks: Optional[List[Dict[str, Any]]],
) -> Tuple[bool, bool, str]:
    """
    Returns:
      keep: bool         - should keep this page for downstream heavy inference
      force_keep: bool   - True only when STRICT OCR phrase match succeeds (v2 behavior)
      reason: str        - one of:
          "native_text"
          "ocr_force_keep"
          "ocr_miss"
          "visual_reject"
          "image_load_failed"
          "native_text_miss_ocr_disabled"
          "ocr_error"
    """
    # ---- Step 1: native text gate ----
    native_text = " ".join((b.get("text", "") for b in (native_text_blocks or [])))
    if native_text_has_kw(native_text):
        keep = True
        force_keep = False  # v2: native hit does NOT imply force_keep
        reason = "native_text"

        # Optional: apply visual prefilter even on native hit (bypass only if force_keep)
        if getattr(config, "GATE_USE_VISUAL_PREFILTER", True):
            img_bgr = cv2.imread(image_path)
            if img_bgr is None:
                return True, False, "native_text"  # don't block if image unavailable
            ok_vis, _ = _is_blueprint_like_page(img_bgr)
            if not ok_vis:
                if getattr(config, "GATE_VISUAL_HARD_REJECT", True):
                    return False, False, "visual_reject"
        return keep, force_keep, reason

    # ---- Step 2: OCR gate (full page, no strip) ----
    if not getattr(config, "GATE_USE_OCR", True):
        return False, False, "native_text_miss_ocr_disabled"

    prep_path = _prepare_gate_image_fullpage(image_path)
    if prep_path is None:
        return False, False, "image_load_failed"

    try:
        items = _ocr_items_from_service(prep_path)
    except Exception:
        items = []
        hit = False
        force_keep = False
        reason = "ocr_error"
    else:
        hit = _kw_match_floor_plan(items)
        force_keep = bool(hit)
        reason = "ocr_force_keep" if force_keep else "ocr_miss"

    if not getattr(config, "GATE_KEEP_PREPROCESS", False):
        try:
            os.remove(prep_path)
        except OSError:
            pass

    if not hit:
        return False, False, reason

    # ---- Step 3: Visual blueprint prefilter (bypassed if force_keep=True) ----
    if getattr(config, "GATE_USE_VISUAL_PREFILTER", True) and not force_keep:
        img_bgr = cv2.imread(image_path)
        if img_bgr is not None:
            ok_vis, _ = _is_blueprint_like_page(img_bgr)
            if not ok_vis and getattr(config, "GATE_VISUAL_HARD_REJECT", True):
                return False, False, "visual_reject"

    return True, force_keep, reason

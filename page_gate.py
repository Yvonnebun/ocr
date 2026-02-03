"""
Page-level floorplan gate (v2-aligned, with visual fallback):

Desired behavior:
1) If KW hit (native text OR OCR strict phrase)  -> keep=True, force_keep=True
   - Meaning: downstream should "force keep candidates" (bypass visual reject).
2) If NO KW hit -> do NOT immediately drop.
   - Run visual blueprint-like logic at page-level:
       - if blueprint-like -> keep=True, force_keep=False
       - else -> keep=False

Returns: (keep: bool, force_keep: bool, reason: str)

Notes:
- OCR is FULL PAGE (no strip), closer to your v2 script.
- Visual filter is meaningful because it is used as fallback when kw misses.
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
    # keep strict by default; can override via config.PAGE_KEYWORDS
    return [kw.lower() for kw in getattr(config, "PAGE_KEYWORDS", ["floor plan", "floorplan"])]


def native_text_has_kw(native_text: str) -> bool:
    t = _normalize_text(native_text)
    # strict baseline, plus configurable list
    if ("floorplan" in t) or ("floor plan" in t):
        return True
    return any(kw in t for kw in _keywords())


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
    """FULL PAGE OCR (no strip). Writes a temp preprocessed image."""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return None

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
    out_path = os.path.join(out_dir, f"gate_full_{uuid.uuid4().hex}.png")
    cv2.imwrite(out_path, img_bgr)
    return out_path


# ----------------------------
# OCR item normalization + strict phrase match
# ----------------------------
def _ocr_items_from_service(image_path: str) -> List[Dict[str, Any]]:
    blocks = paddle_ocr(image_path)
    items: List[Dict[str, Any]] = []
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
    """Strict: floorplan / floor plan / (floor + plan nearby)."""
    if not items:
        return False

    for it in items:
        t = it["text"]
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
# Visual blueprint-like page filter (fallback path)
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


def _visual_blueprint_like(img_bgr: np.ndarray) -> Tuple[bool, Dict[str, float]]:
    """
    Conservative blueprint-like filter.
    Used ONLY when KW misses (fallback).
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
    New behavior:
      - KW hit (native or OCR strict) => keep=True, force_keep=True
      - KW miss => visual fallback:
          visual pass => keep=True, force_keep=False
          visual fail => keep=False, force_keep=False
    """
    # Always load image once (needed for OCR + visual fallback)
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return False, False, "image_load_failed"

    # ---- Step 1: native text KW ----
    native_text = " ".join((b.get("text", "") for b in (native_text_blocks or [])))
    if native_text_has_kw(native_text):
        # You requested: "命中kw就强制保留cands"
        return True, True, "kw_native"

    # ---- Step 2: OCR strict KW (full page) ----
    ocr_hit = False
    if getattr(config, "GATE_USE_OCR", True):
        prep_path = _prepare_gate_image_fullpage(image_path)
        if prep_path is not None:
            try:
                items = _ocr_items_from_service(prep_path)
                ocr_hit = _kw_match_floor_plan(items)
            finally:
                if not getattr(config, "GATE_KEEP_PREPROCESS", False):
                    try:
                        os.remove(prep_path)
                    except OSError:
                        pass

    if ocr_hit:
        # You requested: "命中kw就强制保留cands"
        return True, True, "kw_ocr_strict"

    # ---- Step 3: Visual fallback (meaningful even when KW misses) ----
    if not getattr(config, "GATE_USE_VISUAL_FALLBACK", True):
        return False, False, "kw_miss_visual_disabled"

    ok_vis, feats = _visual_blueprint_like(img_bgr)
    if ok_vis:
        return True, False, f"visual_keep(ed={feats['page_edge_density']:.4f},cr={feats['page_color_ratio']:.3f})"
    return False, False, f"visual_reject(ed={feats['page_edge_density']:.4f},cr={feats['page_color_ratio']:.3f})"

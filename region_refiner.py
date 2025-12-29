"""
region_refiner_paddleocr_safe.py

Safe Region Refiner for BLUEPRINT pipeline.

Key features:
- PaddleOCR 2.x / 3.x compatible (auto-detect supported kwargs)
- No show_log arg (3.x removed it); config via paddleocr.logger if available
- Strong guards against huge/strip crops to avoid native crashes
- Preprocess with optional resize + CLAHE + adaptive threshold (+ optional invert)
- Saves ALL candidate crops per page when DEBUG_REFINE_DIR is set
- Optional: inject full-page candidate to ensure blueprint region never missed
"""

from __future__ import annotations

import os
import re
import inspect
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

import config
import utils

# PaddleOCR import (3.x uses centralized logger)
from paddleocr import PaddleOCR  # type: ignore


_ocr: Optional[PaddleOCR] = None


# -----------------------------
# Helpers
# -----------------------------
def _safe_name(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s)
    return s[:160] if len(s) > 160 else s


def _mkdir(p: str) -> None:
    if p:
        os.makedirs(p, exist_ok=True)


def _clip_bbox(bbox: List[float], w: int, h: int) -> Optional[List[int]]:
    try:
        x0, y0, x1, y1 = [int(float(c)) for c in bbox]
    except Exception:
        return None
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(w, x1)
    y1 = min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def _save_crop(page_img: Image.Image, bbox_xyxy: List[int], out_path: str) -> None:
    try:
        x0, y0, x1, y1 = bbox_xyxy
        page_img.crop((x0, y0, x1, y1)).save(out_path)
    except Exception:
        pass


def _filter_kwargs(callable_obj, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only kwargs supported by callable_obj signature."""
    try:
        sig = inspect.signature(callable_obj)
        supported = set(sig.parameters.keys())
        return {k: v for k, v in kwargs.items() if k in supported}
    except Exception:
        # If signature introspection fails, return as-is (best effort).
        return kwargs


def _configure_paddleocr_logging() -> None:
    """
    PaddleOCR 3.x uses paddleocr.logger (Python logging). show_log is removed in 3.x.
    We set level to ERROR to reduce spam if logger exists.
    """
    try:
        from paddleocr import logger  # type: ignore
        import logging

        logger.setLevel(logging.ERROR)
        logger.propagate = False
    except Exception:
        pass


# -----------------------------
# OCR initialization
# -----------------------------
def get_ocr() -> PaddleOCR:
    global _ocr
    if _ocr is None:
        _configure_paddleocr_logging()

        init_kwargs = {
            "lang": getattr(config, "OCR_LANG", "en"),
            "use_angle_cls": getattr(config, "OCR_USE_ANGLE_CLS", True),
            # NOTE: do NOT pass show_log (removed in PaddleOCR 3.x)
        }
        init_kwargs = _filter_kwargs(PaddleOCR.__init__, init_kwargs)

        _ocr = PaddleOCR(**init_kwargs)  # type: ignore
    return _ocr


# -----------------------------
# Preprocess
# -----------------------------
def preprocess_for_ocr(region_pil: Image.Image) -> np.ndarray:
    """
    Returns BGR uint8 image for PaddleOCR.
    Includes safety resize to avoid huge inputs.
    """
    import cv2

    rgb = np.array(region_pil.convert("RGB"))
    h, w = rgb.shape[:2]

    # Safety resize for preprocessing (separate from region guard)
    max_pre_side = getattr(config, "OCR_PREPROC_MAX_SIDE", 2000)
    if max_pre_side and max(h, w) > max_pre_side:
        scale = max(h, w) / float(max_pre_side)
        new_w = max(1, int(w / scale))
        new_h = max(1, int(h / scale))
        rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # CLAHE
    if getattr(config, "OCR_USE_CLAHE", True):
        clahe = cv2.createCLAHE(
            clipLimit=getattr(config, "OCR_CLAHE_CLIP", 2.0),
            tileGridSize=getattr(config, "OCR_CLAHE_GRID", (8, 8)),
        )
        gray = clahe.apply(gray)

    # Adaptive threshold
    if getattr(config, "OCR_USE_ADAPTIVE_THRESHOLD", True):
        block_size = int(getattr(config, "OCR_ADAPTIVE_BLOCK_SIZE", 31))
        if block_size < 3:
            block_size = 3
        if block_size % 2 == 0:
            block_size += 1
        C = int(getattr(config, "OCR_ADAPTIVE_C", 10))

        bw = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size, C
        )
    else:
        bw = gray

    # Optional invert
    if getattr(config, "OCR_AUTO_INVERT", True):
        white_ratio = float((bw > 0).mean())
        thr = float(getattr(config, "OCR_INVERT_WHITE_RATIO_THRESHOLD", 0.5))
        if white_ratio < thr:
            bw = 255 - bw

    if bw.ndim == 2:
        bgr = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
    else:
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    return bgr


# -----------------------------
# Core refine
# -----------------------------
def refine_region(page_img: Image.Image, candidate_bbox: List[float]) -> Dict[str, Any]:
    page_w, page_h = page_img.size
    clipped = _clip_bbox(candidate_bbox, page_w, page_h)
    if clipped is None:
        return {"type": "uncertain", "bbox_px": candidate_bbox, "confidence": 0.0}

    cx0, cy0, cx1, cy1 = clipped
    cw, ch = (cx1 - cx0), (cy1 - cy0)

    area = float(cw * ch)
    page_area = max(float(page_w * page_h), 1.0)
    area_ratio = area / page_area

    # --- Blueprint-friendly guard 1: big regions are image-like
    big_ratio = float(getattr(config, "BIG_REGION_AREA_RATIO", 0.25))
    if area_ratio >= big_ratio:
        return {"type": "image", "bbox_px": candidate_bbox, "confidence": 0.90}

    # --- Blueprint-friendly guard 2: huge side length -> skip OCR (prevents native crash)
    ocr_max_side = int(getattr(config, "OCR_MAX_SIDE_PX", 3000))
    if ocr_max_side and max(cw, ch) > ocr_max_side:
        return {"type": "image", "bbox_px": candidate_bbox, "confidence": 0.85}

    # --- Blueprint-friendly guard 3: strip-like regions -> skip OCR
    strip_ar = float(getattr(config, "OCR_STRIP_ASPECT_RATIO", 6.0))
    ar = cw / max(ch, 1)
    if strip_ar and (ar > strip_ar or (1.0 / max(ar, 1e-6)) > strip_ar):
        return {"type": "image", "bbox_px": candidate_bbox, "confidence": 0.85}

    # --- Optional tiny short-circuit
    min_side = int(getattr(config, "MIN_REGION_SIDE_PX", 0))
    if min_side and (cw < min_side or ch < min_side):
        return {"type": "image", "bbox_px": candidate_bbox, "confidence": 0.60}

    region_img = page_img.crop((cx0, cy0, cx1, cy1))

    try:
        ocr = get_ocr()
        region_bgr = preprocess_for_ocr(region_img)

        # PaddleOCR 2.x: ocr.ocr(img, cls=True, det=True, rec=True) common
        # PaddleOCR 3.x: signature may differ; auto-filter kwargs.
        call_kwargs = {
            "cls": bool(getattr(config, "OCR_CALL_CLS", True)),
            "det": bool(getattr(config, "OCR_CALL_DET", True)),
            "rec": bool(getattr(config, "OCR_CALL_REC", True)),
        }
        call_kwargs = _filter_kwargs(ocr.ocr, call_kwargs)

        ocr_raw = ocr.ocr(region_bgr, **call_kwargs)  # type: ignore

    except Exception as e:
        print(f"[refine_region] OCR error: {e}")
        return {"type": "uncertain", "bbox_px": candidate_bbox, "confidence": 0.0}

    # Normalize outputs:
    # - PaddleOCR 2.x often returns [ [ [box, (text, conf)], ... ] ]
    # - PaddleOCR 3.x may return list of result objects; but ocr.ocr still often returns nested list.
    # We handle the common nested-list path; if unknown, treat as "image".
    try:
        if not ocr_raw:
            return {"type": "image", "bbox_px": candidate_bbox, "confidence": 0.80}

        # If it’s list and first element is list => assume 2.x style
        if isinstance(ocr_raw, list) and len(ocr_raw) > 0 and isinstance(ocr_raw[0], list):
            lines = ocr_raw[0]
        else:
            # Unknown structure (3.x result objects etc.)
            # For region-refiner (text-vs-image), safest fallback:
            return {"type": "image", "bbox_px": candidate_bbox, "confidence": 0.80}

    except Exception:
        return {"type": "image", "bbox_px": candidate_bbox, "confidence": 0.80}

    if not lines:
        return {"type": "image", "bbox_px": candidate_bbox, "confidence": 0.80}

    ocr_results: List[Dict[str, Any]] = []
    for item in lines:
        try:
            pts4, (text, score) = item
            text = (text or "").strip()
            if not text:
                continue

            xs = [p[0] for p in pts4]
            ys = [p[1] for p in pts4]
            bx0, by0, bx1, by1 = float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))

            # region coords -> page coords
            bx0 += cx0
            bx1 += cx0
            by0 += cy0
            by1 += cy0

            ocr_results.append({
                "text": text,
                "bbox": [bx0, by0, bx1, by1],
                "conf": float(score) * 100.0,
            })
        except Exception:
            continue

    if not ocr_results:
        return {"type": "image", "bbox_px": candidate_bbox, "confidence": 0.80}

    total_chars = sum(len(r["text"]) for r in ocr_results)
    num_lines = utils.cluster_lines(ocr_results, tolerance=config.LINE_CLUSTERING_TOLERANCE)
    alignment_score = utils.calculate_text_alignment_score(ocr_results)

    min_chars = int(getattr(config, "MIN_TEXT_CHARS", 30))
    min_lines = int(getattr(config, "MIN_TEXT_LINES", 2))
    align_th = float(getattr(config, "TEXT_ALIGNMENT_THRESHOLD", 0.5))

    is_text_block = (total_chars >= min_chars) and (num_lines >= min_lines) and (alignment_score >= align_th)

    if is_text_block:
        return {"type": "text", "bbox_px": candidate_bbox, "confidence": float(alignment_score)}
    return {"type": "image", "bbox_px": candidate_bbox, "confidence": float(1.0 - alignment_score)}


def refine_all_candidates(image_path: str, candidates: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    page_img = Image.open(image_path).convert("RGB")
    page_w, page_h = page_img.size

    # Optional: inject full-page candidate
    if bool(getattr(config, "INJECT_FULL_PAGE_CANDIDATE", False)):
        candidates = [{"bbox_px": [0, 0, page_w, page_h], "source": "injected_full_page"}] + list(candidates)

    debug_root = getattr(config, "DEBUG_REFINE_DIR", "")
    page_debug_dir = ""
    if debug_root:
        base = _safe_name(os.path.splitext(os.path.basename(image_path))[0])
        page_debug_dir = os.path.join(debug_root, base)
        _mkdir(page_debug_dir)
        page_img.save(os.path.join(page_debug_dir, "page.png"))

    image_regions: List[Dict[str, Any]] = []
    text_regions: List[Dict[str, Any]] = []
    uncertain_regions: List[Dict[str, Any]] = []

    for idx, candidate in enumerate(candidates):
        bbox = candidate.get("bbox_px", None)
        if not bbox or len(bbox) != 4:
            continue

        clipped = _clip_bbox(bbox, page_w, page_h)
        if clipped is None:
            continue

        # Debug: save ALL candidate crops
        if page_debug_dir:
            x0, y0, x1, y1 = clipped
            crop_path = os.path.join(page_debug_dir, f"{idx:03d}_{x0}_{y0}_{x1}_{y1}.png")
            _save_crop(page_img, clipped, crop_path)

        result = refine_region(page_img, bbox)
        t = result.get("type", "uncertain")

        payload = {
            "bbox_px": result["bbox_px"],
            "confidence": result.get("confidence", 0.0),
            "source": candidate,
        }

        if t == "image":
            image_regions.append(payload)
        elif t == "text":
            text_regions.append(payload)
        else:
            uncertain_regions.append(payload)

    return {
        "image_regions_final": image_regions,
        "text_regions_override": text_regions,
        "uncertain": uncertain_regions,
    }

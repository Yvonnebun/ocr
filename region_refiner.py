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
from typing import Any, Dict, List

from PIL import Image
import uuid

import config
import utils
from ocr_service import paddle_ocr
from run_logger import get_run_logger


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


# codex update: paddle-service handles OCR; no local PaddleOCR initialization


# -----------------------------
# Preprocess
# -----------------------------
def preprocess_for_ocr(region_pil: Image.Image) -> Image.Image:
    """
    Returns a PIL image for paddle-service OCR.
    """
    # codex update: no local OCR preprocessing required
    return region_pil.convert("RGB")


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
        region_rgb = preprocess_for_ocr(region_img)
        crop_dir = getattr(config, "PADDLE_CROP_DIR", "output/paddle_crops")
        os.makedirs(crop_dir, exist_ok=True)
        crop_name = f"crop_{uuid.uuid4().hex}.png"
        crop_path = os.path.join(crop_dir, crop_name)
        region_rgb.save(crop_path)

        # codex update: use paddle-service OCR on crop
        ocr_blocks = paddle_ocr(crop_path)

        if not getattr(config, "PADDLE_KEEP_CROPS", False):
            try:
                os.remove(crop_path)
            except OSError:
                pass

    except Exception as e:
        print(f"[refine_region] OCR error: {e}")
        return {"type": "uncertain", "bbox_px": candidate_bbox, "confidence": 0.0}

    # Normalize outputs:
    # - PaddleOCR 2.x often returns [ [ [box, (text, conf)], ... ] ]
    # - PaddleOCR 3.x may return list of result objects; but ocr.ocr still often returns nested list.
    # We handle the common nested-list path; if unknown, treat as "image".
    if not ocr_blocks:
        return {"type": "image", "bbox_px": candidate_bbox, "confidence": 0.80}

    ocr_results: List[Dict[str, Any]] = []
    for item in ocr_blocks:
        try:
            text = (item.get("text") or "").strip()
            if not text:
                continue
            bbox = item.get("bbox", [])
            if not bbox or len(bbox) < 4:
                continue
            bx0, by0, bx1, by1 = map(float, bbox[:4])
            # crop coords -> page coords
            bx0 += cx0
            bx1 += cx0
            by0 += cy0
            by1 += cy0

            ocr_results.append({
                "text": text,
                "bbox": [bx0, by0, bx1, by1],
                "conf": float(item.get("score") or 0.0) * 100.0,
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

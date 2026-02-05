"""
Step 5: Image Asset Extraction - Crop and deduplicate images
"""
import os
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image

import config
import utils
from run_logger import get_run_logger


def _resize_proportional(img_bgr: np.ndarray, limit: int) -> Tuple[np.ndarray, float]:
    h, w = img_bgr.shape[:2]
    max_side = max(h, w)
    if max_side <= limit:
        return img_bgr, 1.0
    ratio = limit / float(max_side)
    new_w, new_h = max(1, int(round(w * ratio))), max(1, int(round(h * ratio)))
    return cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA), ratio


def _edge_density(img_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return float(np.mean(edges > 0))


def _color_ratio(img_bgr: np.ndarray) -> float:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)
    mask = (s > 25) & (v > 25)
    return float(np.count_nonzero(mask) / (img_bgr.shape[0] * img_bgr.shape[1] + 1e-6))


def _is_blueprint_like_crop(crop_bgr: np.ndarray, page_area: float) -> Tuple[bool, Dict[str, float]]:
    # codex update: blueprint visual prefilter
    if not getattr(config, "BLUEPRINT_FILTER_ENABLED", True):
        return True, {"filter_disabled": 1.0}

    h, w = crop_bgr.shape[:2]
    area_ratio = float((h * w) / (page_area + 1e-6))

    min_area_ratio = float(getattr(config, "BLUEPRINT_MIN_AREA_RATIO", 0.35))
    if area_ratio < min_area_ratio:
        return False, {"area_ratio": area_ratio, "edge_density": 0.0, "color_ratio": 0.0}

    crop_small, _ = _resize_proportional(crop_bgr, limit=3900)
    edge_density = _edge_density(crop_small)
    color_ratio = _color_ratio(crop_small)

    if edge_density < float(getattr(config, "BLUEPRINT_EDGE_DENSITY_MIN", 0.010)):
        return False, {"area_ratio": area_ratio, "edge_density": edge_density, "color_ratio": color_ratio}
    if color_ratio > float(getattr(config, "BLUEPRINT_COLOR_MAX", 0.30)):
        return False, {"area_ratio": area_ratio, "edge_density": edge_density, "color_ratio": color_ratio}
    return True, {"area_ratio": area_ratio, "edge_density": edge_density, "color_ratio": color_ratio}


def extract_image_assets(
    image_path: str,
    image_regions: List[Dict],
    output_dir: str,
    page_idx: int,
    force_keep: bool = False,
) -> List[Dict]:
    """
    Extract image assets by cropping regions from page image.
    
    Args:
        image_path: Path to full page image
        image_regions: List of dicts with 'bbox_px'
        output_dir: Directory to save extracted images
        page_idx: Page index for naming
    
    Returns:
        List of dicts with:
        {
            'image_path': str,
            'bbox_px': [x0, y0, x1, y1]
        }
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if not image_regions:
        return []
    
    # Deduplicate regions on same page
    bboxes = [region['bbox_px'] for region in image_regions]
    keep_indices = utils.deduplicate_bboxes(bboxes, config.IOU_THRESHOLD)
    
    # Load page image
    img = Image.open(image_path)
    width, height = img.size
    
    extracted_images = []
    skipped_blueprint = 0
    run_logger = get_run_logger()
    page_area = float(width * height)
    for idx, region_idx in enumerate(keep_indices):
        region = image_regions[region_idx]
        bbox = region['bbox_px']
        
        # Crop image
        x0, y0, x1, y1 = [int(coord) for coord in bbox]
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(width, x1)
        y1 = min(height, y1)
        
        if x1 <= x0 or y1 <= y0:
            continue
        
        cropped = img.crop((x0, y0, x1, y1))
        crop_rgb = np.array(cropped)
        crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
        if not force_keep:
            ok_bp, feats = _is_blueprint_like_crop(crop_bgr, page_area)
            if not ok_bp:
                # codex update: skip non-blueprint crops, log for diagnostics
                skipped_blueprint += 1
                if run_logger:
                    run_logger.log_event(
                        "blueprint_skip",
                        {
                            "bbox_px": bbox,
                            "page_idx": page_idx,
                            "area_ratio": feats.get("area_ratio", 0.0),
                            "edge_density": feats.get("edge_density", 0.0),
                            "color_ratio": feats.get("color_ratio", 0.0),
                        },
                    )
                continue
        
        # Save cropped image
        image_filename = f"page_{page_idx:04d}_image_{idx:04d}.png"
        image_path_out = os.path.join(output_dir, image_filename)
        cropped.save(image_path_out, "PNG")
        
        extracted_images.append({
            'image_path': image_path_out,
            'bbox_px': bbox,
            'image_size': [cropped.size[0], cropped.size[1]],
        })
    
    if run_logger:
        run_logger.log_event(
            "blueprint_summary",
            {
                "page_idx": page_idx,
                "kept": len(extracted_images),
                "skipped_blueprint": skipped_blueprint,
            },
        )
    return extracted_images


def extend_image_assets(
    image_path: str,
    extracted_images: List[Dict],
    output_dir: str,
    page_idx: int,
    pad_ratio: float = 0.2,
) -> List[Dict]:
    """
    Extend extracted image crops by a padding ratio and re-crop from the page image.

    Args:
        image_path: Path to full page image
        extracted_images: List of dicts with 'bbox_px'
        output_dir: Directory to save extended images
        page_idx: Page index for naming
        pad_ratio: Padding ratio based on crop width/height

    Returns:
        List of dicts with:
        {
            'image_path': str,
            'bbox_px': [x0, y0, x1, y1],
            'extended_bbox_px': [x0, y0, x1, y1],
            'source_index': int,
            'source_image_path': str,
        }
    """
    os.makedirs(output_dir, exist_ok=True)

    if not extracted_images:
        return []

    img = Image.open(image_path)
    width, height = img.size

    extended_images = []
    for idx, img_info in enumerate(extracted_images):
        bbox = img_info.get("bbox_px") or []
        if len(bbox) < 4:
            continue
        x0, y0, x1, y1 = [int(coord) for coord in bbox]
        crop_w = max(0, x1 - x0)
        crop_h = max(0, y1 - y0)
        if crop_w <= 0 or crop_h <= 0:
            continue

        pad_x = int(round(pad_ratio * crop_w))
        pad_y = int(round(pad_ratio * crop_h))
        x0e = max(0, x0 - pad_x)
        y0e = max(0, y0 - pad_y)
        x1e = min(width, x1 + pad_x)
        y1e = min(height, y1 + pad_y)
        if x1e <= x0e or y1e <= y0e:
            continue

        cropped = img.crop((x0e, y0e, x1e, y1e))
        image_filename = f"page_{page_idx:04d}_image_{idx:04d}_extended.png"
        image_path_out = os.path.join(output_dir, image_filename)
        cropped.save(image_path_out, "PNG")

        extended_images.append(
            {
                "image_path": image_path_out,
                "bbox_px": bbox,
                "extended_bbox_px": [x0e, y0e, x1e, y1e],
                "image_size": [cropped.size[0], cropped.size[1]],
                "source_index": idx,
                "source_image_path": img_info.get("image_path"),
            }
        )

    return extended_images

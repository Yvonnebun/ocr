"""
Utility functions for PDF Image-Text Separation Pipeline
"""
import re
import numpy as np
from shapely.geometry import box as shapely_box
from typing import List, Tuple, Dict, Any


# codex update: candidate preprocessing helpers (align with floorplan script)
def clamp_bbox_float(bbox: List[float], width: int, height: int) -> List[float]:
    x0, y0, x1, y1 = bbox
    x0 = float(max(0, min(width - 1, round(x0))))
    x1 = float(max(0, min(width, round(x1))))
    y0 = float(max(0, min(height - 1, round(y0))))
    y1 = float(max(0, min(height, round(y1))))
    if x1 <= x0:
        x1 = min(float(width), x0 + 1.0)
    if y1 <= y0:
        y1 = min(float(height), y0 + 1.0)
    return [x0, y0, x1, y1]


def bbox_wh(bbox: List[float]) -> Tuple[float, float]:
    x0, y0, x1, y1 = bbox
    return max(0.0, x1 - x0), max(0.0, y1 - y0)


def bbox_area(bbox: List[float]) -> float:
    w, h = bbox_wh(bbox)
    return w * h


def bbox_intersection_area(a: List[float], b: List[float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    x0, y0 = max(ax0, bx0), max(ay0, by0)
    x1, y1 = min(ax1, bx1), min(ay1, by1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0)


def overlap_ratio(a: List[float], b: List[float]) -> float:
    inter = bbox_intersection_area(a, b)
    mina = min(bbox_area(a), bbox_area(b)) + 1e-6
    return float(inter / mina)


def merge_union(a: List[float], b: List[float]) -> List[float]:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return [min(ax0, bx0), min(ay0, by0), max(ax1, bx1), max(ay1, by1)]


def is_sidebar_bbox(
    bbox: List[float],
    width: int,
    height: int,
    side_margin_ratio: float = 0.12,
    min_height_ratio: float = 0.55,
    max_width_ratio: float = 0.22,
) -> bool:
    x0, y0, x1, y1 = bbox
    bw = (x1 - x0) / (width + 1e-6)
    bh = (y1 - y0) / (height + 1e-6)
    near_left = (x0 / (width + 1e-6)) <= side_margin_ratio
    near_right = (x1 / (width + 1e-6)) >= (1.0 - side_margin_ratio)
    return (bh >= min_height_ratio) and (bw <= max_width_ratio) and (near_left or near_right)


def merge_overlaps(candidates: List[List[float]], overlap_th: float) -> List[List[float]]:
    if not candidates:
        return []
    candidates = sorted(candidates, key=bbox_area, reverse=True)
    changed = True
    while changed:
        changed = False
        kept: List[List[float]] = []
        for bbox in candidates:
            merged = False
            for idx in range(len(kept)):
                if overlap_ratio(bbox, kept[idx]) >= overlap_th:
                    kept[idx] = merge_union(kept[idx], bbox)
                    merged = True
                    changed = True
                    break
            if not merged:
                kept.append(bbox)
        candidates = sorted(kept, key=bbox_area, reverse=True)
    return candidates


def preprocess_candidates(
    candidates: List[List[float]],
    width: int,
    height: int,
    min_w: float,
    min_h: float,
    overlap_th: float,
    min_area_ratio: float,
    sidebar_params: Dict[str, float],
) -> List[List[float]]:
    # codex update: align candidate cleanup with floorplan script
    page_area = float(width * height)
    tmp: List[List[float]] = []
    for bbox in candidates:
        bb = clamp_bbox_float(bbox, width, height)
        bw, bh = bbox_wh(bb)
        if bw < min_w or bh < min_h:
            continue
        tmp.append(bb)

    tmp = [bbox for bbox in tmp if not is_sidebar_bbox(bbox, width, height, **sidebar_params)]
    tmp = merge_overlaps(tmp, overlap_th=overlap_th)

    final = [bbox for bbox in tmp if bbox_area(bbox) >= (page_area * min_area_ratio)]
    return final


def bbox_intersect(bbox1: List[float], bbox2: List[float]) -> bool:
    """
    Check if two bboxes intersect (pixel coordinates).
    bbox format: [x0, y0, x1, y1]
    """
    x0_1, y0_1, x1_1, y1_1 = bbox1
    x0_2, y0_2, x1_2, y1_2 = bbox2
    
    return not (x1_1 < x0_2 or x1_2 < x0_1 or y1_1 < y0_2 or y1_2 < y0_1)


def bbox_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Calculate IoU (Intersection over Union) of two bboxes.
    bbox format: [x0, y0, x1, y1]
    """
    x0_1, y0_1, x1_1, y1_1 = bbox1
    x0_2, y0_2, x1_2, y1_2 = bbox2
    
    # Intersection
    x0_i = max(x0_1, x0_2)
    y0_i = max(y0_1, y0_2)
    x1_i = min(x1_1, x1_2)
    y1_i = min(y1_1, y1_2)
    
    if x1_i <= x0_i or y1_i <= y0_i:
        return 0.0
    
    intersection = (x1_i - x0_i) * (y1_i - y0_i)
    area1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    area2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union


def bbox_contains(bbox1: List[float], bbox2: List[float], threshold: float = 0.8) -> bool:
    """
    Check if bbox1 contains bbox2 (with threshold for partial overlap).
    Returns True if bbox2 is mostly inside bbox1.
    """
    x0_1, y0_1, x1_1, y1_1 = bbox1
    x0_2, y0_2, x1_2, y1_2 = bbox2
    
    # Calculate intersection
    x0_i = max(x0_1, x0_2)
    y0_i = max(y0_1, y0_2)
    x1_i = min(x1_1, x1_2)
    y1_i = min(y1_1, y1_2)
    
    if x1_i <= x0_i or y1_i <= y0_i:
        return False
    
    intersection = (x1_i - x0_i) * (y1_i - y0_i)
    area2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    
    if area2 == 0:
        return False
    
    coverage = intersection / area2
    return coverage >= threshold


def deduplicate_bboxes(bboxes: List[List[float]], iou_threshold: float = 0.8) -> List[int]:
    """
    Deduplicate bboxes based on IoU threshold.
    Returns list of indices to keep.
    """
    if len(bboxes) == 0:
        return []
    
    keep = []
    n = len(bboxes)
    suppressed = [False] * n
    
    # Sort by area (largest first)
    areas = [(i, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) for i, bbox in enumerate(bboxes)]
    areas.sort(key=lambda x: x[1], reverse=True)
    
    for i, (idx, _) in enumerate(areas):
        if suppressed[idx]:
            continue
        
        keep.append(idx)
        
        # Suppress overlapping boxes
        for j in range(i + 1, n):
            other_idx = areas[j][0]
            if not suppressed[other_idx]:
                iou = bbox_iou(bboxes[idx], bboxes[other_idx])
                if iou >= iou_threshold:
                    suppressed[other_idx] = True
    
    return sorted(keep)


def filter_text_by_regions(text_blocks: List[Dict], exclude_regions: List[List[float]], 
                          threshold: float = 0.5) -> List[Dict]:
    """
    Filter text blocks by excluding those that overlap with exclude_regions.
    threshold: minimum overlap ratio to exclude
    """
    filtered = []
    
    for text_block in text_blocks:
        text_bbox = text_block.get('bbox_px', [])
        if not text_bbox:
            continue
        
        should_exclude = False
        for exclude_bbox in exclude_regions:
            if bbox_contains(exclude_bbox, text_bbox, threshold=threshold):
                should_exclude = True
                break
        
        if not should_exclude:
            filtered.append(text_block)
    
    return filtered


def cluster_lines(ocr_results: List[Dict], tolerance: float = 10.0) -> int:
    """
    Cluster OCR words into lines based on y-coordinate proximity.
    More robust than simple unique y-counting.
    
    Args:
        ocr_results: List of OCR word dicts with 'bbox'
        tolerance: Y-coordinate tolerance for grouping (pixels)
    
    Returns:
        Number of distinct lines (after clustering)
    """
    if len(ocr_results) == 0:
        return 0
    
    # Extract y-coordinates (top of bbox)
    y_coords = []
    for result in ocr_results:
        bbox = result.get('bbox', [])
        if len(bbox) >= 4:
            y_coords.append(bbox[1])  # y0 (top)
    
    if len(y_coords) == 0:
        return 0
    
    # Cluster y-coordinates
    y_coords = sorted(y_coords)
    clusters = []
    current_cluster = [y_coords[0]]
    
    for y in y_coords[1:]:
        # Check if y is within tolerance of current cluster
        cluster_center = np.mean(current_cluster)
        if abs(y - cluster_center) <= tolerance:
            current_cluster.append(y)
        else:
            # Start new cluster
            clusters.append(current_cluster)
            current_cluster = [y]
    
    if current_cluster:
        clusters.append(current_cluster)
    
    return len(clusters)


def calculate_text_alignment_score(ocr_results: List[Dict]) -> float:
    """
    Calculate alignment score for OCR results.
    Higher score means better alignment (more likely to be text block).
    Returns score between 0 and 1.
    """
    if len(ocr_results) < 2:
        return 0.0
    
    # Extract x-coordinates of left edges
    left_edges = []
    for result in ocr_results:
        bbox = result.get('bbox', [])
        if len(bbox) >= 4:
            left_edges.append(bbox[0])
    
    if len(left_edges) < 2:
        return 0.0
    
    # Calculate standard deviation of left edges
    left_edges = np.array(left_edges)
    std = np.std(left_edges)
    
    # Normalize by average width
    avg_width = np.mean([r.get('bbox', [2, 0, 0, 0])[2] - r.get('bbox', [0, 0, 0, 0])[0] 
                        for r in ocr_results if len(r.get('bbox', [])) >= 4])
    if avg_width == 0:
        return 0.0
    
    # Lower std relative to width = better alignment
    alignment_score = 1.0 / (1.0 + std / avg_width)
    return min(alignment_score, 1.0)


def is_caption_text(text: str, patterns: List[str]) -> bool:
    """
    Check if text matches caption patterns.
    """
    text = text.strip()
    for pattern in patterns:
        if re.match(pattern, text, re.IGNORECASE):
            return True
    return False


def pdf_coord_to_pixel(bbox_pdf: Tuple[float, float, float, float], 
                       page_width_px: int, page_height_px: int,
                       page_width_pdf: float, page_height_pdf: float) -> List[float]:
    """
    Convert PDF coordinates to pixel coordinates.
    bbox_pdf: (x0, y0, x1, y1) in PDF coordinates
    Returns: [x0, y0, x1, y1] in pixel coordinates
    """
    x0_pdf, y0_pdf, x1_pdf, y1_pdf = bbox_pdf
    
    scale_x = page_width_px / page_width_pdf
    scale_y = page_height_px / page_height_pdf
    
    x0_px = x0_pdf * scale_x
    y0_px = y0_pdf * scale_y
    x1_px = x1_pdf * scale_x
    y1_px = y1_pdf * scale_y
    
    return [x0_px, y0_px, x1_px, y1_px]

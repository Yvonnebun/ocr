from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


def polygons_to_mask(polys: List[object], image_shape: Tuple[int, int]) -> np.ndarray:
    height, width = image_shape
    mask = np.zeros((height, width), dtype=np.uint8)
    if not polys:
        return mask
    for poly in polys:
        if not poly:
            continue
        if isinstance(poly, dict):
            poly = poly.get("points", [])
        if not poly or len(poly) < 6:
            continue
        coords = np.array(poly, dtype=np.float32).reshape(-1, 2)
        coords = np.rint(coords).astype(np.int32)
        coords[:, 0] = np.clip(coords[:, 0], 0, width - 1)
        coords[:, 1] = np.clip(coords[:, 1], 0, height - 1)
        cv2.fillPoly(mask, [coords], 255)
    return mask


def mask_to_polygons(mask: np.ndarray) -> List[List[float]]:
    if mask is None or mask.size == 0:
        return []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: List[List[float]] = []
    for contour in contours:
        if contour.shape[0] < 3:
            continue
        coords = contour.reshape(-1, 2)
        poly = coords.astype(np.float32).flatten().tolist()
        polygons.append(poly)
    return polygons


def merge_wall_predictions(
    wall_a: Dict[str, Any],
    wall_b: Dict[str, Any],
    image_shape: Tuple[int, int],
    **kwargs
) -> Dict[str, Any]:
    """
    Merge wall polygons from two models by unioning their masks, then extracting polygons.
    """
    wall_a_polys = wall_a.get("polygons", []) or []
    wall_b_polys = wall_b.get("polygons", []) or []
    mask_a = polygons_to_mask(wall_a_polys, image_shape=image_shape)
    mask_b = polygons_to_mask(wall_b_polys, image_shape=image_shape)
    combined_mask = np.maximum(mask_a, mask_b)
    combined_polys = mask_to_polygons(combined_mask)

    merged_result = dict(wall_a)
    merged_result["model_name"] = "wall_merged"
    merged_result["model_version"] = "merged"
    merged_result["polygons"] = combined_polys
    merged_result["detections"] = (wall_a.get("detections", []) or []) + (wall_b.get("detections", []) or [])

    meta = dict(wall_a.get("meta", {}))
    meta["merged_from"] = ["wall_a", "wall_b"]
    meta["merged"] = True
    meta["source_polygon_counts"] = {"wall_a": len(wall_a_polys), "wall_b": len(wall_b_polys)}
    merged_result["meta"] = meta
    return merged_result


def process_room_polygons(
    room: Dict[str, Any],
    wall: Dict[str, Any],
    image_shape: Tuple[int, int],
    **kwargs
) -> Dict[str, Any]:
    """
    Placeholder for room polygon postprocess.
    For now: return room unchanged, but set meta fields indicating no-op.
    """
    meta = dict(room.get("meta", {}))
    meta["poly_processed"] = False
    room["meta"] = meta
    return room

from typing import Any, Dict, List, Tuple

import numpy as np


def _cv2_module():
    import cv2

    return cv2


def polygons_to_mask(polys: List[object], image_shape: Tuple[int, int]) -> np.ndarray:
    height, width = image_shape
    mask = np.zeros((height, width), dtype=np.uint8)
    if not polys:
        return mask
    cv2 = _cv2_module()
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


def rings_to_mask(rings: List[Dict[str, Any]], image_shape: Tuple[int, int]) -> np.ndarray:
    height, width = image_shape
    mask = np.zeros((height, width), dtype=np.uint8)
    if not rings:
        return mask
    cv2 = _cv2_module()
    for ring in rings:
        exterior = ring.get("exterior") or []
        if len(exterior) < 6:
            continue
        exterior_pts = np.array(exterior, dtype=np.float32).reshape(-1, 2)
        exterior_pts = np.rint(exterior_pts).astype(np.int32)
        exterior_pts[:, 0] = np.clip(exterior_pts[:, 0], 0, width - 1)
        exterior_pts[:, 1] = np.clip(exterior_pts[:, 1], 0, height - 1)
        cv2.fillPoly(mask, [exterior_pts], 255)

        for hole in ring.get("holes", []) or []:
            if len(hole) < 6:
                continue
            hole_pts = np.array(hole, dtype=np.float32).reshape(-1, 2)
            hole_pts = np.rint(hole_pts).astype(np.int32)
            hole_pts[:, 0] = np.clip(hole_pts[:, 0], 0, width - 1)
            hole_pts[:, 1] = np.clip(hole_pts[:, 1], 0, height - 1)
            cv2.fillPoly(mask, [hole_pts], 0)
    return mask


def mask_to_polygons(mask: np.ndarray) -> List[List[float]]:
    if mask is None or mask.size == 0 or not np.any(mask):
        return []
    cv2 = _cv2_module()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: List[List[float]] = []
    for contour in contours:
        if contour.shape[0] < 3:
            continue
        coords = contour.reshape(-1, 2)
        poly = coords.astype(np.float32).flatten().tolist()
        polygons.append(poly)
    return polygons


def mask_to_polygons_rings(mask: np.ndarray) -> List[Dict[str, Any]]:
    """
    Return polygons as rings with holes:
    [{"exterior":[...], "holes":[[...],[...]]}, ...]
    Points are flattened [x1, y1, x2, y2, ...].
    """
    if mask is None or mask.size == 0 or not np.any(mask):
        return []
    height, width = mask.shape[:2]
    cv2 = _cv2_module()
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None or len(contours) == 0:
        return []

    hierarchy = hierarchy[0]
    rings: List[Dict[str, Any]] = []

    for i, h in enumerate(hierarchy):
        parent = int(h[3])
        if parent != -1:
            continue
        contour = contours[i]
        if contour is None or contour.shape[0] < 3:
            continue
        exterior_coords = contour.reshape(-1, 2).astype(np.int32)
        exterior_coords[:, 0] = np.clip(exterior_coords[:, 0], 0, width - 1)
        exterior_coords[:, 1] = np.clip(exterior_coords[:, 1], 0, height - 1)
        exterior = exterior_coords.astype(np.float32).flatten().tolist()
        holes: List[List[float]] = []
        child = int(h[2])
        while child != -1:
            hole_contour = contours[child]
            if hole_contour is not None and hole_contour.shape[0] >= 3:
                hole_coords = hole_contour.reshape(-1, 2).astype(np.int32)
                hole_coords[:, 0] = np.clip(hole_coords[:, 0], 0, width - 1)
                hole_coords[:, 1] = np.clip(hole_coords[:, 1], 0, height - 1)
                hole = hole_coords.astype(np.float32).flatten().tolist()
                holes.append(hole)
            child = int(hierarchy[child][0])
        rings.append({"exterior": exterior, "holes": holes})

    return rings


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
    wall_a_rings = wall_a.get("polygons_rings", []) or []
    wall_b_rings = wall_b.get("polygons_rings", []) or []

    if wall_a_rings:
        mask_a = rings_to_mask(wall_a_rings, image_shape=image_shape)
    else:
        mask_a = polygons_to_mask(wall_a_polys, image_shape=image_shape)
    if wall_b_rings:
        mask_b = rings_to_mask(wall_b_rings, image_shape=image_shape)
    else:
        mask_b = polygons_to_mask(wall_b_polys, image_shape=image_shape)
    combined_mask = np.maximum(mask_a, mask_b)
    combined_rings = mask_to_polygons_rings(combined_mask)
    combined_polys = [ring["exterior"] for ring in combined_rings if ring.get("exterior")]

    merged_result = dict(wall_a)
    merged_result["model_name"] = "wall_merged"
    merged_result["model_version"] = "merged"
    merged_result["polygons"] = combined_polys
    merged_result["polygons_rings"] = combined_rings
    merged_result["detections"] = (wall_a.get("detections", []) or []) + (wall_b.get("detections", []) or [])

    meta = dict(wall_a.get("meta", {}))
    meta["merged_from"] = ["wall_a", "wall_b"]
    meta["merged"] = True
    meta["source_polygon_counts"] = {"wall_a": len(wall_a_polys), "wall_b": len(wall_b_polys)}
    meta["source_ring_counts"] = {"wall_a": len(wall_a_rings), "wall_b": len(wall_b_rings)}

    meta["merged_polygon_counts"] = {
        "exteriors": len(combined_polys),
        "rings": len(combined_rings),
        "holes_total": sum(len(ring.get("holes", [])) for ring in combined_rings),
    }
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

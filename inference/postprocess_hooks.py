from typing import Any, Dict, Tuple


def merge_wall_predictions(
    wall_a: Dict[str, Any],
    wall_b: Dict[str, Any],
    image_shape: Tuple[int, int],
    **kwargs
) -> Dict[str, Any]:
    """
    Placeholder for wall merge logic.
    For now: return wall_a, but set meta fields indicating merge not implemented.
    """
    meta = dict(wall_a.get("meta", {}))
    meta["merged_from"] = ["wall_a", "wall_b"]
    meta["merged"] = False
    wall_a["meta"] = meta
    return wall_a


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

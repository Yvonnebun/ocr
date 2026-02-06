from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np


def _iter_polygons(polys: Iterable[object]) -> Iterable[np.ndarray]:
    for poly in polys:
        if not poly:
            continue
        if isinstance(poly, dict):
            poly = poly.get("points", [])
        if not poly or len(poly) < 6:
            continue
        coords = np.array(poly, dtype=np.float32).reshape(-1, 2)
        yield coords


def _scale_polygon(coords: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return coords
    return coords / scale


def _draw_polygons(image: np.ndarray, polys: Iterable[object], color: Tuple[int, int, int]) -> None:
    for coords in _iter_polygons(polys):
        pts = coords.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [pts], isClosed=True, color=color, thickness=2)


def _resolve_image_path(image_info: Dict[str, object]) -> Path:
    extended_path = image_info.get("extended_image_path")
    image_path = image_info.get("image_path")
    path_str = extended_path or image_path
    if not path_str:
        raise ValueError("Missing image_path for image entry.")
    return Path(str(path_str))


def _annotate_image(image_info: Dict[str, object], output_dir: Path) -> Path | None:
    bundle = image_info.get("floorplan_inference")
    if not isinstance(bundle, dict):
        return None

    image_path = _resolve_image_path(image_info)
    image = cv2.imread(str(image_path))
    if image is None:
        return None

    scale_factor = float(bundle.get("image", {}).get("scale_factor", 1.0) or 1.0)
    wall_polys = bundle.get("wall", {}).get("result", {}).get("polygons", [])
    room_polys = bundle.get("room", {}).get("result", {}).get("polygons", [])

    if scale_factor != 1.0:
        wall_polys = [
            _scale_polygon(
                np.array((poly.get("points") if isinstance(poly, dict) else poly), dtype=np.float32).reshape(-1, 2),
                scale_factor,
            ).flatten().tolist()
            for poly in wall_polys
            if poly
        ]
        room_polys = [
            _scale_polygon(
                np.array((poly.get("points") if isinstance(poly, dict) else poly), dtype=np.float32).reshape(-1, 2),
                scale_factor,
            ).flatten().tolist()
            for poly in room_polys
            if poly
        ]

    _draw_polygons(image, wall_polys, (0, 255, 0))
    _draw_polygons(image, room_polys, (255, 0, 0))

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{image_path.stem}_polys.png"
    cv2.imwrite(str(output_path), image)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Annotate floorplan polygons on extracted images.")
    parser.add_argument("result_json", type=Path, help="Path to result.json from pipeline output.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output") / "poly_overlays",
        help="Directory to write annotated images.",
    )
    args = parser.parse_args()

    with args.result_json.open("r", encoding="utf-8") as handle:
        result = json.load(handle)

    pages = result.get("pages", [])
    for page in pages:
        page_idx = page.get("page_idx")
        door_count = page.get("door_count")
        print(f"Page {page_idx}: door_count={door_count}")
        for image_info in page.get("images", []):
            output_path = _annotate_image(image_info, args.output_dir)
            if output_path:
                print(f"  saved: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

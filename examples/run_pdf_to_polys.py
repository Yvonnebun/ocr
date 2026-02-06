from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config
from pipeline import process_pdf, save_result


def _collect_polys(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    polys: List[Dict[str, Any]] = []
    for page in result.get("pages", []):
        page_idx = page.get("page_idx")
        for img_idx, image in enumerate(page.get("images", [])):
            bundle = image.get("floorplan_inference")
            if not bundle:
                continue
            polys.append(
                {
                    "page_idx": page_idx,
                    "image_idx": img_idx,
                    "image_path": image.get("image_path"),
                    "extended_image_path": image.get("extended_image_path"),
                    "floorplan": {
                        "image": bundle.get("image"),
                        "gate": bundle.get("gate"),
                        "wall_polygons": bundle.get("wall", {}).get("result", {}).get("polygons", []),
                        "room_polygons": bundle.get("room", {}).get("result", {}).get("polygons", []),
                        "window_count": bundle.get("window", {}).get("count", 0),
                        "total_ms": bundle.get("total_ms"),
                    },
                }
            )
    return polys


def _floorplan_configured() -> bool:
    return bool(
        config.FLOORPLAN_INFERENCE_ENABLED
        and config.FLOORPLAN_WALL_A_WEIGHTS
        and config.FLOORPLAN_WALL_B_WEIGHTS
        and config.FLOORPLAN_ROOM_WEIGHTS
        and config.FLOORPLAN_WINDOW_WEIGHTS
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full PDF pipeline and emit floorplan polygons as JSON."
    )
    parser.add_argument("pdf_path", help="Path to input PDF.")
    parser.add_argument(
        "--output-dir",
        default=config.OUTPUT_DIR,
        help=f"Pipeline output directory (default: {config.OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--polys-out",
        default=None,
        help="Path to write polygons JSON (default: <output-dir>/polys.json).",
    )
    parser.add_argument(
        "--result-out",
        default=None,
        help="Optional path to write full pipeline result JSON for debugging.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    polys_out = args.polys_out or str(Path(output_dir) / "polys.json")

    if not _floorplan_configured():
        print(
            "WARNING: floorplan inference is not fully configured. "
            "Set FLOORPLAN_INFERENCE_ENABLED=true and provide weight paths "
            "(FLOORPLAN_WALL_A_WEIGHTS, FLOORPLAN_WALL_B_WEIGHTS, "
            "FLOORPLAN_ROOM_WEIGHTS, FLOORPLAN_WINDOW_WEIGHTS)."
        )

    result = process_pdf(args.pdf_path, output_dir)
    if args.result_out:
        save_result(result, args.result_out)
    polys = _collect_polys(result)

    output_path = Path(polys_out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pdf_path": args.pdf_path,
        "page_count": result.get("meta", {}).get("page_count"),
        "polys": polys,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(polys)} polygon bundles to {output_path}")


if __name__ == "__main__":
    main()

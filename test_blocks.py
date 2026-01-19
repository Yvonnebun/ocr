"""
Block-based test script with visual outputs.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict

from PIL import Image, ImageDraw

import config
from page_render import render_pdf_pages
from layout_detect import detect_layout, filter_figure_blocks
from image_extraction import extract_image_assets
from native_text import extract_native_text
from page_gate import page_has_floorplan_keyword
from run_logger import init_run_logger, get_run_logger
import utils


def _save_overlay(image_path: str, bboxes: List[List[float]], out_path: str, color: str) -> None:
    # codex update: visualize bounding boxes for test outputs
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for bbox in bboxes:
        x0, y0, x1, y1 = [int(v) for v in bbox]
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
    img.save(out_path)


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python test_blocks.py <pdf_path>")
        return 1

    pdf_path = sys.argv[1]
    if not os.path.exists(pdf_path):
        print(f"PDF not found: {pdf_path}")
        return 1

    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    # codex update: default test outputs under shared output directory
    out_root = Path(config.TEST_OUTPUT_DIR) / f"run_{run_id}"
    out_root.mkdir(parents=True, exist_ok=True)

    run_logger = init_run_logger(str(out_root))
    run_logger.log_event("test_start", {"pdf_path": pdf_path})
    start_time = time.monotonic()

    # Block 1: Render pages
    print("\n[Block 1] Render PDF pages")
    page_info = render_pdf_pages(pdf_path, str(out_root / "renders"))
    print(f"Rendered {len(page_info)} pages")

    # Focus on first page for visualization
    image_path, width_px, height_px = page_info[0]

    # Block 2: Native text + keyword gate
    print("\n[Block 2] Keyword gate")
    native_text_blocks, _ = extract_native_text(pdf_path, 0, width_px, height_px)
    gate_passed, force_keep, gate_source = page_has_floorplan_keyword(image_path, native_text_blocks)
    print(f"Gate passed: {gate_passed} (force_keep={force_keep}, source={gate_source})")
    run_logger.log_event(
        "test_gate",
        {"passed": gate_passed, "force_keep": force_keep, "source": gate_source},
    )

    # Block 3: Layout detection
    print("\n[Block 3] Layout detection")
    layout_blocks = detect_layout(image_path)
    figure_blocks = filter_figure_blocks(layout_blocks)
    print(f"Layout blocks: {len(layout_blocks)}, figure candidates: {len(figure_blocks)}")
    layout_bbox = [b["bbox_px"] for b in layout_blocks]
    _save_overlay(image_path, layout_bbox, str(out_root / "layout_overlay.png"), "red")

    # Block 4: Region refinement (bypassed)
    # codex update: skip region_refiner and use candidate preprocessing directly
    print("\n[Block 4] Region refinement (bypassed)")
    candidate_bboxes = [block["bbox_px"] for block in figure_blocks]
    candidate_bboxes = utils.preprocess_candidates(
        candidate_bboxes,
        width_px,
        height_px,
        min_w=config.CANDIDATE_MIN_W,
        min_h=config.CANDIDATE_MIN_H,
        overlap_th=config.CANDIDATE_OVERLAP_TH,
        min_area_ratio=config.CANDIDATE_MIN_AREA_RATIO,
        sidebar_params=config.SIDEBAR_PARAMS,
    )
    image_regions = [{"bbox_px": bbox, "source": "preprocess"} for bbox in candidate_bboxes]
    image_bbox = [r["bbox_px"] for r in image_regions]
    _save_overlay(image_path, image_bbox, str(out_root / "refined_overlay.png"), "green")
    print(f"Kept image regions: {len(image_regions)}")
    # codex update: original refiner block (commented out for later gate work)
    # print("\n[Block 4] Region refinement")
    # refine_result = refine_all_candidates(image_path, figure_blocks)
    # image_regions = refine_result["image_regions_final"]
    # image_bbox = [r["bbox_px"] for r in image_regions]
    # _save_overlay(image_path, image_bbox, str(out_root / "refined_overlay.png"), "green")
    # print(f"Refined image regions: {len(image_regions)}")

    # Block 5: Image extraction with blueprint filter
    print("\n[Block 5] Image extraction")
    extracted_images = extract_image_assets(
        image_path,
        image_regions,
        str(out_root / "extracted"),
        page_idx=0,
        force_keep=force_keep,
    )
    print(f"Extracted images: {len(extracted_images)}")
    run_logger.record_images(extracted_images)

    # Block 6: Summary
    elapsed = time.monotonic() - start_time
    summary = {
        "pdf_path": pdf_path,
        "run_id": run_id,
        "elapsed_seconds": round(elapsed, 3),
        "gate_passed": gate_passed,
        "gate_source": gate_source,
        "layout_blocks": len(layout_blocks),
        "figure_blocks": len(figure_blocks),
        "image_regions": len(image_regions),
        "extracted_images": [img["image_path"] for img in extracted_images],
    }
    with open(out_root / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    run_logger.log_event("test_complete", summary)
    run_logger.finalize()

    print(f"\nOutputs saved to: {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

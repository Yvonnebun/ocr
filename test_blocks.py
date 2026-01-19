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
from native_text import extract_native_text, filter_text_excluding_images
from page_gate import page_has_floorplan_keyword
from run_logger import init_run_logger, get_run_logger
from ocr_service import paddle_ocr
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

    extracted_images_all: List[Dict] = []
    page_summaries: List[Dict] = []
    text_outputs: List[Dict] = []

    for page_idx, (image_path, width_px, height_px) in enumerate(page_info):
        print(f"\n[Page {page_idx + 1}/{len(page_info)}]")

        # Block 2: Native text + keyword gate
        print("[Block 2] Keyword gate")
        native_text_blocks, has_native_text = extract_native_text(pdf_path, page_idx, width_px, height_px)
        gate_passed, force_keep, gate_source = page_has_floorplan_keyword(image_path, native_text_blocks)
        print(f"Gate passed: {gate_passed} (force_keep={force_keep}, source={gate_source})")
        run_logger.log_event(
            "test_gate",
            {"page_idx": page_idx, "passed": gate_passed, "force_keep": force_keep, "source": gate_source},
        )

        if not gate_passed:
            page_summaries.append(
                {
                    "page_idx": page_idx,
                    "gate_passed": False,
                    "gate_source": gate_source,
                    "layout_blocks": 0,
                    "figure_blocks": 0,
                    "image_regions": 0,
                    "extracted_images": [],
                    "text_block_count": 0,
                    "text_content": "",
                }
            )
            continue

        # Block 3: Layout detection
        print("[Block 3] Layout detection")
        layout_blocks = detect_layout(image_path)
        figure_blocks = filter_figure_blocks(layout_blocks)
        print(f"Layout blocks: {len(layout_blocks)}, figure candidates: {len(figure_blocks)}")
        layout_bbox = [b["bbox_px"] for b in layout_blocks]
        _save_overlay(
            image_path,
            layout_bbox,
            str(out_root / f"layout_overlay_page_{page_idx:04d}.png"),
            "red",
        )

        # Block 4: Region refinement (bypassed)
        # codex update: skip region_refiner and use candidate preprocessing directly
        print("[Block 4] Region refinement (bypassed)")
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
        _save_overlay(
            image_path,
            image_bbox,
            str(out_root / f"refined_overlay_page_{page_idx:04d}.png"),
            "green",
        )
        print(f"Kept image regions: {len(image_regions)}")

        # Block 5: Image extraction with blueprint filter
        print("[Block 5] Image extraction")
        extracted_images = extract_image_assets(
            image_path,
            image_regions,
            str(out_root / "extracted"),
            page_idx=page_idx,
            force_keep=force_keep,
        )
        print(f"Extracted images: {len(extracted_images)}")
        run_logger.record_images(extracted_images)
        extracted_images_all.extend(extracted_images)

        # Block 5.5: Text extraction (native or OCR) excluding image regions
        print("[Block 5.5] Text extraction")
        text_blocks: List[Dict] = []
        if has_native_text:
            try:
                text_blocks = filter_text_excluding_images(native_text_blocks, image_bbox)
            except Exception:
                text_blocks = native_text_blocks
        else:
            try:
                ocr_blocks = paddle_ocr(image_path)
                text_blocks = [
                    {"text": block.get("text", ""), "bbox_px": block.get("bbox", [])}
                    for block in ocr_blocks
                    if block.get("bbox")
                ]
                text_blocks = utils.filter_text_by_regions(text_blocks, image_bbox, threshold=0.5)
            except Exception:
                text_blocks = []
        text_content = " ".join(
            block.get("text", "").strip() for block in text_blocks if block.get("text", "").strip()
        )
        text_path = out_root / f"text_page_{page_idx:04d}.txt"
        with open(text_path, "w", encoding="utf-8") as text_handle:
            text_handle.write(text_content)
        text_outputs.append({"page_idx": page_idx, "text_path": str(text_path)})
        print(f"Text blocks: {len(text_blocks)}")

        page_summaries.append(
            {
                "page_idx": page_idx,
                "gate_passed": gate_passed,
                "gate_source": gate_source,
                "layout_blocks": len(layout_blocks),
                "figure_blocks": len(figure_blocks),
                "image_regions": len(image_regions),
                "extracted_images": [img["image_path"] for img in extracted_images],
                "text_block_count": len(text_blocks),
                "text_content": text_content,
            }
        )

    # Block 6: Summary
    elapsed = time.monotonic() - start_time
    summary = {
        "pdf_path": pdf_path,
        "run_id": run_id,
        "elapsed_seconds": round(elapsed, 3),
        "pages": page_summaries,
        "extracted_images": [img["image_path"] for img in extracted_images_all],
        "text_outputs": text_outputs,
    }
    with open(out_root / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    run_logger.log_event("test_complete", summary)
    run_logger.finalize()

    print(f"\nOutputs saved to: {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

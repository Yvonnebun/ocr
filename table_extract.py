"""
Table extraction utilities for keyword-matched tables.
"""
from __future__ import annotations

import os
from typing import Dict, List

from PIL import Image

import config
import utils


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _has_keyword(text: str) -> bool:
    normalized = _normalize(text)
    for kw in config.TABLE_KEYWORDS:
        if kw in normalized:
            return True
    return False


def _collect_text_for_bbox(text_blocks: List[Dict], bbox: List[float]) -> str:
    parts = []
    for block in text_blocks:
        block_bbox = block.get("bbox_px")
        if not block_bbox:
            continue
        if utils.overlap_ratio(block_bbox, bbox) >= 0.25:
            block_text = block.get("text", "").strip()
            if block_text:
                parts.append(block_text)
    return " ".join(parts)


def extract_keyword_tables(
    image_path: str,
    table_blocks: List[Dict],
    text_blocks: List[Dict],
    output_dir: str,
    page_idx: int,
) -> List[Dict]:
    os.makedirs(output_dir, exist_ok=True)
    if not table_blocks:
        return []

    img = Image.open(image_path)
    width, height = img.size
    extracted = []

    for idx, block in enumerate(table_blocks):
        bbox = block.get("bbox_px")
        if not bbox:
            continue
        bbox = utils.clamp_bbox_float(bbox, width, height)
        text = _collect_text_for_bbox(text_blocks, bbox)
        if not text or not _has_keyword(text):
            continue

        x0, y0, x1, y1 = [int(coord) for coord in bbox]
        if x1 <= x0 or y1 <= y0:
            continue
        crop = img.crop((x0, y0, x1, y1))
        table_filename = f"page_{page_idx:04d}_table_{idx:04d}.png"
        table_path = os.path.join(output_dir, table_filename)
        crop.save(table_path, "PNG")
        extracted.append(
            {
                "table_path": table_path,
                "bbox_px": bbox,
                "image_size": [crop.size[0], crop.size[1]],
                "matched_text": text,
            }
        )

    return extracted

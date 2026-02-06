"""
Step 7: Text Extraction - Extract text blocks from rendered page images via Paddle OCR.
"""
from typing import List, Dict, Tuple

from ocr_service import paddle_ocr
import utils


def extract_native_text(
    pdf_path: str,
    page_idx: int,
    page_width_px: int,
    page_height_px: int,
    image_path: str,
) -> Tuple[List[Dict], bool]:
    """
    Extract text blocks from rendered page image via Paddle OCR.
    
    Args:
        pdf_path: Path to PDF file (unused, kept for compatibility)
        page_idx: Page index (unused, kept for compatibility)
        page_width_px: Page width in pixels (unused, kept for compatibility)
        page_height_px: Page height in pixels (unused, kept for compatibility)
        image_path: Path to rendered page image
    
    Returns:
        Tuple of:
        - List of text blocks: [{'text': str, 'bbox_px': [x0, y0, x1, y1]}, ...]
        - has_text: bool (True if page OCR extracted text)
    """
    _ = (pdf_path, page_idx, page_width_px, page_height_px)

    try:
        ocr_blocks = paddle_ocr(image_path)
    except Exception as e:
        print(f"OCR error in extract_native_text: {e}")
        return [], False

    text_blocks = []
    for block in ocr_blocks:
        text = (block.get("text") or "").strip()
        bbox = block.get("bbox") or []
        if not text or len(bbox) < 4:
            continue

        x0, y0, x1, y1 = map(float, bbox[:4])
        text_blocks.append({"text": text, "bbox_px": [x0, y0, x1, y1]})

    return text_blocks, bool(text_blocks)


def filter_text_excluding_images(text_blocks: List[Dict], 
                                image_regions: List[List[float]]) -> List[Dict]:
    """
    Filter text blocks by excluding those that overlap with image regions.
    
    Args:
        text_blocks: List of text block dicts with 'bbox_px'
        image_regions: List of image bboxes [x0, y0, x1, y1]
    
    Returns:
        Filtered text blocks
    """
    return utils.filter_text_by_regions(text_blocks, image_regions, threshold=0.5)

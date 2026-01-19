"""
Step 9: Scanned Page Handling - OCR on non-image regions
"""
from typing import List, Dict

from ocr_service import paddle_ocr
import utils

def ocr_non_image_regions(image_path: str, image_regions: List[List[float]]) -> List[Dict]:
    """
    Run OCR on page image, excluding image regions.
    
    Args:
        image_path: Path to page image
        image_regions: List of image bboxes [x0, y0, x1, y1]
    
    Returns:
        List of text blocks: [{'text': str, 'bbox_px': [x0, y0, x1, y1]}, ...]
    """
    # Run OCR on full page
    try:
        ocr_blocks = paddle_ocr(image_path)
    except Exception as e:
        print(f"OCR error: {e}")
        return []
    
    # Group words into lines and blocks
    words = []
    for block in ocr_blocks:
        text = str(block.get("text", "")).strip()
        bbox = block.get("bbox", [])
        if not text or not bbox or len(bbox) < 4:
            continue
        x0, y0, x1, y1 = bbox[:4]

        # Check if this word overlaps with any image region
        overlaps_image = False
        for img_bbox in image_regions:
            if utils.bbox_intersect([x0, y0, x1, y1], img_bbox):
                overlaps_image = True
                break

        if not overlaps_image:
            words.append({
                'text': text,
                'bbox': [x0, y0, x1, y1],
                'y_center': (y0 + y1) / 2
            })
    
    if not words:
        return []
    
    # Group words into lines (by y-coordinate)
    words.sort(key=lambda w: (w['y_center'], w['bbox'][0]))
    
    lines = []
    current_line = []
    current_y = None
    
    for word in words:
        y = word['y_center']
        if current_y is None or abs(y - current_y) < 10:  # Same line
            current_line.append(word)
            current_y = y if current_y is None else (current_y + y) / 2
        else:
            if current_line:
                lines.append(current_line)
            current_line = [word]
            current_y = y
    
    if current_line:
        lines.append(current_line)
    
    # Convert lines to text blocks
    text_blocks = []
    for line in lines:
        if not line:
            continue
        
        # Merge words in line
        line_text = " ".join(w['text'] for w in line)
        x0 = min(w['bbox'][0] for w in line)
        y0 = min(w['bbox'][1] for w in line)
        x1 = max(w['bbox'][2] for w in line)
        y1 = max(w['bbox'][3] for w in line)
        
        text_blocks.append({
            'text': line_text,
            'bbox_px': [x0, y0, x1, y1]
        })
    
    # Merge nearby lines into paragraphs
    if not text_blocks:
        return []
    
    merged_blocks = [text_blocks[0]]
    for block in text_blocks[1:]:
        last_block = merged_blocks[-1]
        last_bbox = last_block['bbox_px']
        block_bbox = block['bbox_px']
        
        # Check if vertically close (same paragraph)
        vertical_gap = block_bbox[1] - last_bbox[3]
        if vertical_gap < 30:  # Close vertically
            # Merge
            last_block['text'] += " " + block['text']
            last_block['bbox_px'] = [
                min(last_bbox[0], block_bbox[0]),
                min(last_bbox[1], block_bbox[1]),
                max(last_bbox[2], block_bbox[2]),
                max(last_bbox[3], block_bbox[3])
            ]
        else:
            merged_blocks.append(block)
    
    return merged_blocks

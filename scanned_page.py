"""
Step 9: Scanned Page Handling - OCR on non-image regions
"""
import pytesseract
from PIL import Image
import numpy as np
from typing import List, Dict
import config
import utils
import config

if config.TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD

def ocr_non_image_regions(image_path: str, image_regions: List[List[float]]) -> List[Dict]:
    """
    Run OCR on page image, excluding image regions.
    
    Args:
        image_path: Path to page image
        image_regions: List of image bboxes [x0, y0, x1, y1]
    
    Returns:
        List of text blocks: [{'text': str, 'bbox_px': [x0, y0, x1, y1]}, ...]
    """
    img = Image.open(image_path)
    width, height = img.size
    
    # Run OCR on full page
    try:
        ocr_data = pytesseract.image_to_data(
            img,
            lang=config.OCR_LANG,
            output_type=pytesseract.Output.DICT
        )
    except Exception as e:
        print(f"OCR error: {e}")
        return []
    
    # Group words into lines and blocks
    words = []
    n_boxes = len(ocr_data['text'])
    
    for i in range(n_boxes):
        text = ocr_data['text'][i].strip()
        conf = int(ocr_data['conf'][i])
        
        if text and conf > 0:
            left = ocr_data['left'][i]
            top = ocr_data['top'][i]
            width_box = ocr_data['width'][i]
            height_box = ocr_data['height'][i]
            bbox = [left, top, left + width_box, top + height_box]
            
            # Check if this word overlaps with any image region
            overlaps_image = False
            for img_bbox in image_regions:
                if utils.bbox_intersect(bbox, img_bbox):
                    overlaps_image = True
                    break
            
            if not overlaps_image:
                words.append({
                    'text': text,
                    'bbox': bbox,
                    'y_center': top + height_box / 2
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


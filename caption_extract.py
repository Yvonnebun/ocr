"""
Caption Extraction - Extract captions from native text or OCR
"""
from typing import List, Dict
import config
import utils
import re


def extract_captions_from_native(text_blocks: List[Dict]) -> List[Dict]:
    """
    Extract captions from native text blocks.
    
    Args:
        text_blocks: List of text block dicts with 'text' and 'bbox_px'
    
    Returns:
        List of caption dicts: [{'text': str, 'bbox_px': [x0, y0, x1, y1]}, ...]
    """
    captions = []
    
    for block in text_blocks:
        text = block.get('text', '').strip()
        if utils.is_caption_text(text, config.CAPTION_PATTERNS):
            captions.append({
                'text': text,
                'bbox_px': block.get('bbox_px', [])
            })
    
    return captions


def extract_captions_from_ocr(ocr_blocks: List[Dict], image_regions: List[List[float]]) -> List[Dict]:
    """
    Extract captions from OCR results (around image regions).
    
    Args:
        ocr_blocks: List of OCR text blocks
        image_regions: List of image bboxes
    
    Returns:
        List of caption dicts
    """
    captions = []
    
    for img_bbox in image_regions:
        # Look for text blocks near image (below or above)
        img_x0, img_y0, img_x1, img_y1 = img_bbox
        img_center_x = (img_x0 + img_x1) / 2
        img_width = img_x1 - img_x0
        
        # Search in region below image (typical caption position)
        search_y_start = img_y1
        search_y_end = img_y1 + 100  # 100px below image
        search_x_start = img_x0 - img_width * 0.1
        search_x_end = img_x1 + img_width * 0.1
        
        for block in ocr_blocks:
            bbox = block.get('bbox_px', [])
            if len(bbox) < 4:
                continue
            
            block_x0, block_y0, block_x1, block_y1 = bbox
            block_center_x = (block_x0 + block_x1) / 2
            
            # Check if block is in search region
            if (search_y_start <= block_y0 <= search_y_end and
                search_x_start <= block_center_x <= search_x_end):
                
                text = block.get('text', '').strip()
                if utils.is_caption_text(text, config.CAPTION_PATTERNS):
                    captions.append({
                        'text': text,
                        'bbox_px': bbox
                    })
                    break  # One caption per image
    
    return captions


def filter_captions_from_text(text_blocks: List[Dict], captions: List[Dict]) -> List[Dict]:
    """
    Remove caption blocks from text blocks.
    
    Args:
        text_blocks: List of text blocks
        captions: List of caption dicts
    
    Returns:
        Filtered text blocks (without captions)
    """
    if not captions:
        return text_blocks
    
    caption_bboxes = [cap['bbox_px'] for cap in captions]
    filtered = []
    
    for block in text_blocks:
        block_bbox = block.get('bbox_px', [])
        if not block_bbox:
            continue
        
        # Check if block overlaps with any caption
        is_caption = False
        for cap_bbox in caption_bboxes:
            if utils.bbox_iou(block_bbox, cap_bbox) > 0.5:
                is_caption = True
                break
        
        if not is_caption:
            filtered.append(block)
    
    return filtered


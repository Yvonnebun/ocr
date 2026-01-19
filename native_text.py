"""
Step 7: Native Text Extraction - Extract text from PDF with pixel coordinates
"""
import fitz  # PyMuPDF
from typing import List, Dict, Tuple
import utils
import config


def extract_native_text(pdf_path: str, page_idx: int, 
                       page_width_px: int, page_height_px: int) -> Tuple[List[Dict], bool]:
    """
    Extract native text from PDF page.
    
    Args:
        pdf_path: Path to PDF file
        page_idx: Page index (0-based)
        page_width_px: Page width in pixels (from rendered image)
        page_height_px: Page height in pixels (from rendered image)
    
    Returns:
        Tuple of:
        - List of text blocks: [{'text': str, 'bbox_px': [x0, y0, x1, y1]}, ...]
        - has_native_text: bool (True if page has extractable text)
    """
    doc = fitz.open(pdf_path)
    if page_idx >= len(doc):
        return [], False
    
    page = doc[page_idx]
    
    # Get page dimensions in PDF coordinates
    page_rect = page.rect
    page_width_pdf = page_rect.width
    page_height_pdf = page_rect.height
    
    # Extract text blocks
    text_blocks = []
    text_dict = page.get_text("dict")
    
    has_text = False
    
    for block in text_dict.get("blocks", []):
        if "lines" not in block:
            continue
        
        block_text_parts = []
        block_bbox_pdf = None
        
        for line in block.get("lines", []):
            line_text_parts = []
            line_bbox_pdf = None
            
            for span in line.get("spans", []):
                text = span.get("text", "").strip()
                if text:
                    has_text = True
                    line_text_parts.append(text)
                    
                    # Get bbox in PDF coordinates
                    bbox_pdf = span.get("bbox", [])
                    if bbox_pdf and len(bbox_pdf) >= 4:
                        if line_bbox_pdf is None:
                            line_bbox_pdf = list(bbox_pdf)
                        else:
                            # Expand bbox
                            line_bbox_pdf[0] = min(line_bbox_pdf[0], bbox_pdf[0])
                            line_bbox_pdf[1] = min(line_bbox_pdf[1], bbox_pdf[1])
                            line_bbox_pdf[2] = max(line_bbox_pdf[2], bbox_pdf[2])
                            line_bbox_pdf[3] = max(line_bbox_pdf[3], bbox_pdf[3])
            
            if line_text_parts and line_bbox_pdf:
                line_text = " ".join(line_text_parts)
                block_text_parts.append(line_text)
                
                if block_bbox_pdf is None:
                    block_bbox_pdf = list(line_bbox_pdf)
                else:
                    # Expand bbox
                    block_bbox_pdf[0] = min(block_bbox_pdf[0], line_bbox_pdf[0])
                    block_bbox_pdf[1] = min(block_bbox_pdf[1], line_bbox_pdf[1])
                    block_bbox_pdf[2] = max(block_bbox_pdf[2], line_bbox_pdf[2])
                    block_bbox_pdf[3] = max(block_bbox_pdf[3], line_bbox_pdf[3])
        
        if block_text_parts and block_bbox_pdf:
            # Convert to pixel coordinates
            bbox_px = utils.pdf_coord_to_pixel(
                tuple(block_bbox_pdf),
                page_width_px, page_height_px,
                page_width_pdf, page_height_pdf
            )
            
            text_blocks.append({
                'text': " ".join(block_text_parts),
                'bbox_px': bbox_px
            })
    
    doc.close()
    return text_blocks, has_text


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


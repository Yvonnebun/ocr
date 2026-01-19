"""
Step 6: Image OCR - OCR only on extracted image assets
"""
from typing import List, Dict

from ocr_service import paddle_ocr

def ocr_image(image_path: str) -> str:
    """
    Run OCR on a single image.
    
    Args:
        image_path: Path to image file
    
    Returns:
        OCR text (combined)
    """
    try:
        blocks = paddle_ocr(image_path)
        texts = [block.get("text", "").strip() for block in blocks if block.get("text")]
        return " ".join(texts).strip()
    except Exception as e:
        print(f"OCR error for {image_path}: {e}")
        return ""


def ocr_all_images(extracted_images: List[Dict]) -> List[Dict]:
    """
    Run OCR on all extracted images.
    
    Args:
        extracted_images: List of dicts with 'image_path' and 'bbox_px'
    
    Returns:
        List of dicts with added 'ocr_text' field
    """
    results = []
    for img_info in extracted_images:
        image_path = img_info['image_path']
        ocr_text = ocr_image(image_path)
        
        result = img_info.copy()
        result['ocr_text'] = ocr_text
        results.append(result)
    
    return results

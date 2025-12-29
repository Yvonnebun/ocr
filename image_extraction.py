"""
Step 5: Image Asset Extraction - Crop and deduplicate images
"""
import os
from PIL import Image
from typing import List, Dict
import config
import utils


def extract_image_assets(image_path: str, image_regions: List[Dict], 
                        output_dir: str, page_idx: int) -> List[Dict]:
    """
    Extract image assets by cropping regions from page image.
    
    Args:
        image_path: Path to full page image
        image_regions: List of dicts with 'bbox_px'
        output_dir: Directory to save extracted images
        page_idx: Page index for naming
    
    Returns:
        List of dicts with:
        {
            'image_path': str,
            'bbox_px': [x0, y0, x1, y1]
        }
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if not image_regions:
        return []
    
    # Deduplicate regions on same page
    bboxes = [region['bbox_px'] for region in image_regions]
    keep_indices = utils.deduplicate_bboxes(bboxes, config.IOU_THRESHOLD)
    
    # Load page image
    img = Image.open(image_path)
    width, height = img.size
    
    extracted_images = []
    for idx, region_idx in enumerate(keep_indices):
        region = image_regions[region_idx]
        bbox = region['bbox_px']
        
        # Crop image
        x0, y0, x1, y1 = [int(coord) for coord in bbox]
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(width, x1)
        y1 = min(height, y1)
        
        if x1 <= x0 or y1 <= y0:
            continue
        
        cropped = img.crop((x0, y0, x1, y1))
        
        # Save cropped image
        image_filename = f"page_{page_idx:04d}_image_{idx:04d}.png"
        image_path_out = os.path.join(output_dir, image_filename)
        cropped.save(image_path_out, "PNG")
        
        extracted_images.append({
            'image_path': image_path_out,
            'bbox_px': bbox
        })
    
    return extracted_images


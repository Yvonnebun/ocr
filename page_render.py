"""
Step 1: Page Render - PDF to image conversion
"""
import os
from pdf2image import convert_from_path
from PIL import Image
from typing import List, Tuple
import config


def render_pdf_pages(pdf_path: str, output_dir: str = None) -> List[Tuple[str, int, int]]:
    """
    Render PDF pages to images.
    
    Args:
        pdf_path: Path to PDF file
        output_dir: Directory to save rendered images (default: config.RENDER_DIR)
    
    Returns:
        List of tuples: (image_path, width_px, height_px) for each page
    """
    if output_dir is None:
        output_dir = config.RENDER_DIR
    
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Convert PDF to images (DPI=200 for good quality)
        print(f"  Converting PDF to images (this may take a while)...")
        images = convert_from_path(pdf_path, dpi=200)
        print(f"  Converted {len(images)} pages")
    except Exception as e:
        raise RuntimeError(f"Failed to convert PDF to images: {e}. Make sure Poppler is installed and in PATH.")
    
    page_info = []
    for idx, img in enumerate(images):
        try:
            # Save as PNG
            image_path = os.path.join(output_dir, f"page_{idx:04d}.png")
            img.save(image_path, "PNG")
            
            width, height = img.size
            page_info.append((image_path, width, height))
        except Exception as e:
            print(f"  WARNING: Failed to save page {idx}: {e}")
            # Continue with other pages
    
    if not page_info:
        raise RuntimeError("No pages were successfully rendered")
    
    return page_info


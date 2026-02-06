"""
Step 1: Page Render - PDF to image conversion
"""
import os
import fitz
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
        print("  Converting PDF to images (this may take a while)...")
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise RuntimeError(f"Failed to open PDF: {e}.") from e

    page_info = []
    for idx, page in enumerate(doc):
        try:
            zoom = config.RENDER_DPI / 72.0
            matrix = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            image_path = os.path.join(output_dir, f"page_{idx:04d}.png")
            pix.save(image_path)
            page_info.append((image_path, pix.width, pix.height))
        except Exception as e:
            print(f"  WARNING: Failed to save page {idx}: {e}")
            continue

    doc.close()
    
    if not page_info:
        raise RuntimeError("No pages were successfully rendered")
    
    return page_info

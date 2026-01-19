"""
Main Pipeline for PDF Image-Text Separation
"""
import os
import json
import sys
import traceback
from typing import Dict, List
import config
from page_render import render_pdf_pages
from layout_detect import detect_layout, filter_figure_blocks
from region_refiner import refine_all_candidates
from image_extraction import extract_image_assets
from image_ocr import ocr_all_images
from native_text import extract_native_text, filter_text_excluding_images
from scanned_page import ocr_non_image_regions
from page_gate import page_has_floorplan_keyword
from run_logger import init_run_logger, get_run_logger
import utils
from caption_extract import (
    extract_captions_from_native,
    extract_captions_from_ocr,
    filter_captions_from_text
)


def process_pdf(pdf_path: str, output_dir: str = None) -> Dict:
    """
    Process PDF through the complete pipeline.
    
    Args:
        pdf_path: Path to PDF file
        output_dir: Output directory (default: config.OUTPUT_DIR)
    
    Returns:
        Result dictionary matching result.json format
    """
    try:
        if output_dir is None:
            output_dir = config.OUTPUT_DIR

        # codex update: initialize run logger for diagnostics
        run_logger = init_run_logger(output_dir)
        run_logger.log_event("run_start", {"pdf_path": pdf_path, "output_dir": output_dir})
        
        print(f"Output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(config.IMAGE_DIR, exist_ok=True)
        os.makedirs(config.RENDER_DIR, exist_ok=True)
        
        # Step 1: Page Render
        print("Step 1: Rendering PDF pages...")
        try:
            page_info = render_pdf_pages(pdf_path, config.RENDER_DIR)
            page_count = len(page_info)
            print(f"  Rendered {page_count} pages")
        except Exception as e:
            print(f"ERROR in Step 1 (Page Render): {e}")
            traceback.print_exc()
            raise
        
        result = {
            "meta": {
                "page_count": page_count
            },
            "text_content": "",
            "pages": []
        }
        
        all_text_content = []
        
        # Process each page
        for page_idx, (image_path, width_px, height_px) in enumerate(page_info):
            try:
                print(f"Processing page {page_idx + 1}/{page_count}...")
                run_logger = get_run_logger()
                if run_logger:
                    run_logger.increment("pages_processed")
                    run_logger.log_event("page_start", {"page_idx": page_idx, "image_path": image_path})
                
                page_result = {
                    "page_idx": page_idx,
                    "flags": {
                        "is_scanned": False,
                        "page_keyword_gate": True
                    },
                    "images": [],
                    "captions": []
                }

                # Step 1.5: Page keyword gate (native text + OCR)
                # codex update: gate pages before layout detection
                print(f"  Step 1.5: Page keyword gate...")
                try:
                    native_text_blocks, has_native_text = extract_native_text(
                        pdf_path, page_idx, width_px, height_px
                    )
                    page_has_kw, force_keep, gate_source = page_has_floorplan_keyword(
                        image_path, native_text_blocks
                    )
                    page_result["flags"]["page_keyword_gate"] = page_has_kw
                    page_result["flags"]["page_keyword_gate_source"] = gate_source
                    page_result["flags"]["page_keyword_force_keep"] = force_keep
                    if run_logger:
                        run_logger.log_event(
                            "page_gate",
                            {
                                "page_idx": page_idx,
                                "passed": page_has_kw,
                                "force_keep": force_keep,
                                "source": gate_source,
                            }
                        )
                except Exception as e:
                    print(f"    WARNING in Step 1.5 (Page Gate): {e}")
                    traceback.print_exc()
                    native_text_blocks = []
                    has_native_text = False
                    page_result["flags"]["page_keyword_gate"] = True
                    page_result["flags"]["page_keyword_force_keep"] = False
                    page_result["flags"]["page_keyword_gate_source"] = "gate_error"

                if not page_result["flags"]["page_keyword_gate"]:
                    print(f"  Page {page_idx + 1} skipped by keyword gate")
                    if run_logger:
                        run_logger.log_event("page_skipped", {"page_idx": page_idx, "reason": "keyword_gate"})
                    result["pages"].append(page_result)
                    continue

                # Step 2: Layout Detect
                print(f"  Step 2: Detecting layout...")
                try:
                    layout_blocks = detect_layout(image_path)
                    print(f"    Found {len(layout_blocks)} layout blocks")
                except Exception as e:
                    print(f"    WARNING in Step 2 (Layout Detect): {e}")
                    traceback.print_exc()
                    layout_blocks = []
                
                # Step 3: Image Candidates (only Figure)
                print(f"  Step 3: Filtering image candidates...")
                figure_blocks = filter_figure_blocks(layout_blocks)
                print(f"    Found {len(figure_blocks)} figure candidates")

                # Step 4: Region Refiner
                print(f"  Step 4: Refining regions...")
                if figure_blocks:
                    try:
                        candidate_bboxes = [block["bbox_px"] for block in figure_blocks]
                        candidate_bboxes = utils.preprocess_candidates(
                            candidate_bboxes,
                            width_px,
                            height_px,
                            min_w=config.CANDIDATE_MIN_W,
                            min_h=config.CANDIDATE_MIN_H,
                            overlap_th=config.CANDIDATE_OVERLAP_TH,
                            min_area_ratio=config.CANDIDATE_MIN_AREA_RATIO,
                            sidebar_params=config.SIDEBAR_PARAMS,
                        )
                        refined_candidates = [{"bbox_px": bbox, "source": "preprocess"} for bbox in candidate_bboxes]
                        refine_result = refine_all_candidates(image_path, refined_candidates)
                        image_regions_final = refine_result['image_regions_final']
                        text_regions_override = refine_result['text_regions_override']
                        uncertain_regions = refine_result['uncertain']
                        
                        print(f"    Refined to {len(image_regions_final)} image regions")
                        print(f"    {len(text_regions_override)} regions overridden as text")
                        print(f"    {len(uncertain_regions)} uncertain regions (not used in MVP)")
                        
                        # MVP: uncertain regions are ignored (not output, not used)
                        # They don't go into image_regions_final or text_regions_override
                    except Exception as e:
                        print(f"    WARNING in Step 4 (Region Refiner): {e}")
                        traceback.print_exc()
                        image_regions_final = []
                        text_regions_override = []
                else:
                    image_regions_final = []
                    text_regions_override = []
                
                # Step 5: Image Asset Extraction
                print(f"  Step 5: Extracting image assets...")
                image_bboxes = [region['bbox_px'] for region in image_regions_final]
                try:
                    extracted_images = extract_image_assets(
                        image_path,
                        image_regions_final,
                        config.IMAGE_DIR,
                        page_idx,
                        force_keep=page_result["flags"].get("page_keyword_force_keep", False),
                    )
                    print(f"    Extracted {len(extracted_images)} images")
                    if run_logger:
                        run_logger.record_images(extracted_images)
                except Exception as e:
                    print(f"    WARNING in Step 5 (Image Extraction): {e}")
                    traceback.print_exc()
                    extracted_images = []
                
                # Step 6: Image OCR
                print(f"  Step 6: Running OCR on images...")
                try:
                    images_with_ocr = ocr_all_images(extracted_images)
                    page_result["images"] = images_with_ocr
                    print(f"    OCR completed for {len(images_with_ocr)} images")
                except Exception as e:
                    print(f"    WARNING in Step 6 (Image OCR): {e}")
                    traceback.print_exc()
                    page_result["images"] = extracted_images
                
                # Step 7: Native Text Extraction
                # codex update: reuse native text from gate step
                print(f"  Step 7: Using native text from gate step...")
                print(f"    Extracted {len(native_text_blocks)} native text blocks, has_native={has_native_text}")
                
                # Step 8: Text Excluding Image Regions
                print(f"  Step 8: Filtering text excluding image regions...")
                ocr_text_blocks = []  # Initialize for caption extraction
                if native_text_blocks:
                    try:
                        clean_text_blocks = filter_text_excluding_images(
                            native_text_blocks, image_bboxes
                        )
                        print(f"    Filtered to {len(clean_text_blocks)} text blocks")
                    except Exception as e:
                        print(f"    WARNING in Step 8 (Text Filtering): {e}")
                        traceback.print_exc()
                        clean_text_blocks = native_text_blocks
                else:
                    clean_text_blocks = []
                
                # Step 9: Scanned Page Handling
                print(f"  Step 9: Handling scanned pages...")
                if not has_native_text:
                    # Scanned page: use OCR on non-image regions
                    page_result["flags"]["is_scanned"] = True
                    try:
                        ocr_text_blocks = ocr_non_image_regions(image_path, image_bboxes)
                        clean_text_blocks = ocr_text_blocks
                        print(f"    OCR extracted {len(ocr_text_blocks)} text blocks from scanned page")
                    except Exception as e:
                        print(f"    WARNING in Step 9 (Scanned Page OCR): {e}")
                        traceback.print_exc()
                        ocr_text_blocks = []
                        clean_text_blocks = []
                
                # Caption Extraction
                print(f"  Extracting captions...")
                try:
                    if has_native_text:
                        captions = extract_captions_from_native(native_text_blocks)
                        # Filter captions from text blocks
                        clean_text_blocks = filter_captions_from_text(clean_text_blocks, captions)
                    else:
                        # Use OCR to find captions near images
                        captions = extract_captions_from_ocr(ocr_text_blocks, image_bboxes)
                        # Filter captions from OCR text blocks
                        clean_text_blocks = filter_captions_from_text(clean_text_blocks, captions)
                    print(f"    Found {len(captions)} captions")
                except Exception as e:
                    print(f"    WARNING in Caption Extraction: {e}")
                    traceback.print_exc()
                    captions = []
                
                page_result["captions"] = captions
                
                # Collect text content
                page_text_parts = []
                for block in clean_text_blocks:
                    text = block.get('text', '').strip()
                    if text:
                        page_text_parts.append(text)
                
                page_text = " ".join(page_text_parts)
                all_text_content.append(page_text)
                
                result["pages"].append(page_result)
                print(f"  Page {page_idx + 1} completed")
                if run_logger:
                    run_logger.log_event("page_complete", {"page_idx": page_idx})
                
            except Exception as e:
                print(f"ERROR processing page {page_idx + 1}: {e}")
                traceback.print_exc()
                # Continue with next page
                result["pages"].append({
                    "page_idx": page_idx,
                    "flags": {"is_scanned": False},
                    "images": [],
                    "captions": [],
                    "error": str(e)
                })
    
        # Combine all text content
        result["text_content"] = " ".join(all_text_content)

        # codex update: finalize run logger
        run_logger = get_run_logger()
        if run_logger:
            run_logger.log_event("run_complete", {"pages": page_count})
            run_logger.finalize()
        
        return result
        
    except Exception as e:
        print(f"FATAL ERROR in process_pdf: {e}")
        traceback.print_exc()
        raise


def save_result(result: Dict, output_path: str = None):
    """
    Save result to JSON file.
    
    Args:
        result: Result dictionary
        output_path: Output file path (default: output/result.json)
    """
    if output_path is None:
        output_path = os.path.join(config.OUTPUT_DIR, "result.json")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"Result saved to: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <pdf_path> [output_dir]")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        result = process_pdf(pdf_path, output_dir)
        save_result(result)
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)

"""
Step 2: Layout Detection via HTTP call to Layout Service

This module calls a remote layout-service (running in Linux/Docker) to detect
layout blocks. The service uses Prima (Detectron2 via LayoutParser) which is
not available on Windows.
"""
import requests
from typing import List, Dict
import config
import os
import time


def _convert_to_shared_path(local_path: str) -> str:
    """
    Convert local file path to shared volume path (container absolute path).
    
    Args:
        local_path: Local file path (e.g., "output/renders/page_0000.png")
    
    Returns:
        Shared volume absolute path (e.g., "/app/shared_data/output/renders/page_0000.png")
    """
    # If already an absolute path starting with shared volume root, return as-is
    if local_path.startswith(config.SHARED_VOLUME_ROOT):
        return local_path
    
    # Normalize local path (handle both relative and absolute)
    if os.path.isabs(local_path):
        # Extract relative part from absolute path
        # Assume local path structure matches shared volume structure
        # For Windows: C:\Users\...\output\renders\page_0000.png
        # Extract: output/renders/page_0000.png
        parts = local_path.replace('\\', '/').split('/')
        # Find 'output' or similar marker
        try:
            output_idx = next(i for i, p in enumerate(parts) if p in [ 'output','renders'])
            relative_parts = parts[output_idx:]
            relative_path = '/'.join(relative_parts)
        except StopIteration:
            # Fallback: use filename only
            relative_path = os.path.basename(local_path)
    else:
        relative_path = local_path.replace('\\', '/')


    if relative_path.startswith("output/"):
        relative_path = relative_path[len("output/"):]

    # Join with shared volume root
    shared_path = os.path.join(config.SHARED_VOLUME_ROOT, relative_path).replace('\\', '/')
    return shared_path


def detect_layout(image_path: str) -> List[Dict]:
    """
    Detect layout blocks by calling layout-service via HTTP.
    
    Args:
        image_path: Path to rendered page image (local or shared volume path)
    
    Returns:
        List of layout blocks, each with:
        {
            'type': str,  # 'Figure', 'Table', 'Text', 'List', etc.
            'bbox_px': [x0, y0, x1, y1],
            'score': float or None
        }
    """
    # Convert to shared volume path
    shared_path = _convert_to_shared_path(image_path)
    
    # Prepare request
    url = f"{config.LAYOUT_SERVICE_URL}/predict"
    payload = {"image_path": shared_path}
    
    # HTTP request with retry logic
    max_retries = config.LAYOUT_MAX_RETRIES
    connect_timeout = config.LAYOUT_CONNECT_TIMEOUT
    read_timeout = config.LAYOUT_READ_TIMEOUT
    
    for attempt in range(max_retries + 1):
        try:
            print(f"  Calling layout-service: {url}")
            print(f"    Image path: {shared_path}")
            if attempt > 0:
                print(f"    Retry attempt {attempt}/{max_retries}")
            
            response = requests.post(
                url,
                json=payload,
                timeout=(connect_timeout, read_timeout)
            )
            
            # Check HTTP status
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            
            # Extract blocks from response
            # Compatible with different field names: label/type, bbox/bbox_px
            blocks_data = result.get('blocks', [])
            
            # Normalize to our format
            blocks = []
            for block_data in blocks_data:
                # Get type (try 'label' first, then 'type')
                block_type = block_data.get('label') or block_data.get('type', '')
                if not block_type:
                    continue
                
                # Get bbox (try 'bbox_px' first, then 'bbox')
                bbox = block_data.get('bbox_px') or block_data.get('bbox')
                if not bbox or len(bbox) < 4:
                    continue
                
                # Get score
                score = block_data.get('score')
                if score is not None:
                    try:
                        score = float(score)
                    except (ValueError, TypeError):
                        score = None
                
                blocks.append({
                    'type': str(block_type),
                    'bbox_px': [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                    'score': score
                })
            
            print(f"  Layout-service returned {len(blocks)} blocks")
            
            # DEBUG: Count block types
            if blocks:
                type_counts = {}
                for block in blocks:
                    block_type = block.get('type', 'Unknown')
                    type_counts[block_type] = type_counts.get(block_type, 0) + 1
                print(f"  DEBUG: Block type counts: {type_counts}")
            
            return blocks
            
        except requests.exceptions.Timeout as e:
            if attempt < max_retries:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"  Timeout error: {e}")
                print(f"  Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            else:
                print(f"  ERROR: Layout-service timeout after {max_retries + 1} attempts")
                print(f"    Connect timeout: {connect_timeout}s, Read timeout: {read_timeout}s")
                return []
        
        except requests.exceptions.ConnectionError as e:
            if attempt < max_retries:
                wait_time = 2 ** attempt
                print(f"  Connection error: {e}")
                print(f"  Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            else:
                print(f"  ERROR: Cannot connect to layout-service at {config.LAYOUT_SERVICE_URL}")
                print(f"    Please ensure layout-service is running and accessible")
                return []
        
        except requests.exceptions.HTTPError as e:
            print(f"  ERROR: Layout-service HTTP error: {e}")
            print(f"    Status code: {response.status_code if 'response' in locals() else 'N/A'}")
            try:
                error_detail = response.json() if 'response' in locals() else {}
                print(f"    Error detail: {error_detail}")
            except:
                pass
            return []
        
        except Exception as e:
            print(f"  ERROR: Unexpected error calling layout-service: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    return []


def filter_figure_blocks(layout_blocks: List[Dict]) -> List[Dict]:
    """
    Filter layout blocks to only include Figure type.
    Strict matching: only "Figure" type (case-insensitive).
    
    Args:
        layout_blocks: List of all layout blocks
    
    Returns:
        List of Figure blocks only
    """
    figure_blocks = []
    all_types_seen = set()
    
    for block in layout_blocks:
        block_type = str(block.get('type', '')).strip()
        all_types_seen.add(block_type)
        
        # Strict match: only "Figure" (case-insensitive)
        if block_type.lower() in {"image", "figure", "picture"}:
            figure_blocks.append(block)
    
    # DEBUG: Show what types we actually got
    if len(figure_blocks) == 0 and len(layout_blocks) > 0:
        print(f"  DEBUG: No 'Figure' blocks found. All types seen: {sorted(all_types_seen)}")
        print(f"  DEBUG: This might indicate:")
        print(f"    - Model didn't detect figures")
        print(f"    - Type name is different (e.g., 'FIGURE', 'Image', number)")
        print(f"    - Figures were classified as other types (Table/List)")
    
    return figure_blocks

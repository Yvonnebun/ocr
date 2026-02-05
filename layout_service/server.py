"""
Layout Service - HTTP server for layout detection using Prima (Detectron2 via LayoutParser)

This service runs in Linux/Docker environment where detectron2 is available.
"""
import os
import sys
import shutil
import yaml
from flask import Flask, request, jsonify
from flask_cors import CORS
import layoutparser as lp
from PIL import Image
import numpy as np
from typing import List, Dict
import traceback

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Global model singleton
_layout_model = None

def _xyxy_from_any(obj):
    """
    Return [x1,y1,x2,y2] from LayoutParser Rectangle/Coordinates or tuple/list/ndarray.
    """
    if obj is None:
        return None

    # LayoutParser Rectangle or Coordinates with x_1...
    if hasattr(obj, "x_1") and hasattr(obj, "y_1") and hasattr(obj, "x_2") and hasattr(obj, "y_2"):
        return [float(obj.x_1), float(obj.y_1), float(obj.x_2), float(obj.y_2)]

    # Some objects expose .coordinates (Rectangle -> Coordinates or tuple)
    if hasattr(obj, "coordinates"):
        return _xyxy_from_any(obj.coordinates)

    # tuple/list/ndarray
    if isinstance(obj, (tuple, list, np.ndarray)):
        if len(obj) >= 4 and all(isinstance(v, (int, float, np.number)) for v in obj[:4]):
            return [float(obj[0]), float(obj[1]), float(obj[2]), float(obj[3])]

        # sometimes it's [[x1,y1],[x2,y2]] or polygon-like
        if len(obj) == 2 and all(isinstance(p, (tuple, list, np.ndarray)) and len(p) == 2 for p in obj):
            x1, y1 = obj[0]
            x2, y2 = obj[1]
            return [float(x1), float(y1), float(x2), float(y2)]

    return None


def _bbox_from_block(block):
    """
    Try common LayoutParser shapes:
    - block (TextBlock/LayoutElement) may have .block (Rectangle), .coordinates, etc.
    """
    # best-effort extraction in typical priority order
    for candidate in (
        getattr(block, "block", None),
        getattr(block, "coordinates", None),
        block,
    ):
        bbox = _xyxy_from_any(candidate)
        if bbox is not None:
            return bbox
    return None

def get_layout_model():
    """Get or create the layout detection model (singleton)."""
    global _layout_model
    if _layout_model is None:
        def _load_model() -> lp.Detectron2LayoutModel:
            config_path = "lp://PrimaLayout/mask_rcnn_R_50_FPN_3x/config"
            model_path = "lp://PrimaLayout/mask_rcnn_R_50_FPN_3x/model"
            return lp.Detectron2LayoutModel(
                config_path,
                model_path,
                label_map={1:"Text", 2:"Image", 3:"Table", 4:"Maths", 5:"Separator", 6:"Other"},
                extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
            )

        try:
            print("Loading Prima layout model...")
            _layout_model = _load_model()
            print("Layout model loaded successfully")
        except yaml.scanner.ScannerError as e:
            print(f"WARNING: Layout model config parse failed: {e}")
            cache_dir = os.path.expanduser("~/.torch/iopath_cache")
            print(f"Clearing iopath cache: {cache_dir}")
            shutil.rmtree(cache_dir, ignore_errors=True)
            try:
                _layout_model = _load_model()
                print("Layout model loaded successfully after cache clear")
            except Exception as inner_exc:
                print(f"ERROR: Failed to load layout model after cache clear: {inner_exc}")
                traceback.print_exc()
                raise
        except Exception as e:
            print(f"ERROR: Failed to load layout model: {e}")
            traceback.print_exc()
            raise
    return _layout_model


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    try:
        # Try to get model (will load if not loaded)
        model = get_layout_model()
        return jsonify({
            "status": "ok",
            "service": "layout-service",
            "detectron2_available": lp.is_detectron2_available()
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict layout blocks from image.
    
    Request JSON:
        {
            "image_path": "/app/shared_data/output/renders/page_0000.png"
        }
    
    Response JSON:
        {
            "blocks": [
                {
                    "label": "Figure",
                    "bbox": [100, 200, 500, 600],
                    "score": 0.87
                },
                ...
            ]
        }
    """
    try:
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        image_path = data.get('image_path')
        if not image_path:
            return jsonify({"error": "Missing 'image_path' in request"}), 400
        
        # Check if file exists
        if not os.path.exists(image_path):
            return jsonify({
                "error": f"Image file not found: {image_path}"
            }), 404
        
        # Load image
        try:
            img = Image.open(image_path)
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img_array = np.array(img)
        except Exception as e:
            return jsonify({
                "error": f"Failed to load image: {e}"
            }), 400
        
        # Get model and detect
        model = get_layout_model()
        layout = model.detect(img_array)
        
        # Convert to response format
        blocks = []
        for block in layout:
            try:
                # Get block type
                block_type = str(block.type)
                
                # Extract bbox coordinates
                bbox_coords = _bbox_from_block(block)
                if bbox_coords is None:
                    # optional debug
                    # print("DEBUG: cannot extract bbox from:", type(block), block)
                    continue
                
                # Extract score
                score = None
                if hasattr(block, 'score'):
                    try:
                        score = float(block.score)
                    except:
                        pass
                elif hasattr(block, '_score'):
                    try:
                        score = float(block._score)
                    except:
                        pass
                
                blocks.append({
                    "label": block_type,  # Use 'label' for compatibility
                    "bbox": bbox_coords,  # Use 'bbox' for compatibility
                    "score": score
                })
            except Exception as e:
                print(f"Warning: Error processing block: {e}")
                continue
        
        return jsonify({
            "blocks": blocks,
            "count": len(blocks)
        })
        
    except Exception as e:
        print(f"ERROR in /predict: {e}")
        traceback.print_exc()
        return jsonify({
            "error": str(e),
            "type": type(e).__name__
        }), 500


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Layout Service')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8001, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Layout Service Starting")
    print("=" * 60)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Debug: {args.debug}")
    print(f"Detectron2 available: {lp.is_detectron2_available()}")
    print("=" * 60)
    
    # Warm up model on startup
    try:
        print("Warming up model...")
        get_layout_model()
        print("Model ready!")
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        sys.exit(1)
    
    app.run(host=args.host, port=args.port, debug=args.debug)

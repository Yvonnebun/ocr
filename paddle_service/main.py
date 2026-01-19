# main.py
"""
PaddleOCR Service ASGI entrypoint (FastAPI + uvicorn).

- /predict : {"image_path": "/app/shared_data/..."}
- /predict : {"blocks": [{"text": "...", "bbox": [...], "score": 0.xx}, ...]}
"""

from typing import List, Dict, Any, Optional
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from paddleocr import PaddleOCR


app = FastAPI()
_ocr_model: Optional[PaddleOCR] = None


class PredictRequest(BaseModel):
    image_path: str


def get_ocr_model() -> PaddleOCR:
    global _ocr_model
    if _ocr_model is None:
        _ocr_model = PaddleOCR(use_angle_cls=True, lang="en")
    return _ocr_model


@app.get("/health")
def health() -> Dict[str, Any]:
    try:
        get_ocr_model()
        return {"status": "ok", "service": "paddle-service"}
    except Exception as e:
        return {"status": "error", "service": "paddle-service", "error": str(e)}


@app.on_event("startup")
def startup_event() -> None:
    print("=" * 60)
    print("Paddle Service starting (FastAPI + uvicorn)")
    print("=" * 60)
    try:
        get_ocr_model()
        print("Model warm-up done, service ready.")
    except Exception as e:
        import traceback

        print(f"ERROR: Failed to load PaddleOCR model: {e}")
        traceback.print_exc()
        raise


def _bbox_from_quad(quad: Any) -> Optional[List[float]]:
    if not isinstance(quad, (list, tuple)) or len(quad) < 4:
        return None
    xs = []
    ys = []
    for pt in quad[:4]:
        if not isinstance(pt, (list, tuple)) or len(pt) < 2:
            return None
        xs.append(float(pt[0]))
        ys.append(float(pt[1]))
    return [min(xs), min(ys), max(xs), max(ys)]


@app.post("/predict")
def predict(req: PredictRequest) -> Dict[str, Any]:
    image_path = req.image_path

    if not image_path:
        raise HTTPException(status_code=400, detail="Missing image_path")
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail=f"Image file not found: {image_path}")

    model = get_ocr_model()

    try:
        result = model.ocr(image_path, cls=True)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to run OCR: {e}")

    entries = []
    if isinstance(result, list):
        if result and isinstance(result[0], list):
            entries = result[0]
        else:
            entries = result

    blocks: List[Dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            continue
        quad = entry[0]
        text_info = entry[1]
        text = None
        score = None
        if isinstance(text_info, (list, tuple)) and len(text_info) >= 1:
            text = str(text_info[0])
            if len(text_info) >= 2:
                try:
                    score = float(text_info[1])
                except Exception:
                    score = None

        bbox = _bbox_from_quad(quad)
        if bbox is None:
            continue

        blocks.append({"text": text, "bbox": bbox, "score": score})

    return {"blocks": blocks, "count": len(blocks)}

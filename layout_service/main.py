# main.py
"""
Layout Service ASGI entrypoint (FastAPI + uvicorn).

- /predict : {"image_path": "/app/shared_data/..."}
- /predict : {"blocks": [{"label": "...", "bbox": [...], "score": 0.xx}, ...]}
"""

from typing import List, Dict, Any, Optional

import os
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image

import server  


app = FastAPI()


class PredictRequest(BaseModel):
    image_path: str  


@app.get("/health")
def health() -> Dict[str, Any]:
    # 这里给一点更有用的信息
    return {"status": "ok", "service": "layout-service"}


@app.on_event("startup")
def startup_event() -> None:

    print("=" * 60)
    print("Layout Service starting (FastAPI + uvicorn)")
    print("=" * 60)
    try:
        server.get_layout_model()
        print("Model warm-up done, service ready.")
    except Exception as e:
        import traceback

        print(f"ERROR: Failed to load layout model: {e}")
        traceback.print_exc()
        raise


@app.post("/predict")
def predict(req: PredictRequest) -> Dict[str, Any]:

    image_path = req.image_path


    if not image_path:
        raise HTTPException(status_code=400, detail="Missing image_path")
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail=f"Image file not found: {image_path}")


    try:
        img = Image.open(image_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        img_arr = np.array(img)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load image: {e}")


    model = server.get_layout_model()

    # 3) 检测 layout
    layout = model.detect(img_arr)

    blocks: List[Dict[str, Any]] = []
    for block in layout:
        try:
            label = str(getattr(block, "type", ""))


            bbox: Optional[List[float]] = server._bbox_from_block(block)  
            if bbox is None:
                continue

            score = None

            for attr in ("score", "_score"):
                if hasattr(block, attr):
                    try:
                        score = float(getattr(block, attr))
                    except Exception:
                        score = None
                    break

            blocks.append({"label": label, "bbox": bbox, "score": score})

        except Exception as e:
            import traceback


            print(f"WARNING: Error processing block {block}: {e}")
            traceback.print_exc()
            continue

    return {"blocks": blocks, "count": len(blocks)}

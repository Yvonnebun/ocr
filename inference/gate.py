from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image


def _resize_bgr_image(image_bgr: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
    # Use Pillow so this gate works even when OpenCV native libs are unavailable.
    image_rgb = image_bgr[:, :, ::-1]
    pil_image = Image.fromarray(image_rgb)
    resized_rgb = np.asarray(pil_image.resize((new_w, new_h), resample=Image.Resampling.BOX))
    return resized_rgb[:, :, ::-1]


def apply_oom_gate(
    image_bgr: np.ndarray,
    max_pixels: int = 8_000_000,
    max_side: int = 4096,
    action: str = "downscale",
) -> Tuple[Optional[np.ndarray], Dict[str, object]]:
    h, w = image_bgr.shape[:2]
    limits = {"max_pixels": int(max_pixels), "max_side": int(max_side), "action": str(action)}
    exceeds_pixels = (w * h) > max_pixels
    exceeds_side = max(w, h) > max_side
    if not (exceeds_pixels or exceeds_side):
        return image_bgr, {
            "status": "ok",
            "reason": "",
            "limits": limits,
            "orig_width": int(w),
            "orig_height": int(h),
            "infer_width": int(w),
            "infer_height": int(h),
            "scale_factor": 1.0,
        }

    if action == "reject":
        reason = "max_pixels" if exceeds_pixels else "max_side"
        return None, {
            "status": "rejected",
            "reason": reason,
            "limits": limits,
            "orig_width": int(w),
            "orig_height": int(h),
            "infer_width": 0,
            "infer_height": 0,
            "scale_factor": 0.0,
        }

    scale = min(
        math.sqrt(max_pixels / float(w * h)),
        max_side / float(max(w, h)),
        1.0,
    )
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = _resize_bgr_image(image_bgr, new_w, new_h)
    reason = "max_pixels" if exceeds_pixels else "max_side"
    return resized, {
        "status": "downscaled",
        "reason": reason,
        "limits": limits,
        "orig_width": int(w),
        "orig_height": int(h),
        "infer_width": int(new_w),
        "infer_height": int(new_h),
        "scale_factor": float(scale),
    }

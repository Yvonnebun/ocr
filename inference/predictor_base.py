from typing import Any, Dict, List, Optional, Protocol

import numpy as np


class Predictor(Protocol):
    def predict(
        self,
        image_bgr: np.ndarray,
        conf: float = 0.25,
        iou: float = 0.7,
        max_det: int = 300,
        classes: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        ...

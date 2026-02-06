from typing import Dict, List, TypedDict


class DetectionTD(TypedDict):
    class_id: int
    class_name: str
    score: float
    bbox_xyxy: List[float]


class ModelImageTD(TypedDict):
    width: int
    height: int


class ModelResultTD(TypedDict):
    model_name: str
    model_version: str
    image: ModelImageTD
    detections: List[DetectionTD]
    polygons: List[Dict[str, object]]
    meta: Dict[str, object]


class BundleImageTD(TypedDict):
    image_id: object
    orig_width: int
    orig_height: int
    infer_width: int
    infer_height: int
    scale_factor: float


class GateLimitsTD(TypedDict):
    max_pixels: int
    max_side: int
    action: str


class GateTD(TypedDict):
    status: str
    reason: str
    limits: GateLimitsTD


class BundleWallTD(TypedDict):
    source_models: List[str]
    merged: bool
    result: ModelResultTD


class BundleRoomTD(TypedDict):
    source_models: List[str]
    postprocessed: bool
    result: ModelResultTD


class BundleWindowTD(TypedDict):
    source_models: List[str]
    count: int


class BundleResultTD(TypedDict):
    model_bundle: str
    image: BundleImageTD
    gate: GateTD
    wall: BundleWallTD
    room: BundleRoomTD
    window: BundleWindowTD
    total_ms: float

from .contracts import BundleResultTD, DetectionTD, ModelResultTD

__all__ = [
    "BundleResultTD",
    "DetectionTD",
    "ModelResultTD",
    "FloorplanPipelinePredictor",
    "UltralyticsYoloPredictor",
    "WallRoomPipelinePredictor",
]


def __getattr__(name: str):
    if name == "FloorplanPipelinePredictor":
        from .pipeline_floorplan import FloorplanPipelinePredictor

        return FloorplanPipelinePredictor
    if name == "UltralyticsYoloPredictor":
        from .yolo_ultralytics import UltralyticsYoloPredictor

        return UltralyticsYoloPredictor
    if name == "WallRoomPipelinePredictor":
        from .pipeline_wall_room import WallRoomPipelinePredictor

        return WallRoomPipelinePredictor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

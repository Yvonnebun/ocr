import numpy as np
import pytest

from inference.pipeline_floorplan import FloorplanPipelinePredictor


class DummyPredictor:
    def __init__(self, name: str) -> None:
        self._name = name

    def predict(self, image_bgr: np.ndarray, **kwargs):  # type: ignore[override]
        h, w = image_bgr.shape[:2]
        return {
            "model_name": self._name,
            "model_version": "test",
            "image": {"width": int(w), "height": int(h)},
            "detections": [
                {
                    "class_id": 0,
                    "class_name": "obj",
                    "score": 0.9,
                    "bbox_xyxy": [1.0, 2.0, 3.0, 4.0],
                }
            ],
            "polygons": [],
            "meta": {},
        }


def test_floorplan_pipeline_output_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    image = np.zeros((640, 480, 3), dtype=np.uint8)
    predictor = FloorplanPipelinePredictor(
        wall_a_weights="wall_a.pt",
        wall_b_weights="wall_b.pt",
        room_weights="room.pt",
        window_weights="window.pt",
    )

    monkeypatch.setattr(predictor, "wall_a", DummyPredictor("wall_a"))
    monkeypatch.setattr(predictor, "wall_b", DummyPredictor("wall_b"))
    monkeypatch.setattr(predictor, "room", DummyPredictor("room"))
    monkeypatch.setattr(predictor, "window", DummyPredictor("window"))

    bundle = predictor.predict_bundle(image)

    assert bundle["model_bundle"] == "floorplan_bundle"
    assert bundle["image"]["orig_width"] == 480
    assert bundle["image"]["orig_height"] == 640
    assert bundle["image"]["infer_width"] == 480
    assert bundle["image"]["infer_height"] == 640
    assert bundle["image"]["scale_factor"] == 1.0
    assert bundle["gate"]["status"] == "ok"
    assert bundle["wall"]["merged"] is False
    assert bundle["room"]["postprocessed"] is False
    assert bundle["window"]["count"] == 1
    assert bundle["total_ms"] >= 0.0


def test_floorplan_pipeline_gate_reject(monkeypatch: pytest.MonkeyPatch) -> None:
    image = np.zeros((9000, 9000, 3), dtype=np.uint8)
    predictor = FloorplanPipelinePredictor(
        wall_a_weights="wall_a.pt",
        wall_b_weights="wall_b.pt",
        room_weights="room.pt",
        window_weights="window.pt",
        max_pixels=1,
        max_side=1,
        gate_action="reject",
    )

    monkeypatch.setattr(predictor, "wall_a", DummyPredictor("wall_a"))
    monkeypatch.setattr(predictor, "wall_b", DummyPredictor("wall_b"))
    monkeypatch.setattr(predictor, "room", DummyPredictor("room"))
    monkeypatch.setattr(predictor, "window", DummyPredictor("window"))

    bundle = predictor.predict_bundle(image)

    assert bundle["gate"]["status"] == "rejected"
    assert bundle["window"]["count"] == 0
    assert bundle["wall"]["result"]["detections"] == []
    assert bundle["room"]["result"]["detections"] == []

import json

from inference.contracts import BundleResultTD


def test_bundle_contract_is_jsonable() -> None:
    bundle: BundleResultTD = {
        "model_bundle": "floorplan_bundle",
        "image": {
            "image_id": None,
            "orig_width": 100,
            "orig_height": 200,
            "infer_width": 100,
            "infer_height": 200,
            "scale_factor": 1.0,
        },
        "gate": {
            "status": "ok",
            "reason": "",
            "limits": {"max_pixels": 8000000, "max_side": 4096, "action": "downscale"},
        },
        "wall": {
            "source_models": ["wall_a", "wall_b"],
            "merged": False,
            "result": {
                "model_name": "yolo",
                "model_version": "v0",
                "image": {"width": 100, "height": 200},
                "detections": [],
                "polygons": [],
                "meta": {},
            },
        },
        "room": {
            "source_models": ["room"],
            "postprocessed": False,
            "result": {
                "model_name": "yolo",
                "model_version": "v0",
                "image": {"width": 100, "height": 200},
                "detections": [],
                "polygons": [],
                "meta": {},
            },
        },
        "window": {
            "source_models": ["window"],
            "count": 0,
        },
        "total_ms": 0.0,
    }

    json.dumps(bundle)

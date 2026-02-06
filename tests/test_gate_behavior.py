import numpy as np

from inference.gate import apply_oom_gate


def test_gate_ok_keeps_scale() -> None:
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    gated, info = apply_oom_gate(image, max_pixels=1_000_000, max_side=1024, action="downscale")
    assert gated is not None
    assert info["status"] == "ok"
    assert info["scale_factor"] == 1.0
    assert info["infer_width"] == 300
    assert info["infer_height"] == 200


def test_gate_downscale() -> None:
    image = np.zeros((4000, 4000, 3), dtype=np.uint8)
    gated, info = apply_oom_gate(image, max_pixels=8_000_000, max_side=3000, action="downscale")
    assert gated is not None
    assert info["status"] == "downscaled"
    assert info["scale_factor"] < 1.0
    assert info["infer_width"] <= 3000
    assert info["infer_height"] <= 3000


def test_gate_reject() -> None:
    image = np.zeros((5000, 5000, 3), dtype=np.uint8)
    gated, info = apply_oom_gate(image, max_pixels=8_000_000, max_side=4096, action="reject")
    assert gated is None
    assert info["status"] == "rejected"

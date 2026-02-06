from __future__ import annotations

import argparse
import json
from pathlib import Path

from door_detect import detect_door_count


def main() -> int:
    parser = argparse.ArgumentParser(description="Run door detection on a single image.")
    parser.add_argument("image_path", type=Path, help="Path to an image file.")
    args = parser.parse_args()

    image_path = args.image_path
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    door_count, stats = detect_door_count(str(image_path))
    print(json.dumps({"image_path": str(image_path), "door_count": door_count, "stats": stats}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

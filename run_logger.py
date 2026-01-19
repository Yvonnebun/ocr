"""
Run logging helpers for pipeline diagnostics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
import os
import time
from typing import Any, Dict, List, Optional


@dataclass
class RunLogger:
    # codex update: capture run-level metrics and artifacts
    output_dir: str
    run_id: str = field(default_factory=lambda: datetime.utcnow().strftime("%Y%m%d_%H%M%S"))
    counters: Dict[str, int] = field(default_factory=lambda: {
        "layout_calls": 0,
        "paddle_calls": 0,
        "pages_processed": 0,
        "images_saved": 0,
    })
    events: List[Dict[str, Any]] = field(default_factory=list)
    start_time: float = field(default_factory=time.monotonic)
    log_path: str = ""
    summary_path: str = ""

    def __post_init__(self) -> None:
        run_dir = os.path.join(self.output_dir, "run_logs", self.run_id)
        os.makedirs(run_dir, exist_ok=True)
        self.log_path = os.path.join(run_dir, "run_log.jsonl")
        self.summary_path = os.path.join(run_dir, "summary.json")

    def log_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        entry = {
            "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "event": event_type,
            "payload": payload,
        }
        self.events.append(entry)
        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def increment(self, key: str, amount: int = 1) -> None:
        self.counters[key] = self.counters.get(key, 0) + amount

    def record_images(self, images: List[Dict[str, Any]]) -> None:
        self.increment("images_saved", len(images))
        self.log_event("images_saved", {"paths": [img.get("image_path") for img in images]})

    def finalize(self) -> None:
        elapsed = time.monotonic() - self.start_time
        summary = {
            "run_id": self.run_id,
            "elapsed_seconds": round(elapsed, 3),
            "counters": self.counters,
            "events": self.events,
        }
        with open(self.summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)


_RUN_LOGGER: Optional[RunLogger] = None


def init_run_logger(output_dir: str) -> RunLogger:
    # codex update: initialize a singleton run logger
    global _RUN_LOGGER
    _RUN_LOGGER = RunLogger(output_dir=output_dir)
    return _RUN_LOGGER


def get_run_logger() -> Optional[RunLogger]:
    # codex update: return the singleton run logger
    return _RUN_LOGGER

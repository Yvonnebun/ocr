"""
HTTP client for PaddleOCR service.
"""
from __future__ import annotations

import os
import time
from typing import Dict, List

import requests

import config
from run_logger import get_run_logger


def _convert_to_shared_path(local_path: str) -> str:
    # codex update: convert local paths to shared volume paths for paddle-service
    if local_path.startswith(config.SHARED_VOLUME_ROOT):
        return local_path
    if os.path.isabs(local_path):
        parts = local_path.replace("\\", "/").split("/")
        try:
            output_idx = next(i for i, p in enumerate(parts) if p in ["output", "renders"])
            relative_parts = parts[output_idx:]
            relative_path = "/".join(relative_parts)
        except StopIteration:
            relative_path = os.path.basename(local_path)
    else:
        relative_path = local_path.replace("\\", "/")

    if relative_path.startswith("output/"):
        relative_path = relative_path[len("output/"):]
    return os.path.join(config.SHARED_VOLUME_ROOT, relative_path).replace("\\", "/")


def paddle_ocr(image_path: str) -> List[Dict]:
    # codex update: call paddle-service for OCR blocks
    shared_path = _convert_to_shared_path(image_path)
    url = f"{config.PADDLE_SERVICE_URL}/predict"
    payload = {"image_path": shared_path}

    run_logger = get_run_logger()
    if run_logger:
        run_logger.increment("paddle_calls")
        run_logger.log_event("paddle_call", {"image_path": shared_path})

    max_retries = config.PADDLE_MAX_RETRIES
    connect_timeout = config.PADDLE_CONNECT_TIMEOUT
    read_timeout = config.PADDLE_READ_TIMEOUT

    for attempt in range(max_retries + 1):
        try:
            response = requests.post(url, json=payload, timeout=(connect_timeout, read_timeout))
            response.raise_for_status()
            result = response.json()
            return result.get("blocks", [])
        except requests.exceptions.Timeout as exc:
            if attempt < max_retries:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"Paddle-service timeout: {exc}") from exc
        except requests.exceptions.ConnectionError as exc:
            if attempt < max_retries:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"Cannot connect to paddle-service at {config.PADDLE_SERVICE_URL}") from exc
        except requests.exceptions.HTTPError as exc:
            response_text = ""
            try:
                response_text = response.text
            except Exception:
                response_text = ""
            detail = f"{exc}"
            if response_text:
                detail = f"{detail} - {response_text.strip()}"
            raise RuntimeError(f"Paddle-service HTTP error: {detail}") from exc

    return []

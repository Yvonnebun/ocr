"""
HTTP client for floorplan inference service.
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List

import requests

import config
from run_logger import get_run_logger


def _convert_to_shared_path(local_path: str) -> str:
    if local_path.startswith(config.SHARED_VOLUME_ROOT):
        return local_path
    if os.path.isabs(local_path):
        parts = local_path.replace("\\", "/").split("/")
        try:
            output_idx = next(i for i, p in enumerate(parts) if p in ["output", "renders"])
            relative_parts = parts[output_idx:]
            relative_path = "/".join(relative_parts)
        except StopIteration:
            raise ValueError(
                f"Cannot map path to shared volume: {local_path}. "
                "Expected a path containing output/ or renders/."
            )
    else:
        relative_path = local_path.replace("\\", "/")

    if relative_path.startswith("output/"):
        relative_path = relative_path[len("output/"):]
    return os.path.join(config.SHARED_VOLUME_ROOT, relative_path).replace("\\", "/")


def infer_polygons(image_path: str) -> Dict[str, Any]:
    shared_path = _convert_to_shared_path(image_path)
    url = f"{config.INFERENCE_SERVICE_URL}/predict"
    payload = {"image_path": shared_path}

    run_logger = get_run_logger()
    if run_logger:
        run_logger.increment("inference_calls")
        run_logger.log_event("inference_call", {"image_path": shared_path})

    max_retries = config.INFERENCE_MAX_RETRIES
    connect_timeout = config.INFERENCE_CONNECT_TIMEOUT
    read_timeout = config.INFERENCE_READ_TIMEOUT

    for attempt in range(max_retries + 1):
        try:
            response = requests.post(url, json=payload, timeout=(connect_timeout, read_timeout))
            if response.status_code in {429} or 500 <= response.status_code < 600:
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
                    continue
                snippet = response.text[:200] if response.text else ""
                raise RuntimeError(
                    f"Inference-service HTTP {response.status_code}: {snippet}"
                )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout as exc:
            if attempt < max_retries:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"Inference-service timeout: {exc}") from exc
        except requests.exceptions.ConnectionError as exc:
            if attempt < max_retries:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"Cannot connect to inference service at {config.INFERENCE_SERVICE_URL}") from exc
        except requests.exceptions.HTTPError as exc:
            snippet = ""
            if exc.response is not None:
                snippet = exc.response.text[:200] if exc.response.text else ""
            raise RuntimeError(f"Inference-service HTTP error: {exc} {snippet}") from exc

    return {}

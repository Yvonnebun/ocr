# Docker usage

This repo does not require Docker for basic unit tests, but using Docker keeps
dependencies (especially Ultralytics + torch) consistent across machines.
The base image includes CUDA so the container can access GPUs when you run with
the NVIDIA container runtime.
PaddleOCR and Detectron2 should be run via their dedicated Docker images rather
than installed locally.

For local (non-Docker) dependency setup and OpenCV `libGL.so.1` troubleshooting,
see `docs/ENV_REQUIREMENTS.md`.

## Build

```bash
docker build -t ocr-pipeline .
```

## Run unit tests

```bash
docker run --rm -it ocr-pipeline pytest tests
```

## Run the floorplan pipeline example

Mount your local files (image + weights) into the container, then run:

```bash
docker run --rm -it \
  --gpus all \
  -v "$PWD:/app" \
  ocr-pipeline \
  python examples/run_floorplan_pipeline.py
```

Update the paths in `examples/run_floorplan_pipeline.py` to point to the mounted
image and weight files before running.【F:examples/run_floorplan_pipeline.py†L1-L41】

# Environment requirements and troubleshooting

This project mixes Python packages with native CV/ML dependencies. If you run tests
outside Docker, make sure both Python and system libraries are installed.

## Python dependencies

Install from the repo root:

```bash
pip install -r requirements.txt
```

## System libraries required by OpenCV (`cv2`)

If you see:

```text
ImportError: libGL.so.1: cannot open shared object file: No such file or directory
```

install the following packages on Debian/Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y libgl1 libglib2.0-0 libsm6 libxext6 libxrender1
```

These packages are now included in all project Dockerfiles:

- root `Dockerfile` (ocr-pipeline)
- `layout_service/Dockerfile`
- `paddle_service/Dockerfile`

## Recommended runtime approach

For reproducible environment setup (especially GPU + detectron2/paddle stacks), use
Docker Compose from the repository root:

```bash
docker compose up --build
```

## Quick verification

After installing dependencies, validate that OpenCV can be imported:

```bash
python -c "import cv2; print(cv2.__version__)"
```

Then run tests:

```bash
pytest -q
```

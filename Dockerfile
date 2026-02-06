FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    DOOR_DETECT_ENABLED=true \
    DOOR_MODEL_PATH=/app/models/door.pt \
    DOOR_CONF_THRESHOLD=0.25 \
    DOOR_IOU_THRESHOLD=0.45 \
    DOOR_CLASS_NAMES=door \
    DOOR_MIN_COUNT=1 \
    FLOORPLAN_WALL_A_WEIGHTS=/app/models/wall_a.pt \
    FLOORPLAN_WALL_B_WEIGHTS=/app/models/wall_b.pt \
    FLOORPLAN_ROOM_WEIGHTS=/app/models/room.pt \
    FLOORPLAN_WINDOW_WEIGHTS=/app/models/window.pt

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN mkdir -p /app/shared_data

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

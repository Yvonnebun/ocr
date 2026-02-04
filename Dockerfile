FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

RUN mkdir -p /app/models

# Default model paths (mount your weights into /app/models at runtime)
ENV INFERENCE_WALL_A_WEIGHTS=/app/models/wall_a.pt \
    INFERENCE_WALL_B_WEIGHTS=/app/models/wall_b.pt \
    INFERENCE_ROOM_WEIGHTS=/app/models/room.pt \
    INFERENCE_WINDOW_WEIGHTS=/app/models/window.pt

COPY . /app

FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

RUN apt-get update -o Acquire::ForceIPv4=true && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# Detectron2: official prebuilt wheel (Linux only)
RUN --mount=type=cache,target=/root/.cache/pip \
    python -c "import torch; print('torch=', torch.__version__, 'cuda=', torch.version.cuda)" && \
    python -m pip install detectron2 -f \
      https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html

COPY . .

EXPOSE 8001
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]



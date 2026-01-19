# Paddle Service

HTTP OCR service using PaddleOCR (FastAPI + uvicorn).

## Docker (recommended)

```bash
cd paddle_service
docker compose up -d
```

The container runs on port `8002`.

## Local (Linux/WSL2)

```bash
cd paddle_service
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8002
```

## API

### Health Check

```bash
GET /health
```

### Predict

```bash
POST /predict
Content-Type: application/json

{
  "image_path": "/app/shared_data/output/renders/page_0000.png"
}
```

Response:
```json
{
  "blocks": [
    {
      "text": "Example",
      "bbox": [100, 200, 500, 260],
      "score": 0.92
    }
  ],
  "count": 1
}
```

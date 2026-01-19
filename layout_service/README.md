# Layout Service

HTTP service for layout detection using Prima (Detectron2 via LayoutParser).

## 功能

- 提供 `/predict` 接口进行布局检测
- 使用 Detectron2 + LayoutParser (Prima 模型)
- 支持健康检查 `/health`

## 快速启动

### 方法 1: Docker Compose（推荐）

```bash
cd layout_service
docker-compose up -d
```

### 方法 2: Docker Build & Run

```bash
cd layout_service
docker build -t layout-service .
docker run -d --name layout-service -p 8001:8001 \
  -v "$(pwd)/../output:/app/shared_data/output:ro" \
  layout-service
```

### 方法 3: 直接运行（Linux/WSL2）

```bash
cd layout_service
pip install -r requirements.txt
# Install detectron2 separately if needed
python -m layout_service.server --port 8001
```

## API 文档

### Health Check

```bash
GET /health
```

Response:
```json
{
  "status": "ok",
  "service": "layout-service",
  "detectron2_available": true
}
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
      "label": "Figure",
      "bbox": [100, 200, 500, 600],
      "score": 0.87
    },
    {
      "label": "Text",
      "bbox": [50, 50, 400, 150],
      "score": 0.92
    }
  ],
  "count": 2
}
```

## 配置

### 环境变量

- `SHARED_VOLUME_ROOT`: 共享卷根路径（默认: `/app/shared_data`）

### 命令行参数

- `--host`: 绑定主机（默认: `0.0.0.0`）
- `--port`: 绑定端口（默认: `8001`）
- `--debug`: 启用调试模式

## 测试

### 测试健康检查

```bash
curl http://localhost:8001/health
```

### 测试预测接口

```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"image_path": "/app/shared_data/output/renders/page_0000.png"}'
```

### 从 Windows 客户端测试

```bash
# 在 Windows 上运行
python test_layout_service_client.py output/renders/page_0000.png
```

## 故障排除

### Detectron2 不可用

如果看到 `detectron2_available: false`，检查：
1. 是否在 Linux/Docker 环境中运行
2. detectron2 是否正确安装

### 模型加载失败

检查日志查看具体错误。常见原因：
- 网络问题（首次运行需要下载模型）
- 内存不足
- detectron2 版本不兼容

### 路径问题

确保：
1. Docker volume mount 正确
2. 共享卷路径在容器内可访问
3. 文件权限正确

## 性能

- 首次请求较慢（模型加载）
- 后续请求较快（模型已加载）
- 建议在启动时 warm-up 模型（已实现）


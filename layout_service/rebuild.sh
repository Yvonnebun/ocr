#!/bin/bash
# Rebuild layout-service with no cache

echo "Cleaning up old containers and images..."
docker compose down layout-service 2>/dev/null || true
docker rmi layout-service_layout-service 2>/dev/null || true

echo "Building layout-service (no cache)..."
docker compose build --no-cache layout-service

echo "Starting layout-service..."
docker compose up -d layout-service

echo "Checking status..."
docker compose ps layout-service

echo ""
echo "To view logs:"
echo "  docker compose logs -f layout-service"


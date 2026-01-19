#!/bin/bash
# Rebuild paddle-service with no cache

echo "Cleaning up old containers and images..."
docker compose down paddle-service 2>/dev/null || true
docker rmi paddle-service_paddle-service 2>/dev/null || true

echo "Building paddle-service (no cache)..."
docker compose build --no-cache paddle-service

echo "Starting paddle-service..."
docker compose up -d paddle-service

echo "Checking status..."
docker compose ps paddle-service

echo ""
echo "To view logs:"
echo "  docker compose logs -f paddle-service"

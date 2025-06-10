#!/bin/bash

set -e

echo "🚀 Deploying Synthetic Data Platform..."

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose is required for deployment"
    exit 1
fi

# Build images
echo "🔨 Building Docker images..."
docker-compose build

# Start services
echo "🌟 Starting services..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 10

# Health check
echo "🔍 Performing health checks..."
health_check() {
    local service_name=$1
    local url=$2
    local max_attempts=30
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        if curl -f -s $url > /dev/null; then
            echo "✅ $service_name is healthy"
            return 0
        fi
        echo "⏳ Waiting for $service_name... (attempt $attempt/$max_attempts)"
        sleep 2
        ((attempt++))
    done

    echo "❌ $service_name health check failed"
    return 1
}

health_check "API Server" "http://localhost:8000/health"
health_check "Web Interface" "http://localhost:8001"

echo "✅ Deployment completed successfully!"
echo ""
echo "🌐 Services are running:"
echo "  API Server:     http://localhost:8000"
echo "  Web Interface:  http://localhost:8001"
echo "  API Docs:       http://localhost:8000/docs"
echo ""
echo "📊 Monitor with:"
echo "  docker-compose logs -f"
echo "  docker-compose ps"

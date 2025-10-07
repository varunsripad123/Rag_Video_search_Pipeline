#!/bin/bash
# Deployment script for Video Search Pipeline

set -e

echo "=========================================="
echo "Video Search Pipeline - Deployment Script"
echo "=========================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose is not installed. Please install docker-compose first."
    exit 1
fi

# Parse arguments
MODE=${1:-"basic"}  # basic, full, production

echo ""
echo "🚀 Deployment Mode: $MODE"
echo ""

case $MODE in
    "basic")
        echo "📦 Building basic deployment (API + Redis only)..."
        docker-compose up -d api redis
        ;;
    
    "full")
        echo "📦 Building full deployment (API + Redis + Nginx)..."
        docker-compose up -d api redis nginx
        ;;
    
    "production")
        echo "📦 Building production deployment (all services)..."
        docker-compose up -d
        ;;
    
    "stop")
        echo "🛑 Stopping all services..."
        docker-compose down
        exit 0
        ;;
    
    "restart")
        echo "🔄 Restarting services..."
        docker-compose restart
        exit 0
        ;;
    
    "logs")
        echo "📋 Showing logs..."
        docker-compose logs -f
        exit 0
        ;;
    
    *)
        echo "❌ Invalid mode: $MODE"
        echo ""
        echo "Usage: ./deploy.sh [basic|full|production|stop|restart|logs]"
        echo ""
        echo "Modes:"
        echo "  basic      - API + Redis only"
        echo "  full       - API + Redis + Nginx"
        echo "  production - All services (API + Redis + Nginx + Monitoring)"
        echo "  stop       - Stop all services"
        echo "  restart    - Restart services"
        echo "  logs       - Show logs"
        exit 1
        ;;
esac

echo ""
echo "⏳ Waiting for services to start..."
sleep 10

# Check service health
echo ""
echo "🔍 Checking service health..."

if docker-compose ps | grep -q "api.*Up"; then
    echo "✅ API service is running"
    echo "   URL: http://localhost:8081"
else
    echo "❌ API service failed to start"
    docker-compose logs api
    exit 1
fi

if docker-compose ps | grep -q "redis.*Up"; then
    echo "✅ Redis is running"
else
    echo "⚠️  Redis is not running"
fi

if [[ "$MODE" == "full" || "$MODE" == "production" ]]; then
    if docker-compose ps | grep -q "nginx.*Up"; then
        echo "✅ Nginx is running"
        echo "   URL: http://localhost"
    else
        echo "⚠️  Nginx is not running"
    fi
fi

if [[ "$MODE" == "production" ]]; then
    if docker-compose ps | grep -q "prometheus.*Up"; then
        echo "✅ Prometheus is running"
        echo "   URL: http://localhost:9090"
    fi
    
    if docker-compose ps | grep -q "grafana.*Up"; then
        echo "✅ Grafana is running"
        echo "   URL: http://localhost:3000 (admin/admin)"
    fi
fi

echo ""
echo "=========================================="
echo "✅ Deployment Complete!"
echo "=========================================="
echo ""
echo "📝 Quick Commands:"
echo "  View logs:    docker-compose logs -f api"
echo "  Stop all:     ./deploy.sh stop"
echo "  Restart:      ./deploy.sh restart"
echo "  Check status: docker-compose ps"
echo ""
echo "🔗 Access Points:"
echo "  API:          http://localhost:8081"
echo "  Web UI:       http://localhost:8081/static/index.html"
if [[ "$MODE" == "full" || "$MODE" == "production" ]]; then
    echo "  Nginx:        http://localhost"
fi
if [[ "$MODE" == "production" ]]; then
    echo "  Prometheus:   http://localhost:9090"
    echo "  Grafana:      http://localhost:3000"
fi
echo ""

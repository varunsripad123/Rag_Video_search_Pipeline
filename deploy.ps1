# Memory Search - Production Deployment Script
# Quick deployment to Railway or local testing

Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "Memory Search - Deployment Script" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "Video Search Pipeline - Deployment Script" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Check Docker
try {
    docker --version | Out-Null
} catch {
    Write-Host "❌ Docker is not installed or not running" -ForegroundColor Red
    exit 1
}

# Check docker-compose
try {
    docker-compose --version | Out-Null
} catch {
    Write-Host "❌ docker-compose is not installed" -ForegroundColor Red
    exit 1
}

Write-Host "🚀 Deployment Mode: $Mode" -ForegroundColor Green
Write-Host ""

switch ($Mode) {
    "basic" {
        Write-Host "📦 Building basic deployment (API + Redis only)..." -ForegroundColor Yellow
        docker-compose up -d api redis
    }
    
    "full" {
        Write-Host "📦 Building full deployment (API + Redis + Nginx)..." -ForegroundColor Yellow
        docker-compose up -d api redis nginx
    }
    
    "production" {
        Write-Host "📦 Building production deployment (all services)..." -ForegroundColor Yellow
        docker-compose up -d
    }
    
    "stop" {
        Write-Host "🛑 Stopping all services..." -ForegroundColor Yellow
        docker-compose down
        exit 0
    }
    
    "restart" {
        Write-Host "🔄 Restarting services..." -ForegroundColor Yellow
        docker-compose restart
        exit 0
    }
    
    "logs" {
        Write-Host "📋 Showing logs..." -ForegroundColor Yellow
        docker-compose logs -f
        exit 0
    }
    
    default {
        Write-Host "❌ Invalid mode: $Mode" -ForegroundColor Red
        Write-Host ""
        Write-Host "Usage: .\deploy.ps1 -Mode [basic|full|production|stop|restart|logs]"
        Write-Host ""
        Write-Host "Modes:"
        Write-Host "  basic      - API + Redis only"
        Write-Host "  full       - API + Redis + Nginx"
        Write-Host "  production - All services (API + Redis + Nginx + Monitoring)"
        Write-Host "  stop       - Stop all services"
        Write-Host "  restart    - Restart services"
        Write-Host "  logs       - Show logs"
        exit 1
    }
}

Write-Host ""
Write-Host "⏳ Waiting for services to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# Check service health
Write-Host ""
Write-Host "🔍 Checking service health..." -ForegroundColor Yellow

$services = docker-compose ps

if ($services -match "api.*Up") {
    Write-Host "✅ API service is running" -ForegroundColor Green
    Write-Host "   URL: http://localhost:8081" -ForegroundColor Cyan
} else {
    Write-Host "❌ API service failed to start" -ForegroundColor Red
    docker-compose logs api
    exit 1
}

if ($services -match "redis.*Up") {
    Write-Host "✅ Redis is running" -ForegroundColor Green
} else {
    Write-Host "⚠️  Redis is not running" -ForegroundColor Yellow
}

if ($Mode -eq "full" -or $Mode -eq "production") {
    if ($services -match "nginx.*Up") {
        Write-Host "✅ Nginx is running" -ForegroundColor Green
        Write-Host "   URL: http://localhost" -ForegroundColor Cyan
    } else {
        Write-Host "⚠️  Nginx is not running" -ForegroundColor Yellow
    }
}

if ($Mode -eq "production") {
    if ($services -match "prometheus.*Up") {
        Write-Host "✅ Prometheus is running" -ForegroundColor Green
        Write-Host "   URL: http://localhost:9090" -ForegroundColor Cyan
    }
    
    if ($services -match "grafana.*Up") {
        Write-Host "✅ Grafana is running" -ForegroundColor Green
        Write-Host "   URL: http://localhost:3000 (admin/admin)" -ForegroundColor Cyan
    }
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "✅ Deployment Complete!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "📝 Quick Commands:"
Write-Host "  View logs:    docker-compose logs -f api"
Write-Host "  Stop all:     .\deploy.ps1 -Mode stop"
Write-Host "  Restart:      .\deploy.ps1 -Mode restart"
Write-Host "  Check status: docker-compose ps"
Write-Host ""
Write-Host "🔗 Access Points:"
Write-Host "  API:          http://localhost:8081"
Write-Host "  Web UI:       http://localhost:8081/static/index.html"
if ($Mode -eq "full" -or $Mode -eq "production") {
    Write-Host "  Nginx:        http://localhost"
}
if ($Mode -eq "production") {
    Write-Host "  Prometheus:   http://localhost:9090"
    Write-Host "  Grafana:      http://localhost:3000"
}
Write-Host ""

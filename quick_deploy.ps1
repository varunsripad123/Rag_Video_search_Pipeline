# Memory Search - Quick Deploy Script
# Simple deployment for Windows

Write-Host ""
Write-Host "🚀 Memory Search - Quick Deploy" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Choose an option:" -ForegroundColor Yellow
Write-Host "1. Test locally (localhost:8081)"
Write-Host "2. Deploy to Railway (production)"
Write-Host "3. View documentation"
Write-Host ""

$choice = Read-Host "Enter choice (1-3)"

switch ($choice) {
    "1" {
        Write-Host ""
        Write-Host "🧪 Starting local server..." -ForegroundColor Green
        Write-Host ""
        
        # Check if data exists
        if (-not (Test-Path "data/processed/metadata.json")) {
            Write-Host "⚠️  No processed data found!" -ForegroundColor Yellow
            Write-Host "Run: python run_pipeline.py first" -ForegroundColor Cyan
            Write-Host ""
            exit
        }
        
        Write-Host "✅ Data found" -ForegroundColor Green
        Write-Host "Starting API server..." -ForegroundColor Cyan
        Write-Host ""
        Write-Host "Open in browser: http://localhost:8081" -ForegroundColor Yellow
        Write-Host ""
        
        python run_api.py
    }
    
    "2" {
        Write-Host ""
        Write-Host "🚀 Deploying to Railway..." -ForegroundColor Green
        Write-Host ""
        
        # Check Railway CLI
        try {
            railway --version | Out-Null
        } catch {
            Write-Host "❌ Railway CLI not found!" -ForegroundColor Red
            Write-Host ""
            Write-Host "Install it:" -ForegroundColor Yellow
            Write-Host "npm install -g @railway/cli" -ForegroundColor Cyan
            Write-Host ""
            exit
        }
        
        Write-Host "✅ Railway CLI found" -ForegroundColor Green
        Write-Host ""
        
        # Login
        Write-Host "Logging in to Railway..." -ForegroundColor Cyan
        railway login
        
        # Initialize
        Write-Host "Initializing project..." -ForegroundColor Cyan
        railway init
        
        # Deploy
        Write-Host "Deploying..." -ForegroundColor Cyan
        railway up
        
        Write-Host ""
        Write-Host "✅ Deployment complete!" -ForegroundColor Green
        Write-Host ""
        Write-Host "Get your URL:" -ForegroundColor Yellow
        railway domain
        
        Write-Host ""
        Write-Host "🎉 Your Memory Search is live!" -ForegroundColor Green
        Write-Host ""
    }
    
    "3" {
        Write-Host ""
        Write-Host "📚 Documentation Files:" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "Quick Start:" -ForegroundColor Yellow
        Write-Host "  - READY_TO_DEPLOY.md" -ForegroundColor Green
        Write-Host "  - DEPLOY_NOW.md" -ForegroundColor Green
        Write-Host ""
        Write-Host "Guides:" -ForegroundColor Yellow
        Write-Host "  - COMPLETE_SYSTEM_SUMMARY.md" -ForegroundColor Green
        Write-Host "  - USE_CASES.md" -ForegroundColor Green
        Write-Host "  - FINETUNING_GUIDE.md" -ForegroundColor Green
        Write-Host ""
        Write-Host "Opening READY_TO_DEPLOY.md..." -ForegroundColor Cyan
        Start-Process "READY_TO_DEPLOY.md"
    }
    
    default {
        Write-Host "Invalid choice" -ForegroundColor Red
    }
}

Write-Host ""

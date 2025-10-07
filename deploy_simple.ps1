# Memory Search - Simple Deploy Script

Write-Host ""
Write-Host "Memory Search - Quick Deploy" -ForegroundColor Cyan
Write-Host "============================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Choose an option:" -ForegroundColor Yellow
Write-Host "1. Test locally"
Write-Host "2. Deploy to Railway"
Write-Host ""

$choice = Read-Host "Enter choice (1-2)"

if ($choice -eq "1") {
    Write-Host ""
    Write-Host "Starting local server..." -ForegroundColor Green
    Write-Host ""
    
    if (-not (Test-Path "data/processed/metadata.json")) {
        Write-Host "ERROR: No processed data found!" -ForegroundColor Red
        Write-Host "Run: python run_pipeline.py first" -ForegroundColor Yellow
        Write-Host ""
        exit
    }
    
    Write-Host "Data found. Starting API..." -ForegroundColor Green
    Write-Host ""
    Write-Host "Open: http://localhost:8081" -ForegroundColor Cyan
    Write-Host ""
    
    python run_api.py
}
elseif ($choice -eq "2") {
    Write-Host ""
    Write-Host "Deploying to Railway..." -ForegroundColor Green
    Write-Host ""
    
    try {
        railway --version | Out-Null
        Write-Host "Railway CLI found" -ForegroundColor Green
    }
    catch {
        Write-Host "ERROR: Railway CLI not found!" -ForegroundColor Red
        Write-Host ""
        Write-Host "Install it:" -ForegroundColor Yellow
        Write-Host "npm install -g @railway/cli" -ForegroundColor Cyan
        Write-Host ""
        exit
    }
    
    Write-Host ""
    Write-Host "Logging in..." -ForegroundColor Cyan
    railway login
    
    Write-Host "Initializing..." -ForegroundColor Cyan
    railway init
    
    Write-Host "Deploying..." -ForegroundColor Cyan
    railway up
    
    Write-Host ""
    Write-Host "Deployment complete!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Get your URL:" -ForegroundColor Yellow
    railway domain
    
    Write-Host ""
    Write-Host "Your Memory Search is live!" -ForegroundColor Green
    Write-Host ""
}
else {
    Write-Host "Invalid choice" -ForegroundColor Red
}

Write-Host ""

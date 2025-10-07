#!/bin/bash
# Quick deployment script for Railway/Render

echo "🚀 Memory Search - Quick Deploy"
echo "================================"
echo ""

# Check if git is initialized
if [ ! -d .git ]; then
    echo "📦 Initializing git repository..."
    git init
    git add .
    git commit -m "Initial commit - Memory Search v1.0"
fi

# Create production requirements
echo "📝 Creating production requirements..."
cat > requirements_prod.txt << 'EOF'
fastapi==0.104.1
uvicorn[standard]==0.24.0
torch==2.1.0
transformers==4.35.0
faiss-cpu==1.7.4
pillow==10.1.0
numpy==1.24.3
pydantic==2.5.0
pydantic-settings==2.1.0
python-multipart==0.0.6
opencv-python-headless==4.8.1.78
pyyaml==6.0.1
structlog==23.2.0
prometheus-client==0.19.0
EOF

# Create Procfile for Railway/Render
echo "📝 Creating Procfile..."
cat > Procfile << 'EOF'
web: uvicorn run_api:app --host 0.0.0.0 --port $PORT --workers 2
EOF

# Create railway.json
echo "📝 Creating railway.json..."
cat > railway.json << 'EOF'
{
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "uvicorn run_api:app --host 0.0.0.0 --port $PORT --workers 2",
    "healthcheckPath": "/v1/health",
    "healthcheckTimeout": 100,
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 3
  }
}
EOF

# Create .env.example
echo "📝 Creating .env.example..."
cat > .env.example << 'EOF'
# API Configuration
RAG_CONFIG_PATH=config/pipeline.yaml
PYTHONUNBUFFERED=1

# Security
API_KEY=changeme

# Optional: Database
# DATABASE_URL=postgresql://user:pass@host:5432/db

# Optional: Redis
# REDIS_URL=redis://host:6379
EOF

echo ""
echo "✅ Deployment files created!"
echo ""
echo "📋 Next steps:"
echo ""
echo "Option 1: Deploy to Railway"
echo "  1. npm install -g @railway/cli"
echo "  2. railway login"
echo "  3. railway init"
echo "  4. railway up"
echo ""
echo "Option 2: Deploy to Render"
echo "  1. Push to GitHub"
echo "  2. Connect Render to your repo"
echo "  3. Set environment variables"
echo "  4. Deploy!"
echo ""
echo "Option 3: Deploy to Fly.io"
echo "  1. flyctl launch"
echo "  2. flyctl deploy"
echo ""

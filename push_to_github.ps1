# Push Video Search Pipeline to GitHub

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "🚀 Pushing to GitHub" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Check if git is installed
try {
    git --version | Out-Null
} catch {
    Write-Host "❌ Git is not installed" -ForegroundColor Red
    Write-Host "Download from: https://git-scm.com/download/win"
    exit 1
}

Write-Host "✅ Git is installed" -ForegroundColor Green
Write-Host ""

# Initialize git if not already
if (-not (Test-Path ".git")) {
    Write-Host "📦 Initializing git repository..." -ForegroundColor Yellow
    git init
    Write-Host "✅ Git initialized" -ForegroundColor Green
} else {
    Write-Host "✅ Git repository already initialized" -ForegroundColor Green
}

Write-Host ""
Write-Host "📝 Creating .gitignore..." -ForegroundColor Yellow

# Create comprehensive .gitignore
$gitignore = @"
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
ENV/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db
desktop.ini

# Data files (large)
ground_clips_mp4/
ground_clips_mp4_full/
ground_clips_mp4_subset/
ground_clips_mp4_test/
ucf101_download/
downloaded_videos/
*.rar
*.zip
*.tar.gz

# Processed data (generated)
data/processed/embeddings/*.npy
data/processed/*.json
data/index/*.index
data/raw/
data/interim/

# Model cache
.cache/
models/
*.pt
*.pth
*.ckpt
*.safetensors

# Logs
*.log
logs/

# Temporary files
*.tmp
*.temp
.pytest_cache/
.coverage
htmlcov/

# Environment variables
.env
.env.local

# Jupyter
.ipynb_checkpoints/
*.ipynb

# Keep structure but ignore large files
!data/.gitkeep
!data/processed/.gitkeep
!data/index/.gitkeep
"@

$gitignore | Out-File -FilePath ".gitignore" -Encoding UTF8
Write-Host "✅ .gitignore created" -ForegroundColor Green

Write-Host ""
Write-Host "📝 Creating README.md..." -ForegroundColor Yellow

# Create README if it doesn't exist
if (-not (Test-Path "README.md")) {
    $readme = @"
# 🎬 AI Video Search Pipeline

Complete end-to-end video search system with CLIP embeddings, auto-labeling, and semantic search.

## ✨ Features

- 🔍 **Semantic Video Search** - Find videos using natural language queries
- 🤖 **Auto-Labeling** - Automatic action, object, and caption detection
- ⚡ **CLIP Embeddings** - State-of-the-art visual-semantic matching
- 🚀 **FastAPI Server** - Production-ready REST API
- 🌐 **Web Interface** - Beautiful search UI
- 📊 **FAISS Index** - Fast similarity search
- 🐳 **Docker Ready** - Easy deployment

## 🚀 Quick Start

### Local Setup

\`\`\`bash
# 1. Clone repository
git clone https://github.com/YOUR_USERNAME/Rag_Video_search_Pipeline.git
cd Rag_Video_search_Pipeline

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set Hugging Face token
export HF_TOKEN="your_token_here"

# 4. Download models
python download_clip_model.py
python download_blip_model.py

# 5. Process videos
python run_pipeline.py --enable-labeling

# 6. Start API
python run_api.py

# 7. Open browser
# http://localhost:8081/static/index.html
\`\`\`

### Google Colab (Recommended)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/Rag_Video_search_Pipeline/blob/main/Video_Search_Pipeline_Colab.ipynb)

- ✅ Free GPU (T4)
- ✅ No setup required
- ✅ 10x faster processing

### Docker

\`\`\`bash
# Build and run
docker-compose up -d

# Access at http://localhost:8081
\`\`\`

## 📊 Supported Datasets

- **UCF101** - 13,320 videos, 101 action categories
- **Custom videos** - Any MP4 videos in \`ground_clips_mp4/\`

## 🏗️ Architecture

\`\`\`
Videos → Keyframes → CLIP Embeddings → FAISS Index
                   ↓
              Auto-Labeling (BLIP + Detection)
                   ↓
              FastAPI Server → Web UI
\`\`\`

## 📖 Documentation

- [Deployment Guide](DEPLOYMENT_GUIDE.md) - Production deployment
- [Quick Deploy](QUICK_DEPLOY.md) - Fast setup guide
- [Auto-Labeling Guide](AUTO_LABELING_GUIDE.md) - Labeling features
- [Video Retrieval Guide](VIDEO_RETRIEVAL_GUIDE.md) - Search details

## 🎯 Example Queries

- "person playing basketball"
- "someone running outdoors"
- "cycling in the park"
- "playing guitar"
- "martial arts training"

## 📈 Performance

| Dataset | Videos | Processing Time | Search Speed |
|---------|--------|----------------|--------------|
| Small | 200 | 15 min (GPU) | <100ms |
| UCF101 | 13,320 | 2-3 hours (GPU) | <100ms |

## 🛠️ Tech Stack

- **Backend:** Python, FastAPI, PyTorch
- **Models:** CLIP (OpenAI), BLIP (Salesforce)
- **Search:** FAISS
- **Frontend:** HTML, JavaScript, TailwindCSS
- **Deployment:** Docker, Kubernetes, Cloud Run

## 📝 License

MIT License - see [LICENSE](LICENSE) file

## 🤝 Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md)

## 📧 Contact

- GitHub: [@YOUR_USERNAME](https://github.com/YOUR_USERNAME)
- Issues: [Report bugs](https://github.com/YOUR_USERNAME/Rag_Video_search_Pipeline/issues)

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=YOUR_USERNAME/Rag_Video_search_Pipeline&type=Date)](https://star-history.com/#YOUR_USERNAME/Rag_Video_search_Pipeline&Date)

---

**Built with ❤️ using CLIP and BLIP**
"@
    $readme | Out-File -FilePath "README.md" -Encoding UTF8
    Write-Host "✅ README.md created" -ForegroundColor Green
} else {
    Write-Host "✅ README.md already exists" -ForegroundColor Green
}

Write-Host ""
Write-Host "📦 Adding files to git..." -ForegroundColor Yellow

# Add all files
git add .

Write-Host "✅ Files staged" -ForegroundColor Green

Write-Host ""
Write-Host "💬 Creating commit..." -ForegroundColor Yellow

# Commit
git commit -m "Initial commit: AI Video Search Pipeline with CLIP embeddings and auto-labeling"

Write-Host "✅ Commit created" -ForegroundColor Green

Write-Host ""
Write-Host "🔗 Setting up remote repository..." -ForegroundColor Yellow
Write-Host ""
Write-Host "Please enter your GitHub repository URL:"
Write-Host "Example: https://github.com/varunsripad123/Rag_Video_search_Pipeline.git"
Write-Host ""
$repoUrl = Read-Host "Repository URL"

if ($repoUrl) {
    # Check if remote already exists
    $remoteExists = git remote get-url origin 2>$null
    
    if ($remoteExists) {
        Write-Host "⚠️  Remote 'origin' already exists. Updating..." -ForegroundColor Yellow
        git remote set-url origin $repoUrl
    } else {
        git remote add origin $repoUrl
    }
    
    Write-Host "✅ Remote repository set" -ForegroundColor Green
    
    Write-Host ""
    Write-Host "🚀 Pushing to GitHub..." -ForegroundColor Yellow
    Write-Host ""
    
    # Push to main branch
    try {
        git branch -M main
        git push -u origin main
        
        Write-Host ""
        Write-Host "==========================================" -ForegroundColor Cyan
        Write-Host "✅ Successfully pushed to GitHub!" -ForegroundColor Green
        Write-Host "==========================================" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "🔗 Your repository:" -ForegroundColor Cyan
        Write-Host "   $repoUrl" -ForegroundColor White
        Write-Host ""
        Write-Host "📝 Next steps:" -ForegroundColor Yellow
        Write-Host "   1. Go to your GitHub repository"
        Write-Host "   2. Update README.md with your username"
        Write-Host "   3. Add repository description and topics"
        Write-Host "   4. Share with others!"
        Write-Host ""
        
    } catch {
        Write-Host ""
        Write-Host "❌ Push failed!" -ForegroundColor Red
        Write-Host ""
        Write-Host "Common issues:" -ForegroundColor Yellow
        Write-Host "1. Repository doesn't exist - create it on GitHub first"
        Write-Host "2. Authentication failed - configure git credentials:"
        Write-Host "   git config --global user.name 'Your Name'"
        Write-Host "   git config --global user.email 'your@email.com'"
        Write-Host "3. Use Personal Access Token instead of password"
        Write-Host ""
        Write-Host "Manual push command:" -ForegroundColor Cyan
        Write-Host "   git push -u origin main" -ForegroundColor White
        Write-Host ""
    }
} else {
    Write-Host ""
    Write-Host "⚠️  No repository URL provided" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "To push later, run:" -ForegroundColor Cyan
    Write-Host "   git remote add origin YOUR_REPO_URL" -ForegroundColor White
    Write-Host "   git push -u origin main" -ForegroundColor White
    Write-Host ""
}

Write-Host "==========================================" -ForegroundColor Cyan

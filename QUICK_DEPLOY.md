# 🚀 Quick Deployment Guide

Get your video search pipeline running in production in 5 minutes!

---

## ⚡ Prerequisites

- Docker installed
- Docker Compose installed
- Your videos processed (run `python re_embed_clip_only_parallel.py` first)

---

## 🎯 Option 1: Docker (Simplest - Windows/Mac/Linux)

### Step 1: Build and Run

**Windows (PowerShell):**
```powershell
.\deploy.ps1 -Mode basic
```

**Linux/Mac:**
```bash
chmod +x deploy.sh
./deploy.sh basic
```

### Step 2: Access

- **API:** http://localhost:8081
- **Web UI:** http://localhost:8081/static/index.html

**Done!** Your video search is now running in Docker! 🎉

---

## 🏢 Option 2: Production Deployment (All Features)

Includes: API + Redis Cache + Nginx + Monitoring

**Windows:**
```powershell
.\deploy.ps1 -Mode production
```

**Linux/Mac:**
```bash
./deploy.sh production
```

### Access Points:
- **API:** http://localhost:8081
- **Web UI:** http://localhost (via Nginx)
- **Prometheus:** http://localhost:9090
- **Grafana:** http://localhost:3000 (admin/admin)

---

## ☁️ Option 3: Cloud Deployment

### Google Cloud Run (Easiest Cloud Option)

```bash
# 1. Build and push
gcloud builds submit --tag gcr.io/YOUR-PROJECT/video-search

# 2. Deploy
gcloud run deploy video-search \
  --image gcr.io/YOUR-PROJECT/video-search \
  --platform managed \
  --memory 4Gi \
  --cpu 2 \
  --max-instances 10 \
  --allow-unauthenticated

# Done! Auto-scales, HTTPS included
```

**Cost:** ~$5-20/month for low traffic

---

### AWS ECS (Production Ready)

```bash
# 1. Push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin ACCOUNT.dkr.ecr.us-east-1.amazonaws.com

docker build -t video-search .
docker tag video-search:latest ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/video-search:latest
docker push ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/video-search:latest

# 2. Create ECS service (use AWS Console or Terraform)
# See DEPLOYMENT_GUIDE.md for details
```

**Cost:** ~$100-200/month

---

## 🛠️ Common Commands

### Check Status
```bash
docker-compose ps
```

### View Logs
```bash
docker-compose logs -f api
```

### Restart Services
```bash
# Windows
.\deploy.ps1 -Mode restart

# Linux/Mac
./deploy.sh restart
```

### Stop Everything
```bash
# Windows
.\deploy.ps1 -Mode stop

# Linux/Mac
./deploy.sh stop
```

### Update Code
```bash
# Stop services
docker-compose down

# Rebuild
docker-compose build

# Start again
# Windows
.\deploy.ps1 -Mode basic

# Linux/Mac
./deploy.sh basic
```

---

## 🧪 Testing Deployment

### Health Check
```bash
curl http://localhost:8081/health
```

### Search Test
```bash
curl -X POST http://localhost:8081/v1/search \
  -H "Content-Type: application/json" \
  -H "X-API-Key: changeme" \
  -d '{"query": "person walking", "top_k": 5}'
```

### Load Test
```bash
# Install Apache Bench
# Windows: Download from Apache website
# Linux: sudo apt install apache2-utils

# Run load test (100 requests, 10 concurrent)
ab -n 100 -c 10 -H "X-API-Key: changeme" \
  -p search.json -T application/json \
  http://localhost:8081/v1/search
```

---

## 📊 Scaling

### Scale Up (More Resources)

Edit `docker-compose.yml`:
```yaml
services:
  api:
    deploy:
      resources:
        limits:
          cpus: '8'        # Increase CPUs
          memory: 16G      # Increase RAM
```

Then restart:
```bash
docker-compose up -d --force-recreate
```

### Scale Out (More Instances)

```bash
# Run 3 API instances
docker-compose up -d --scale api=3
```

---

## 🔒 Security Checklist

Before going to production:

- [ ] Change API key in `config/pipeline.yaml`
- [ ] Enable HTTPS (use Let's Encrypt + Certbot)
- [ ] Restrict /metrics endpoint in `nginx.conf`
- [ ] Set firewall rules (only 80/443 public)
- [ ] Use environment variables for secrets
- [ ] Enable rate limiting
- [ ] Add authentication for admin endpoints

---

## 💰 Cost Estimates

### Local/On-Premise
- **Hardware:** 4 CPU, 16GB RAM server
- **Cost:** $50-100/month (cloud VPS) or free (own hardware)

### Cloud - Small Scale
- **Platform:** Google Cloud Run
- **Traffic:** < 10K requests/month
- **Cost:** $5-20/month

### Cloud - Medium Scale
- **Platform:** AWS ECS Fargate
- **Traffic:** < 100K requests/month
- **Cost:** $100-200/month

### Cloud - Large Scale
- **Platform:** Kubernetes (EKS/GKE)
- **Traffic:** > 1M requests/month
- **Cost:** $500-2000/month

---

## 🆘 Troubleshooting

### API Won't Start
```bash
# Check logs
docker-compose logs api

# Common issues:
# 1. Port 8081 already in use → Change port in docker-compose.yml
# 2. Out of memory → Increase memory limit
# 3. Model files missing → Run re_embed script first
```

### Slow Searches
```bash
# Check CPU usage
docker stats

# Solutions:
# 1. Enable Redis caching (already in docker-compose.yml)
# 2. Use GPU for FAISS (edit config/pipeline.yaml)
# 3. Scale horizontally (add more instances)
```

### High Memory Usage
```bash
# Reduce workers
# In docker-compose.yml:
environment:
  - WORKERS=2  # Reduce from 4
```

---

## 📚 More Information

- **Full Deployment Guide:** See `DEPLOYMENT_GUIDE.md`
- **Architecture Details:** See `VIDEO_RETRIEVAL_GUIDE.md`
- **API Documentation:** http://localhost:8081/docs (after starting)

---

## ✅ Quick Start Checklist

1. [ ] Process videos: `python re_embed_clip_only_parallel.py`
2. [ ] Deploy locally: `./deploy.ps1 -Mode basic` (Windows) or `./deploy.sh basic` (Linux/Mac)
3. [ ] Test: Open http://localhost:8081/static/index.html
4. [ ] (Optional) Add monitoring: `./deploy.ps1 -Mode production`
5. [ ] (Optional) Deploy to cloud: Follow cloud deployment section

---

**Your video search pipeline is ready to deploy! 🚀**

Need help? Check `DEPLOYMENT_GUIDE.md` for detailed instructions.

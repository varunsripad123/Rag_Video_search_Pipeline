# 🚀 Deploy Memory Search NOW

Quick deployment guide - get your system live in 30 minutes!

---

## ✅ What You Have

- ✅ Working API (FastAPI)
- ✅ Professional UI (HTML/CSS/JS)
- ✅ Video search system (CLIP + FAISS)
- ✅ Fine-tuning capability
- ✅ 200 videos processed

**You're ready to deploy!**

---

## 🎯 Fastest Path: Railway (15 minutes)

### Step 1: Prepare Files (2 minutes)

```bash
# Run the deployment script
bash deploy_quick.sh

# Or manually on Windows:
# 1. Copy requirements_prod.txt content
# 2. Create Procfile
# 3. Create railway.json
```

### Step 2: Deploy to Railway (10 minutes)

```bash
# 1. Install Railway CLI
npm install -g @railway/cli

# 2. Login to Railway
railway login

# 3. Initialize project
railway init

# 4. Link to new project
railway link

# 5. Set environment variables
railway variables set RAG_CONFIG_PATH=config/pipeline.yaml
railway variables set PYTHONUNBUFFERED=1

# 6. Deploy!
railway up
```

### Step 3: Get Your URL (1 minute)

```bash
# Generate public URL
railway domain

# Your API is live at:
# https://your-app.railway.app
```

### Step 4: Test It (2 minutes)

```bash
# Test health endpoint
curl https://your-app.railway.app/v1/health

# Test search
curl -X POST https://your-app.railway.app/v1/search/similar \
  -H "Content-Type: application/json" \
  -H "X-API-Key: changeme" \
  -d '{"query": "basketball", "options": {"top_k": 5}}'
```

**Done! Your system is live!** 🎉

---

## 🌐 Access Your Professional UI

Open in browser:
```
https://your-app.railway.app/static/index_pro.html
```

**Features:**
- ✅ Professional design
- ✅ Instant search
- ✅ Video playback
- ✅ Download videos
- ✅ Share results
- ✅ Responsive mobile

---

## 💰 Cost Breakdown

### Railway Pricing
- **Hobby Plan:** $5/month
  - 500 hours/month
  - 512 MB RAM
  - 1 GB storage
  - Perfect for MVP!

- **Pro Plan:** $20/month
  - Unlimited hours
  - 8 GB RAM
  - 100 GB storage
  - For production

**First $5 free with trial!**

---

## 🔧 Alternative: Render (Similar to Railway)

### Step 1: Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit - Memory Search"
git remote add origin YOUR_GITHUB_REPO
git push -u origin main
```

### Step 2: Deploy on Render

1. Go to https://render.com
2. Click "New +" → "Web Service"
3. Connect your GitHub repo
4. Configure:
   - **Name:** memory-search
   - **Environment:** Python 3
   - **Build Command:** `pip install -r requirements_prod.txt`
   - **Start Command:** `uvicorn run_api:app --host 0.0.0.0 --port $PORT`
5. Add environment variables:
   - `RAG_CONFIG_PATH=config/pipeline.yaml`
   - `PYTHONUNBUFFERED=1`
6. Click "Create Web Service"

**Done! Live in 5 minutes!**

---

## 🐳 Docker Deployment (Advanced)

### Build Docker Image

```bash
docker build -t memory-search:latest .
```

### Run Locally

```bash
docker run -p 8081:8081 \
  -e RAG_CONFIG_PATH=config/pipeline.yaml \
  memory-search:latest
```

### Deploy to Cloud

```bash
# AWS ECR
aws ecr get-login-password | docker login --username AWS --password-stdin YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com
docker tag memory-search:latest YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/memory-search:latest
docker push YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/memory-search:latest

# Deploy to ECS/Fargate
# (See full AWS guide in DEPLOYMENT_PIPELINE.md)
```

---

## 📊 Post-Deployment Checklist

### ✅ Verify Everything Works

```bash
# 1. Health check
curl https://your-app.railway.app/v1/health

# 2. Search test
curl -X POST https://your-app.railway.app/v1/search/similar \
  -H "Content-Type: application/json" \
  -H "X-API-Key: changeme" \
  -d '{"query": "basketball", "options": {"top_k": 5}}'

# 3. UI test
# Open: https://your-app.railway.app/static/index_pro.html
# Search for "basketball"
# Verify videos play
```

### ✅ Update Configuration

```bash
# Change API key from default
railway variables set API_KEY=your-secure-key-here

# Update in UI settings:
# Open UI → Settings → Change API Key
```

### ✅ Monitor Performance

```bash
# Check logs
railway logs

# Check metrics
railway status
```

---

## 🎯 Next Steps After Deployment

### Week 1: Get First Clients

**1. Create Landing Page**
- Use index_pro.html as demo
- Add pricing page
- Add contact form

**2. Reach Out to 10 Prospects**
- Sports teams
- Media companies
- Corporate training
- E-learning platforms

**3. Offer Free Pilot**
- "We'll index your first 1,000 videos free"
- Get feedback
- Refine product

### Week 2-4: Scale to 10 Clients

**1. Set Up Billing**
- Stripe integration
- $99/mo standard tier
- $299/mo premium tier

**2. Add Features**
- User accounts
- API keys per client
- Usage analytics

**3. Improve UI**
- Client dashboard
- Upload interface
- Analytics page

---

## 💡 Quick Wins

### Add Custom Domain

```bash
# Railway
railway domain add yourdomain.com

# Render
# Settings → Custom Domain → Add yourdomain.com
```

### Add SSL (Automatic!)

Both Railway and Render provide free SSL certificates automatically.

### Add Analytics

```html
<!-- Add to index_pro.html -->
<script async src="https://www.googletagmanager.com/gtag/js?id=GA_ID"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'GA_ID');
</script>
```

---

## 🚨 Troubleshooting

### Issue: "Module not found"
**Fix:** Check requirements_prod.txt includes all dependencies

### Issue: "Port already in use"
**Fix:** Railway/Render automatically set $PORT, don't hardcode 8081

### Issue: "Videos not playing"
**Fix:** Check API key in UI settings matches server

### Issue: "Slow search"
**Fix:** Upgrade to Pro plan for more RAM

---

## 📞 Support

**Questions?**
- Check logs: `railway logs` or Render dashboard
- Test locally first: `python run_api.py`
- Verify data exists: `ls data/processed/`

---

## ✅ Deployment Complete!

**Your Memory Search system is now live!** 🎉

**URLs:**
- API: `https://your-app.railway.app`
- UI: `https://your-app.railway.app/static/index_pro.html`
- Health: `https://your-app.railway.app/v1/health`
- Docs: `https://your-app.railway.app/docs`

**Next:**
1. Share with first users
2. Collect feedback
3. Get first paying client
4. Scale to $10K MRR

**You're ready to launch! 🚀**

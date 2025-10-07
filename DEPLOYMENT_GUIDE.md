# 🚀 Deployment & Scaling Guide

Complete guide to deploying and scaling your video search pipeline to production.

---

## 📋 Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Deployment Options](#deployment-options)
3. [Docker Setup](#docker-setup)
4. [Cloud Deployment](#cloud-deployment)
5. [Scaling Strategies](#scaling-strategies)
6. [Performance Optimization](#performance-optimization)
7. [Monitoring & Observability](#monitoring--observability)
8. [Cost Optimization](#cost-optimization)

---

## 🏗️ Architecture Overview

### Current System Components

```
┌─────────────────────────────────────────────────────────┐
│                    Video Search System                   │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   FastAPI    │  │    CLIP      │  │    FAISS     │  │
│  │   Server     │──│   Encoder    │──│    Index     │  │
│  │  (Port 8081) │  │  (CPU/GPU)   │  │  (Search)    │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│                                                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Video      │  │  Embeddings  │  │   Metadata   │  │
│  │   Storage    │  │   Storage    │  │   (JSON)     │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

### Components to Scale

1. **API Server** (FastAPI) - User queries
2. **Embedding Service** (CLIP) - Video encoding
3. **Search Index** (FAISS) - Vector similarity
4. **Storage** - Videos, embeddings, metadata

---

## 🐳 Docker Setup

### Step 1: Create Dockerfile

```dockerfile
# Dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8081

# Run API server
CMD ["python", "run_api.py"]
```

### Step 2: Create Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8081:8081"
    volumes:
      - ./data:/app/data
      - ./ground_clips_mp4:/app/ground_clips_mp4
    environment:
      - WORKERS=4
      - DEVICE=cpu
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - api
    restart: unless-stopped
```

### Step 3: Build and Run

```bash
# Build image
docker build -t video-search-api .

# Run with docker-compose
docker-compose up -d

# Check logs
docker-compose logs -f api
```

---

## ☁️ Cloud Deployment

### Option 1: AWS Deployment

#### Architecture
```
Internet → CloudFront → ALB → ECS/Fargate → S3 (Videos)
                                         ↓
                                    ElastiCache (FAISS)
```

#### Setup Steps

**1. Containerize & Push to ECR**
```bash
# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

# Build and push
docker build -t video-search .
docker tag video-search:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/video-search:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/video-search:latest
```

**2. Create ECS Task Definition**
```json
{
  "family": "video-search",
  "cpu": "2048",
  "memory": "8192",
  "networkMode": "awsvpc",
  "containerDefinitions": [
    {
      "name": "api",
      "image": "<account-id>.dkr.ecr.us-east-1.amazonaws.com/video-search:latest",
      "portMappings": [
        {
          "containerPort": 8081,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {"name": "DEVICE", "value": "cpu"},
        {"name": "WORKERS", "value": "4"}
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/video-search",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

**3. Setup ECS Service**
```bash
# Create ECS cluster
aws ecs create-cluster --cluster-name video-search-cluster

# Create service
aws ecs create-service \
  --cluster video-search-cluster \
  --service-name video-search-api \
  --task-definition video-search \
  --desired-count 2 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx],assignPublicIp=ENABLED}"
```

**4. Setup Application Load Balancer**
```bash
# Create ALB
aws elbv2 create-load-balancer \
  --name video-search-alb \
  --subnets subnet-xxx subnet-yyy \
  --security-groups sg-xxx

# Create target group
aws elbv2 create-target-group \
  --name video-search-targets \
  --protocol HTTP \
  --port 8081 \
  --vpc-id vpc-xxx \
  --target-type ip
```

#### Cost Estimate (AWS)
- **ECS Fargate (2 tasks):** ~$70/month
- **ALB:** ~$25/month
- **S3 Storage (100GB):** ~$2.30/month
- **CloudFront:** Pay per use (~$0.085/GB)
- **Total:** ~$100-150/month for moderate traffic

---

### Option 2: Google Cloud Run (Easiest)

**Ultra-simple serverless deployment:**

```bash
# 1. Build and push to GCR
gcloud builds submit --tag gcr.io/PROJECT-ID/video-search

# 2. Deploy to Cloud Run
gcloud run deploy video-search \
  --image gcr.io/PROJECT-ID/video-search \
  --platform managed \
  --region us-central1 \
  --memory 4Gi \
  --cpu 2 \
  --max-instances 10 \
  --allow-unauthenticated

# Done! Auto-scales from 0 to 10 instances
```

**Benefits:**
- ✅ Auto-scaling (0 to N instances)
- ✅ Pay per request (no idle costs)
- ✅ Simple deployment
- ✅ Built-in HTTPS

**Cost:** ~$0.00002400/request + compute time (~$5-20/month for low traffic)

---

### Option 3: Azure Container Instances

```bash
# Create resource group
az group create --name video-search-rg --location eastus

# Deploy container
az container create \
  --resource-group video-search-rg \
  --name video-search \
  --image <registry>.azurecr.io/video-search:latest \
  --cpu 2 \
  --memory 8 \
  --ports 8081 \
  --environment-variables DEVICE=cpu WORKERS=4
```

---

### Option 4: Kubernetes (Production Scale)

```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: video-search-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: video-search
  template:
    metadata:
      labels:
        app: video-search
    spec:
      containers:
      - name: api
        image: <registry>/video-search:latest
        ports:
        - containerPort: 8081
        resources:
          requests:
            cpu: "1"
            memory: "4Gi"
          limits:
            cpu: "2"
            memory: "8Gi"
        env:
        - name: DEVICE
          value: "cpu"
---
apiVersion: v1
kind: Service
metadata:
  name: video-search-service
spec:
  selector:
    app: video-search
  ports:
  - port: 80
    targetPort: 8081
  type: LoadBalancer
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: video-search-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: video-search-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

Deploy:
```bash
kubectl apply -f kubernetes/
```

---

## 📈 Scaling Strategies

### 1. Vertical Scaling (Single Instance)

**Increase resources on one machine:**

```yaml
# docker-compose.yml
services:
  api:
    deploy:
      resources:
        limits:
          cpus: '16'        # More CPUs
          memory: 32G       # More RAM
    environment:
      - WORKERS=12          # More workers
```

**When to use:**
- ✅ Simple setup
- ✅ < 100 requests/second
- ✅ Single datacenter

**Limits:**
- ❌ Single point of failure
- ❌ Max ~16-32 cores
- ❌ No geographic distribution

---

### 2. Horizontal Scaling (Multiple Instances)

**Run multiple API servers behind load balancer:**

```
                    ┌─── API Instance 1 (4 cores)
                    │
Internet → LB ──────┼─── API Instance 2 (4 cores)
                    │
                    └─── API Instance 3 (4 cores)
                    
        All share same FAISS index + storage
```

**Setup:**

```yaml
# docker-compose.yml
version: '3.8'

services:
  api1:
    build: .
    environment:
      - INSTANCE_ID=1
  
  api2:
    build: .
    environment:
      - INSTANCE_ID=2
  
  api3:
    build: .
    environment:
      - INSTANCE_ID=3
  
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - api1
      - api2
      - api3
```

**nginx.conf:**
```nginx
upstream api_servers {
    least_conn;
    server api1:8081;
    server api2:8081;
    server api3:8081;
}

server {
    listen 80;
    
    location / {
        proxy_pass http://api_servers;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

**When to use:**
- ✅ > 100 requests/second
- ✅ High availability needed
- ✅ Can add/remove instances dynamically

---

### 3. Microservices Architecture (Advanced)

**Separate concerns into independent services:**

```
                    ┌─── API Service (Query handling)
                    │
Internet → Gateway ─┼─── Embedding Service (CLIP encoding)
                    │
                    ├─── Search Service (FAISS)
                    │
                    └─── Storage Service (Videos)
```

**Benefits:**
- ✅ Scale components independently
- ✅ Different resources per service
- ✅ Can use GPUs for embedding service only

**Implementation:**

Create separate services:

```python
# embedding_service.py
from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.post("/embed")
async def embed_video(video_path: str):
    # Load CLIP, encode video
    embedding = clip_encoder.encode(video_path)
    return {"embedding": embedding.tolist()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8082)
```

```python
# search_service.py
from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.post("/search")
async def search(query_embedding: List[float], top_k: int = 10):
    # Search FAISS index
    results = faiss_index.search(query_embedding, top_k)
    return {"results": results}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8083)
```

---

## ⚡ Performance Optimization

### 1. Caching Strategy

**Add Redis for query caching:**

```python
# src/api/cache.py
import redis
import hashlib
import json

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def cache_search_results(query: str, results: list, ttl: int = 3600):
    """Cache search results for 1 hour."""
    key = f"search:{hashlib.md5(query.encode()).hexdigest()}"
    redis_client.setex(key, ttl, json.dumps(results))

def get_cached_results(query: str):
    """Retrieve cached results."""
    key = f"search:{hashlib.md5(query.encode()).hexdigest()}"
    cached = redis_client.get(key)
    return json.loads(cached) if cached else None
```

**Update search endpoint:**

```python
@app.post("/v1/search")
async def search(request: SearchRequest):
    # Check cache first
    cached = get_cached_results(request.query)
    if cached:
        return cached
    
    # Perform search
    results = retriever.search(...)
    
    # Cache results
    cache_search_results(request.query, results)
    
    return results
```

**docker-compose.yml with Redis:**
```yaml
services:
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
  
  api:
    depends_on:
      - redis
    environment:
      - REDIS_HOST=redis
```

---

### 2. FAISS GPU Acceleration

**Use GPU for faster search:**

```python
# src/core/indexing.py
import faiss

class FAISSIndex:
    def __init__(self, dim: int, use_gpu: bool = True):
        self.dim = dim
        
        if use_gpu and faiss.get_num_gpus() > 0:
            # Create GPU index
            res = faiss.StandardGpuResources()
            index_cpu = faiss.IndexFlatIP(dim)
            self.index = faiss.index_cpu_to_gpu(res, 0, index_cpu)
        else:
            self.index = faiss.IndexFlatIP(dim)
```

**Performance:**
- CPU: ~1000 queries/sec
- GPU: ~10,000+ queries/sec (10x faster!)

---

### 3. Batch Processing

**Process multiple queries together:**

```python
@app.post("/v1/search/batch")
async def batch_search(requests: List[SearchRequest]):
    # Embed all queries at once (efficient)
    query_embeddings = embedder.embed_batch([r.query for r in requests])
    
    # Search in parallel
    results = []
    for emb in query_embeddings:
        results.append(retriever.search(emb))
    
    return results
```

---

## 📊 Monitoring & Observability

### Prometheus + Grafana Setup

```python
# src/api/metrics.py
from prometheus_client import Counter, Histogram, Gauge
import time

# Metrics
search_requests = Counter('search_requests_total', 'Total search requests')
search_latency = Histogram('search_latency_seconds', 'Search latency')
active_connections = Gauge('active_connections', 'Active connections')

@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    search_requests.inc()
    active_connections.inc()
    
    start = time.time()
    response = await call_next(request)
    search_latency.observe(time.time() - start)
    
    active_connections.dec()
    return response
```

**Grafana Dashboard:**
- Request rate (queries/sec)
- Latency (p50, p95, p99)
- Error rate
- CPU/Memory usage
- Cache hit rate

---

## 💰 Cost Optimization

### Strategy 1: Serverless (Google Cloud Run)

**Best for:** Variable traffic

```
Cost = (Requests × $0.00002400) + (Compute time × $0.00002400/GiB-s)

Example:
- 100,000 requests/month
- 500ms avg latency
- 4GB memory

Cost ≈ $2.40 + $10 = ~$12.40/month
```

### Strategy 2: Reserved Instances (AWS)

**Best for:** Predictable traffic

```
On-Demand: $70/month
Reserved (1-year): $45/month (36% savings)
Reserved (3-year): $30/month (57% savings)
```

### Strategy 3: Spot Instances

**Best for:** Batch processing (video encoding)

```
Normal: $0.0416/hour
Spot: ~$0.0125/hour (70% savings!)
```

---

## 🎯 Recommended Architectures by Scale

### Small Scale (< 10 users, < 100 videos)
```
Single Docker container
- 4 CPU, 8GB RAM
- Cost: $20-40/month
```

### Medium Scale (< 1000 users, < 10K videos)
```
Cloud Run (GCP) or Fargate (AWS)
- 2-4 instances auto-scaling
- Redis cache
- S3/GCS storage
- Cost: $100-200/month
```

### Large Scale (> 1000 users, > 100K videos)
```
Kubernetes cluster
- 5-10 API pods
- Separate embedding service with GPU
- Redis cluster
- CDN for videos
- PostgreSQL for metadata
- Cost: $500-1500/month
```

### Enterprise Scale (> 100K users, millions of videos)
```
Multi-region deployment
- Kubernetes in 3+ regions
- Distributed FAISS (Milvus or Weaviate)
- Multi-region storage
- Global CDN
- Dedicated GPU instances
- Cost: $5000-20,000/month
```

---

## 📝 Quick Start Checklist

- [ ] Dockerize application
- [ ] Test locally with docker-compose
- [ ] Choose cloud provider (GCP Cloud Run recommended for start)
- [ ] Deploy to cloud
- [ ] Add monitoring (Prometheus + Grafana)
- [ ] Setup caching (Redis)
- [ ] Configure auto-scaling
- [ ] Add CDN for videos (CloudFront/CloudFlare)
- [ ] Implement CI/CD pipeline
- [ ] Load test with Apache Bench or Locust

---

Need help with any specific deployment? I can create the exact configs for your chosen platform!

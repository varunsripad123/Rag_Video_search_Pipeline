#!/usr/bin/env bash

set -euo pipefail

LOG_FILE=logs/deploy_$(date +%Y%m%d_%H%M%S).log
mkdir -p logs
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting production deployment..."

# Validate kubectl availability
if ! command -v kubectl &> /dev/null; then
  echo "kubectl command not found! Please install kubectl."
  exit 1
fi

# Apply Kubernetes manifests
echo "Applying Kubernetes manifests..."
kubectl apply -f k8s/

echo "Waiting for deployment to complete..."
kubectl rollout status deployment/rag-video-search --timeout=300s

echo "Deployment completed successfully!"
echo "Check status: kubectl get all -l app=rag-video-search"
echo "View logs: kubectl logs -f deployment/rag-video-search"
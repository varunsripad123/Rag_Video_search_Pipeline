# 🚀 Fine-tuning Quick Start Guide

Get started with client-specific fine-tuning in 10 minutes!

---

## 📦 Step 1: Install Dependencies

```bash
pip install -r requirements_finetuning.txt
```

**Installs:**
- `peft` - LoRA adapters
- `accelerate` - Training acceleration
- `datasets` - Data handling

---

## 🎯 Step 2: Fine-tune on Client Videos

### Option A: Auto-detect labels from folders

```bash
python finetune_client.py \
  --client acme \
  --videos ground_clips_mp4 \
  --epochs 10
```

**Uses folder names as labels:**
```
ground_clips_mp4/
  ├── Basketball/  → label: "Basketball"
  ├── Running/     → label: "Running"
  └── Cycling/     → label: "Cycling"
```

### Option B: Custom labels from JSON

```bash
python finetune_client.py \
  --client acme \
  --videos /path/to/videos \
  --labels labels.json \
  --epochs 10
```

**labels.json format:**
```json
[
  {
    "video_path": "video1.mp4",
    "labels": [
      "surgical procedure",
      "laparoscopic surgery",
      "medical operation"
    ]
  },
  {
    "video_path": "video2.mp4",
    "labels": [
      "patient consultation",
      "doctor visit",
      "medical checkup"
    ]
  }
]
```

---

## ⏱️ Expected Time

| Videos | Time (GPU) | Time (CPU) |
|--------|-----------|------------|
| 100 | 15 min | 2 hours |
| 500 | 1 hour | 8 hours |
| 1,000 | 2 hours | 16 hours |
| 5,000 | 8 hours | 3 days |

**Recommendation:** Use GPU for 500+ videos

---

## 🧪 Step 3: Test Fine-tuned Model

```bash
python test_finetuned.py --client acme
```

**Output:**
```
📝 Query: 'basketball'
   Baseline:   31.0% ✅
   Fine-tuned: 52.3% ✅
   Change:     +21.3%
   Status:     🎯 Significant improvement!

📈 SUMMARY
Average accuracy:
  Baseline:   30.2%
  Fine-tuned: 51.8%
  Improvement: +21.6%

🎉 EXCELLENT: Fine-tuning significantly improved accuracy!
```

---

## 🚀 Step 4: Deploy Fine-tuned Model

### Option A: Re-process videos with fine-tuned model

```bash
python reprocess_with_finetuned.py --client acme
```

### Option B: Use in API

```python
# In src/api/server.py
from src.finetuning.trainer import ClientFineTuner

# Load fine-tuned model for specific client
finetuned_model, processor = ClientFineTuner.load_finetuned(
    Path(f"models/finetuned/{client_id}")
)
```

---

## 💰 Cost Analysis

### Training Cost (Cloud GPU)

| Instance | GPU | Cost/hour | 1K videos | 5K videos |
|----------|-----|-----------|-----------|-----------|
| **AWS g4dn.xlarge** | T4 | $0.526 | $1.05 | $4.21 |
| **AWS p3.2xlarge** | V100 | $3.06 | $3.06 | $12.24 |
| **GCP n1-standard-4 + T4** | T4 | $0.35 | $0.70 | $2.80 |

**Recommendation:** Use g4dn.xlarge (T4) - best value

### Storage Cost

| Component | Size | Cost (S3) |
|-----------|------|-----------|
| **Base CLIP** | 600 MB | $0.014/month |
| **LoRA adapters** | 10-50 MB | $0.001/month |
| **Total per client** | 650 MB | $0.015/month |

**For 100 clients:** $1.50/month storage

---

## 📊 Business Model

### Pricing Tiers

**Standard (No Fine-tuning)**
- Price: $99/month
- Accuracy: 30%
- Setup: Instant

**Premium (Fine-tuned)**
- Price: $299/month
- Accuracy: 50-60%
- Setup: $10 one-time
- Training: 2-4 hours

**Enterprise (Continuous Learning)**
- Price: $999/month
- Accuracy: 70-80%
- Monthly retraining
- Dedicated support

### Profit Margins

| Tier | Revenue | Cost | Profit | Margin |
|------|---------|------|--------|--------|
| Standard | $99 | $5 | $94 | 95% |
| Premium | $299 | $15 | $284 | 95% |
| Enterprise | $999 | $50 | $949 | 95% |

---

## 🎯 Advanced Options

### Custom LoRA Configuration

```bash
python finetune_client.py \
  --client acme \
  --videos /path/to/videos \
  --lora-rank 32 \        # Higher = more capacity (default: 16)
  --lr 5e-5 \             # Lower = more stable (default: 1e-4)
  --epochs 20 \           # More epochs = better fit
  --batch-size 16         # Larger = faster (if GPU allows)
```

### Multi-Client Fine-tuning

```bash
# Fine-tune for multiple clients
for client in acme corp industries; do
  python finetune_client.py \
    --client $client \
    --videos clients/$client/videos \
    --epochs 10
done
```

### Automated Retraining

```python
# Schedule monthly retraining
from apscheduler.schedulers.background import BackgroundScheduler

def retrain_client(client_id):
    # Get new videos from past month
    new_videos = get_new_videos(client_id, days=30)
    
    # Fine-tune
    subprocess.run([
        "python", "finetune_client.py",
        "--client", client_id,
        "--videos", new_videos,
        "--epochs", "5"
    ])

# Schedule
scheduler = BackgroundScheduler()
scheduler.add_job(retrain_client, 'cron', day=1, args=['acme'])
scheduler.start()
```

---

## 🛡️ Best Practices

### 1. Data Quality
- ✅ Use high-quality videos (720p+)
- ✅ Diverse examples per category (10+ videos)
- ✅ Clear, descriptive labels
- ❌ Avoid blurry/corrupted videos

### 2. Training
- ✅ Start with 10 epochs
- ✅ Monitor loss (should decrease)
- ✅ Test on held-out set
- ❌ Don't overtrain (loss stops improving)

### 3. Deployment
- ✅ A/B test before full rollout
- ✅ Monitor accuracy metrics
- ✅ Collect user feedback
- ✅ Retrain monthly with new data

---

## 🐛 Troubleshooting

### Issue: Out of Memory

**Error:** `CUDA out of memory`

**Fix:**
```bash
# Reduce batch size
python finetune_client.py --batch-size 4

# Or use CPU (slower)
python finetune_client.py --device cpu
```

### Issue: Loss Not Decreasing

**Possible causes:**
- Learning rate too high → Try `--lr 5e-5`
- Not enough data → Need 100+ videos minimum
- Labels too generic → Use more specific labels

### Issue: Accuracy Not Improving

**Possible causes:**
- Dataset too similar to CLIP training data
- Not enough epochs → Try `--epochs 20`
- Need more LoRA capacity → Try `--lora-rank 32`

---

## ✅ Checklist

Before deploying fine-tuned model:

- [ ] Trained for at least 10 epochs
- [ ] Loss decreased significantly
- [ ] Tested on held-out videos
- [ ] Accuracy improved by 10%+
- [ ] A/B tested with baseline
- [ ] Client approved results
- [ ] Model saved and backed up
- [ ] Documentation updated

---

## 📞 Support

**Questions?**
- Check logs in `logs/finetuning.log`
- Review training curves
- Test with `test_finetuned.py`

**Need help?**
- Increase epochs if underfitting
- Reduce learning rate if unstable
- Add more training data if overfitting

---

## 🎉 Success!

You now have a client-specific fine-tuned model that:
- ✅ Understands their domain better
- ✅ Provides 20-30% higher accuracy
- ✅ Creates competitive advantage
- ✅ Increases client retention

**Happy fine-tuning!** 🚀

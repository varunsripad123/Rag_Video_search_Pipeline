# 🚀 Quick Start Guide

## Start the Demo in 3 Steps

### Step 1: Wait for Pipeline to Complete
```bash
# The pipeline should finish with:
# "Pipeline Complete"
# "FAISS Index: data/index/faiss.index"
```

### Step 2: Start the API Server
```bash
python run_api.py
```

You should see:
```
🚀 Server starting...
📍 API Endpoint:  http://0.0.0.0:8080
🌐 Web Interface: http://localhost:8080/static/index.html
```

### Step 3: Open Your Browser

Navigate to: **http://localhost:8080**

---

## 🎯 Quick Demo Queries

Try these searches to impress investors:

1. **"person waving"** - Shows hand gesture recognition
2. **"walking"** - Demonstrates motion understanding
3. **"hand gestures"** - Highlights fine-grained detection
4. **"person standing"** - Shows pose recognition
5. **"movement"** - Broad semantic search

---

## 💡 Key Features to Show

### 1. Real-Time Search
- Type query → Results in <100ms
- Show the search duration metric

### 2. Confidence Scores
- Each result shows similarity percentage
- Higher scores = better matches

### 3. Video Playback
- Click any result to play video
- Videos start at the relevant timestamp

### 4. Professional UI
- Modern, clean interface
- Responsive design
- Smooth animations

---

## 🎨 Customization (Optional)

### Change API Key
Edit `config/pipeline.yaml`:
```yaml
security:
  api_keys:
    - your_secure_key_here
```

### Adjust Results Count
In the web interface:
- Click "Settings" button
- Change "Results to show" (1-20)

### Change Port
Edit `config/pipeline.yaml`:
```yaml
api:
  port: 8080  # Change to your preferred port
```

---

## ⚡ Troubleshooting

### Problem: "Connection refused"
**Solution:** Make sure API server is running (`python run_api.py`)

### Problem: "No results found"
**Solution:** 
- Check if index was built (`data/index/faiss.index` exists)
- Try different search queries
- Verify API key is correct

### Problem: Videos won't play
**Solution:**
- Check videos are in `ground_clips_mp4/` directory
- Try a different browser (Chrome/Edge recommended)

### Problem: Slow performance
**Solution:**
- Close other applications
- Check CPU/memory usage
- Restart the API server

---

## 📊 What Investors Want to See

✅ **Speed** - Sub-second search times  
✅ **Accuracy** - High confidence scores (>70%)  
✅ **Scalability** - Handles 272+ videos easily  
✅ **UX** - Professional, intuitive interface  
✅ **Technology** - Multi-modal AI (CLIP + VideoMAE + VideoSwin)  

---

## 🎬 30-Second Pitch

> "Watch this: I can find any moment in 272 videos by just describing it. [Type 'person waving'] See? Results in milliseconds with confidence scores. This is powered by three AI models working together - CLIP for vision-language understanding, VideoMAE for temporal patterns, and VideoSwin for 3D spatial features. We're targeting the $50B security market first, where manual video review costs companies millions annually."

---

## 📞 Need Help?

- Check `DEMO_GUIDE.md` for detailed walkthrough
- Review API logs in terminal for errors
- Press F12 in browser to see console errors

---

**You're ready to demo! 🎯**

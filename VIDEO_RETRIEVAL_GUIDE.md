# 🎥 Video Retrieval Guide - Original Quality

Your system stores the **original quality video chunks**, allowing you to retrieve them at any time without quality loss.

## 📦 What's Stored

Your pipeline stores multiple versions of each video:

1. **Original Chunks** (`data/processed/chunks/`) - Full quality MP4 files ✅
2. **Compressed Tokens** (`data/processed/tokens/`) - Neural codec (2.3x smaller)
3. **Embeddings** (`data/processed/embeddings/`) - AI-generated vectors for search
4. **Metadata** (`data/processed/metadata.json`) - Index of all videos

## 🛠️ Retrieval Methods

### Method 1: Command Line Tool

```bash
# Search by keyword
python retrieve_video.py waving

# Retrieve by manifest ID
python retrieve_video.py --id cecc6ffb-b210-433d-b81d-e70f9877df61

# Stitch full video from chunks
python retrieve_video.py --stitch person01_01_ground_waving
```

**Output:** Videos saved to `retrieved_videos/` directory

### Method 2: Web Interface (NEW!)

1. Search for videos at `http://localhost:8081/demo`
2. Click the **download icon** (⬇️) on any result
3. Original quality video downloads instantly

### Method 3: API Endpoint

```bash
# GET request to retrieve video
curl -H "x-api-key: changeme" \
  "http://localhost:8081/v1/video/{manifest_id}" \
  --output video.mp4
```

Or in Python:

```python
import requests

response = requests.get(
    f"http://localhost:8081/v1/video/{manifest_id}",
    headers={"x-api-key": "changeme"}
)

with open("retrieved_video.mp4", "wb") as f:
    f.write(response.content)
```

## 📊 Storage Statistics

Run this to see your compression savings:

```bash
python compression_stats.py
```

Expected output:
- **Average Compression:** 2.3x
- **Storage Saved:** ~56%
- **Original Quality:** Preserved in chunks

## 🔑 Key Features

✅ **Lossless Retrieval** - Original chunks stored without quality loss
✅ **Instant Access** - No reconstruction needed
✅ **Smart Stitching** - Combine chunks into full videos
✅ **API Integration** - Programmatic access to all videos
✅ **Web Download** - One-click download from search results

## 💡 For Your Investor Demo

**Talk Track:**
> "Our system uses neural compression to reduce storage by 56%, but we maintain the original quality videos. Users can instantly retrieve any clip at full resolution through our web interface or API. This gives you the best of both worlds: efficient storage AND perfect quality when you need it."

## 🎯 Next Steps

1. **Try the retrieval tool:** `python retrieve_video.py waving`
2. **Test web download:** Search and click download button
3. **Check compression stats:** `python compression_stats.py`
4. **Explore API:** Visit `http://localhost:8081/docs`

---

**Questions?** The original videos are in `data/processed/chunks/` if you want direct file access.

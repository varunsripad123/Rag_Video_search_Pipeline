# 🔍 Search with Auto-Labels - Complete Guide

## ✅ YES! Search Now Uses Auto-Labels

Your search system has been **enhanced** to use auto-generated labels for:
- ✅ **Filtering** results by objects, actions, and confidence
- ✅ **Including** label metadata in search responses  
- ✅ **Enriching** context for better AI-generated answers

---

## 🎯 What Changed

### Before (Vector-Only Search)
```python
# Old: Only embedding similarity
results = retriever.search(query_embedding, top_k=5)
# Returns: Basic metadata (label, score, timestamps)
```

### After (Enhanced with Auto-Labels)
```python
# New: Embedding similarity + auto-label filters
results = retriever.search(
    query_embedding,
    top_k=5,
    filter_objects=["person"],    # Filter by detected objects
    filter_action="walking",      # Filter by action
    min_confidence=0.7            # Filter by confidence
)
# Returns: Metadata + auto_labels (objects, actions, captions, audio)
```

---

## 🚀 How to Use

### 1. API Search (with filters)

```bash
curl -X POST http://localhost:8081/v1/search/similar \
  -H "Content-Type: application/json" \
  -H "x-api-key: changeme" \
  -d '{
    "query": "person waving",
    "options": {
      "top_k": 5,
      "filter_objects": ["person"],
      "filter_action": "waving",
      "min_confidence": 0.6
    }
  }'
```

**Response includes auto-labels:**
```json
{
  "answer": "Found 3 videos of people waving...",
  "results": [
    {
      "manifest_id": "abc-123",
      "label": "person_waving",
      "score": 0.95,
      "start_time": 0.0,
      "end_time": 2.0,
      "asset_url": "path/to/video.mp4",
      "auto_labels": {
        "objects": ["person", "hand"],
        "action": "waving",
        "caption": "a person waving their hand",
        "confidence": 0.84
      }
    }
  ]
}
```

### 2. Python API Search

```python
from src.config import load_config
from src.core.retrieval import Retriever
from src.api.server import QueryEmbedder
from transformers import CLIPModel, CLIPTokenizer
import torch

# Initialize
config = load_config()
metadata_path = config.data.processed_dir / "metadata.json"
retriever = Retriever(config, metadata_path)

# Initialize query embedder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model = CLIPModel.from_pretrained(config.models.clip_model_name).to(device)
clip_tokenizer = CLIPTokenizer.from_pretrained(config.models.clip_model_name)
embedder = QueryEmbedder(model=clip_model, tokenizer=clip_tokenizer, device=device)

# Search with filters
query_embedding = embedder.embed("person waving")

results = retriever.search(
    query_embedding,
    top_k=5,
    filter_objects=["person"],    # Only videos with people
    filter_action="waving",       # Only waving action
    min_confidence=0.6            # Min 60% confidence
)

# Access auto-labels
for result in results:
    print(f"Video: {result.label}")
    print(f"Objects: {result.auto_labels['objects']}")
    print(f"Action: {result.auto_labels['action']}")
    print(f"Caption: {result.auto_labels['caption']}")
```

### 3. Direct Metadata Search

```python
import json
from pathlib import Path

manifest = json.loads(Path("data/processed/metadata.json").read_text())

# Find videos with specific objects
person_videos = [
    v for v in manifest
    if v.get('auto_labels') and 'person' in v['auto_labels']['objects']
]

# Find videos by action
walking_videos = [
    v for v in manifest
    if v.get('auto_labels') and 'walk' in v['auto_labels']['action'].lower()
]

# Search captions
outdoor_videos = [
    v for v in manifest
    if v.get('auto_labels') and 'outdoor' in v['auto_labels']['caption'].lower()
]
```

---

## 🎨 Filter Options

### Filter by Objects

Only return videos containing specific objects:

```python
results = retriever.search(
    query_embedding,
    filter_objects=["person", "car"]  # Videos with person OR car
)
```

**API:**
```json
{
  "options": {
    "filter_objects": ["person", "car"]
  }
}
```

### Filter by Action

Only return videos with specific actions:

```python
results = retriever.search(
    query_embedding,
    filter_action="walking"  # Videos with "walking" action
)
```

**API:**
```json
{
  "options": {
    "filter_action": "walking"
  }
}
```

### Filter by Confidence

Only return high-confidence results:

```python
results = retriever.search(
    query_embedding,
    min_confidence=0.7  # Only 70%+ confidence
)
```

**API:**
```json
{
  "options": {
    "min_confidence": 0.7
  }
}
```

### Combine Filters

Use multiple filters together:

```python
results = retriever.search(
    query_embedding,
    top_k=5,
    filter_objects=["person"],
    filter_action="walking",
    min_confidence=0.6
)
# Returns: High-confidence videos of people walking
```

---

## 📊 Enhanced Context

The AI answer generation now uses auto-labels to provide **better context**:

**Before:**
```
Context: "person_waving 0.0-2.0s"
Answer: "Found a video labeled person_waving."
```

**After:**
```
Context: "person_waving 0.0-2.0s - a person waving their hand near a car"
Answer: "Found a video of a person waving their hand, with a car visible in the scene."
```

Auto-labels (captions and objects) are automatically included in the context sent to the answer generation system.

---

## 🎯 Use Cases

### 1. Content Discovery

```python
# Find all videos with cars
car_videos = retriever.search(
    embedder.embed("vehicles"),
    filter_objects=["car"],
    top_k=20
)
```

### 2. Action-Based Search

```python
# Find all walking scenes
walking_scenes = retriever.search(
    embedder.embed("movement"),
    filter_action="walking",
    top_k=10
)
```

### 3. High-Quality Results Only

```python
# Only show high-confidence results
quality_results = retriever.search(
    embedder.embed("any query"),
    min_confidence=0.8,
    top_k=5
)
```

### 4. Specific Scenarios

```python
# Find "person walking near car" scenes
specific = retriever.search(
    embedder.embed("person near vehicle"),
    filter_objects=["person", "car"],
    filter_action="walking",
    min_confidence=0.6
)
```

### 5. Content Moderation

```python
# Flag inappropriate content
flagged = retriever.search(
    embedder.embed("potential violations"),
    filter_objects=["weapon", "violence"],
    top_k=100
)
```

---

## 🧪 Test It

### Run Demo Script

```bash
# Local demos (no API server needed)
python demo_search_with_labels.py

# API demos (requires server)
python run_api.py  # In terminal 1
python demo_search_with_labels.py --api  # In terminal 2

# Specific demo
python demo_search_with_labels.py --demo 1
```

### Test Search Manually

```python
# Test script
from src.config import load_config
from src.core.retrieval import Retriever
from src.core.embedding import QueryEmbedder

config = load_config()
metadata_path = config.data.processed_dir / "metadata.json"

retriever = Retriever(config, metadata_path)
embedder = QueryEmbedder(config)

# Test 1: Basic search
query_emb = embedder.embed("person")
results = retriever.search(query_emb, top_k=3)
for r in results:
    print(f"{r.label}: {r.auto_labels}")

# Test 2: Filtered search
results = retriever.search(
    query_emb,
    filter_objects=["person"],
    min_confidence=0.5
)
print(f"Found {len(results)} high-confidence person videos")
```

---

## 📈 Performance

### Filtering Impact

**Without Filters:**
- Search: 100 videos → Return top 5
- Time: ~5ms

**With Filters:**
- Search: 100 videos → Filter → Return top 5
- Time: ~7ms (slightly slower but more relevant)

The system automatically searches **3x more results** when filtering to ensure you still get enough matches.

---

## 🔧 Implementation Details

### Files Modified

1. **`src/core/retrieval.py`**
   - Added `filter_objects`, `filter_action`, `min_confidence` parameters
   - Filter logic in `search()` method
   - Include `auto_labels` in `RetrievalResult`

2. **`src/api/models.py`**
   - Added filter fields to `SearchOptions`
   - Added `auto_labels` field to `SearchResult`

3. **`src/api/server.py`**
   - Pass filter parameters to `retriever.search()`
   - Include auto-labels in context for better answers
   - Return `auto_labels` in API responses

### How Filtering Works

```python
# Pseudo-code
def search(query_embedding, filter_objects=None):
    # 1. Get candidates from FAISS (vector similarity)
    candidates = faiss_index.search(query_embedding, k=top_k*3)
    
    # 2. Apply auto-label filters
    filtered = []
    for candidate in candidates:
        if filter_objects:
            if not any(obj in candidate.auto_labels['objects'] 
                      for obj in filter_objects):
                continue  # Skip this result
        
        filtered.append(candidate)
        
        if len(filtered) >= top_k:
            break
    
    # 3. Return top results
    return filtered[:top_k]
```

---

## ✅ Summary

**Your search now uses auto-labels!**

✅ **Filter by objects** - Only show videos with specific items  
✅ **Filter by actions** - Only show videos with specific activities  
✅ **Filter by confidence** - Only show high-quality labels  
✅ **Enriched responses** - Auto-labels included in API responses  
✅ **Better answers** - Captions used for context generation  
✅ **Backward compatible** - Works with or without auto-labels  

### Quick Example

```python
# Search for high-confidence videos of people walking
results = retriever.search(
    embedder.embed("person movement"),
    filter_objects=["person"],
    filter_action="walking",
    min_confidence=0.7,
    top_k=5
)

for result in results:
    print(f"✓ {result.label}")
    print(f"  Objects: {result.auto_labels['objects']}")
    print(f"  Action: {result.auto_labels['action']}")
    print(f"  Caption: {result.auto_labels['caption']}")
```

---

**Try it now:**
```bash
python demo_search_with_labels.py
```

Enjoy your enhanced search system! 🎯🔍

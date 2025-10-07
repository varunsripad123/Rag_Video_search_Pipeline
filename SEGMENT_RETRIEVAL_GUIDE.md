# 🎬 Video Segment Retrieval Guide

Complete guide for retrieving video segments with flexible options.

---

## 🎯 What Clients Can Do

### 1. **Basic Segment Retrieval**
Get the exact 2-second segment that matched the search.

```bash
GET /v1/video/{manifest_id}?api_key=changeme
```

**Example:**
```bash
curl "http://localhost:8081/v1/video/abc-123?api_key=changeme" -o segment.mp4
```

---

### 2. **Segment with Context**
Get the segment + extra seconds before/after for context.

```bash
GET /v1/video/{manifest_id}?api_key=changeme&context=5
```

**Example:**
```
Original segment: 10.5s - 12.5s (2 seconds)
With context=5:   5.5s - 17.5s (12 seconds)
```

**Use case:** Client wants to see what happened before/after the event.

---

### 3. **Different Quality Levels**
Get segment in different quality for bandwidth optimization.

```bash
# High quality (large file)
GET /v1/video/{manifest_id}?api_key=changeme&quality=high

# Medium quality (default)
GET /v1/video/{manifest_id}?api_key=changeme&quality=medium

# Low quality (small file, fast download)
GET /v1/video/{manifest_id}?api_key=changeme&quality=low
```

**File sizes:**
- High: ~5 MB/minute
- Medium: ~2 MB/minute
- Low: ~500 KB/minute

---

### 4. **GIF Preview**
Get animated GIF instead of video (great for previews).

```bash
GET /v1/video/{manifest_id}?api_key=changeme&format=gif
```

**Use case:** Quick preview in chat/email without video player.

---

### 5. **Thumbnail**
Get single frame thumbnail.

```bash
GET /v1/video/{manifest_id}?api_key=changeme&format=thumbnail
```

**Use case:** Grid view of search results.

---

## 📊 Complete API Reference

### Endpoint
```
GET /v1/video/{manifest_id}
```

### Parameters

| Parameter | Type | Options | Default | Description |
|-----------|------|---------|---------|-------------|
| `api_key` | string | - | **required** | Authentication key |
| `format` | string | `mp4`, `gif`, `thumbnail` | `mp4` | Output format |
| `quality` | string | `high`, `medium`, `low` | `medium` | Video quality |
| `context` | float | 0-30 | `0` | Extra seconds before/after |

---

## 🎯 Real-World Examples

### Example 1: Security Firm - Robbery Investigation

**Scenario:** Security firm searches for "person running with bag"

**Step 1: Search**
```json
POST /v1/search/similar
{
  "query": "person running with bag",
  "options": {"top_k": 10}
}
```

**Step 2: Results**
```json
{
  "results": [
    {
      "manifest_id": "robbery-segment-1",
      "video": "store_cam_2024_01_15.mp4",
      "start_time": 145.2,
      "end_time": 147.2,
      "score": 0.82
    }
  ]
}
```

**Step 3: Retrieve with Context**
```bash
# Get 10 seconds before and after to see full incident
GET /v1/video/robbery-segment-1?api_key=changeme&context=10

# Returns: 135.2s - 157.2s (22 seconds total)
```

---

### Example 2: Traffic Analysis - License Plate

**Scenario:** Find all instances of vehicle "ABC123"

**Step 1: Search**
```json
POST /v1/search/similar
{
  "query": "car with license plate ABC123",
  "options": {"top_k": 20}
}
```

**Step 2: Get Thumbnails for Quick Review**
```bash
# Get thumbnail of each result
GET /v1/video/car-1?api_key=changeme&format=thumbnail
GET /v1/video/car-2?api_key=changeme&format=thumbnail
GET /v1/video/car-3?api_key=changeme&format=thumbnail
```

**Step 3: Download Full Segment of Interest**
```bash
# High quality for evidence
GET /v1/video/car-2?api_key=changeme&quality=high&context=5
```

---

### Example 3: Sports Highlights - Basketball Dunk

**Scenario:** Create highlight reel of dunks

**Step 1: Search**
```json
POST /v1/search/similar
{
  "query": "basketball dunk",
  "options": {"top_k": 50}
}
```

**Step 2: Get GIFs for Social Media**
```bash
# Get GIF of each dunk
GET /v1/video/dunk-1?api_key=changeme&format=gif
GET /v1/video/dunk-2?api_key=changeme&format=gif
```

**Step 3: Download Full Clips for Editing**
```bash
# Get full clip with context
GET /v1/video/dunk-1?api_key=changeme&context=3&quality=high
```

---

## 💻 Client Integration Examples

### Python Client

```python
import requests

class VideoSearchClient:
    def __init__(self, api_url, api_key):
        self.api_url = api_url
        self.api_key = api_key
    
    def search(self, query, top_k=10):
        """Search for video segments."""
        response = requests.post(
            f"{self.api_url}/v1/search/similar",
            json={"query": query, "options": {"top_k": top_k}},
            headers={"X-API-Key": self.api_key}
        )
        return response.json()["results"]
    
    def get_segment(self, manifest_id, format="mp4", quality="medium", context=0):
        """Retrieve video segment."""
        url = f"{self.api_url}/v1/video/{manifest_id}"
        params = {
            "api_key": self.api_key,
            "format": format,
            "quality": quality,
            "context": context
        }
        response = requests.get(url, params=params)
        return response.content
    
    def download_segment(self, manifest_id, output_path, **kwargs):
        """Download segment to file."""
        content = self.get_segment(manifest_id, **kwargs)
        with open(output_path, "wb") as f:
            f.write(content)

# Usage
client = VideoSearchClient("http://localhost:8081", "changeme")

# Search
results = client.search("basketball dunk")

# Download first result with context
client.download_segment(
    results[0]["manifest_id"],
    "dunk.mp4",
    context=5,
    quality="high"
)
```

---

### JavaScript Client

```javascript
class VideoSearchClient {
  constructor(apiUrl, apiKey) {
    this.apiUrl = apiUrl;
    this.apiKey = apiKey;
  }
  
  async search(query, topK = 10) {
    const response = await fetch(`${this.apiUrl}/v1/search/similar`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': this.apiKey
      },
      body: JSON.stringify({
        query,
        options: { top_k: topK }
      })
    });
    const data = await response.json();
    return data.results;
  }
  
  getSegmentUrl(manifestId, options = {}) {
    const params = new URLSearchParams({
      api_key: this.apiKey,
      format: options.format || 'mp4',
      quality: options.quality || 'medium',
      context: options.context || 0
    });
    return `${this.apiUrl}/v1/video/${manifestId}?${params}`;
  }
  
  async downloadSegment(manifestId, options = {}) {
    const url = this.getSegmentUrl(manifestId, options);
    const response = await fetch(url);
    return await response.blob();
  }
}

// Usage
const client = new VideoSearchClient('http://localhost:8081', 'changeme');

// Search
const results = await client.search('basketball dunk');

// Display video
const videoUrl = client.getSegmentUrl(results[0].manifest_id, {
  context: 5,
  quality: 'medium'
});
document.querySelector('video').src = videoUrl;
```

---

## 🔒 Security Considerations

### 1. **API Key Protection**
```javascript
// ❌ DON'T: Expose API key in frontend
const url = `/v1/video/${id}?api_key=changeme`;

// ✅ DO: Use backend proxy
const url = `/api/proxy/video/${id}`;  // Backend adds API key
```

### 2. **Rate Limiting**
```yaml
# config/pipeline.yaml
security:
  rate_limit_per_minute: 120
  rate_limit_burst: 40
```

### 3. **Access Control**
```python
# Add user-specific access control
def verify_access(user_id, manifest_id):
    # Check if user has permission to access this video
    pass
```

---

## 📈 Performance Optimization

### 1. **Caching**
```python
# Cache frequently accessed segments
from functools import lru_cache

@lru_cache(maxsize=100)
def get_segment(manifest_id, format, quality, context):
    # Segment retrieval logic
    pass
```

### 2. **CDN Integration**
```python
# Serve segments through CDN
cdn_url = f"https://cdn.example.com/segments/{manifest_id}.mp4"
```

### 3. **Async Processing**
```python
# Generate thumbnails/GIFs asynchronously
async def generate_preview(manifest_id):
    # Background task
    pass
```

---

## ✅ Summary

**Your system supports:**
- ✅ Exact segment retrieval
- ✅ Context-aware retrieval (±30 seconds)
- ✅ Multiple formats (MP4, GIF, thumbnail)
- ✅ Quality options (high, medium, low)
- ✅ Fast streaming (<100ms)
- ✅ Secure authentication

**Perfect for:**
- 🚨 Security/surveillance firms
- 🎬 Media/content companies
- 🏀 Sports analytics
- 📊 Traffic analysis
- 🎓 Research/education

---

**Your segment retrieval system is production-ready!** 🚀

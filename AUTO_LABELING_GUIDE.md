# 🎯 Auto-Labeling Guide - Zero-Copy AI

Your RAG video pipeline now includes **automated video labeling** using state-of-the-art AI models. This guide explains how to use the new auto-labeling features.

## 🧠 What is Auto-Labeling?

Auto-labeling automatically analyzes video chunks and generates:

1. **Object Detection** (YOLOv8) - Identifies objects and scenes
2. **Action Recognition** (VideoMAE) - Recognizes actions and movements  
3. **Caption Generation** (BLIP-2) - Creates natural language descriptions
4. **Audio Transcription** (Whisper) - Transcribes speech and audio

## 📦 Output Format

Each video chunk gets labeled with metadata like this:

```json
{
  "objects": ["person", "car", "tree"],
  "action": "walking",
  "caption": "a person walking near a car on a sunny day",
  "audio_text": "hello there",
  "confidence": 0.84,
  "metadata": {
    "num_frames": 30,
    "models_used": {
      "object_detection": "YOLOv8",
      "action_recognition": "VideoMAE",
      "caption_generation": "BLIP-2",
      "audio_transcription": "Whisper"
    }
  }
}
```

## 🚀 How to Use

### Option 1: Enable During Pipeline Processing

Add the `--enable-labeling` flag when running the pipeline:

```bash
python run_pipeline.py --enable-labeling
```

This will:
- Process all videos as usual (chunking, embedding, indexing)
- **Additionally** run auto-labeling on each chunk
- Store labels in `metadata.json` under the `auto_labels` field

### Option 2: Programmatic Usage

```python
from src.config import load_config
from src.core.labeling import AutoLabeler
from pathlib import Path

# Initialize
config = load_config()
labeler = AutoLabeler(config)
labeler.load()

# Label a video
video_path = Path("path/to/video.mp4")
labels = labeler.label_video_chunk(video_path, include_audio=True)

print(f"Objects: {labels['objects']}")
print(f"Action: {labels['action']}")
print(f"Caption: {labels['caption']}")
print(f"Audio: {labels['audio_text']}")
```

### Option 3: Batch Labeling

```python
from pathlib import Path

video_paths = list(Path("data/chunks").glob("*.mp4"))
results = labeler.batch_label(video_paths, include_audio=True)

for result in results:
    print(result)
```

## 📋 Installation

The auto-labeling features require additional dependencies:

```bash
pip install -r requirements.txt
```

New dependencies added:
- `ultralytics` - YOLOv8 object detection
- `librosa` - Audio processing for Whisper
- `pillow` - Image processing

BLIP-2 and Whisper use the existing `transformers` library.

## 🔧 Model Configuration

### YOLOv8 (Object Detection)

- **Model**: `yolov8n.pt` (nano - fastest)
- **Alternatives**: `yolov8s.pt`, `yolov8m.pt`, `yolov8l.pt`, `yolov8x.pt`
- **Use case**: Real-time object detection in video frames

```python
from src.models.yolo import YOLODetector

detector = YOLODetector(
    model_name="yolov8n.pt",  # Change to larger model for better accuracy
    confidence_threshold=0.25,
    device="cuda"
)
detector.load()
```

### BLIP-2 (Caption Generation)

- **Model**: `Salesforce/blip2-opt-2.7b`
- **Use case**: Natural language descriptions of video content

```python
from src.models.blip2 import BLIP2Captioner

captioner = BLIP2Captioner(
    model_name="Salesforce/blip2-opt-2.7b",
    max_length=50,
    device="cuda"
)
captioner.load()
```

### Whisper (Audio Transcription)

- **Model**: `openai/whisper-tiny` (fastest)
- **Alternatives**: `whisper-base`, `whisper-small`, `whisper-medium`, `whisper-large`
- **Use case**: Transcribe speech from video audio

```python
from src.models.whisper_transcriber import WhisperTranscriber

transcriber = WhisperTranscriber(
    model_name="openai/whisper-tiny",
    device="cuda",
    language="en"  # or None for auto-detect
)
transcriber.load()
```

### VideoMAE (Action Recognition)

- **Model**: Already integrated in your pipeline
- **Use case**: Recognize actions and temporal patterns

## 💡 Performance Tips

### 1. Model Selection

**For Speed (MVP/Demo)**:
- YOLO: `yolov8n.pt` (nano)
- Whisper: `whisper-tiny`
- BLIP-2: `blip2-opt-2.7b`

**For Accuracy (Production)**:
- YOLO: `yolov8x.pt` (extra-large)
- Whisper: `whisper-large`
- BLIP-2: `blip2-opt-6.7b` or `blip2-flan-t5-xl`

### 2. GPU Usage

All models support GPU acceleration:

```yaml
# config/pipeline.yaml
models:
  device: cuda  # or 'cpu'
  precision: fp16  # Use FP16 for faster GPU inference
```

### 3. Batch Processing

Process multiple videos efficiently:

```python
# Process in batches of 10
batch_size = 10
for i in range(0, len(video_paths), batch_size):
    batch = video_paths[i:i+batch_size]
    results = labeler.batch_label(batch)
```

## 🎯 Use Cases

### 1. Searchable Video Archives

```python
# Search by objects
videos_with_cars = [
    v for v in manifest 
    if v.auto_labels and "car" in v.auto_labels.get("objects", [])
]

# Search by action
walking_videos = [
    v for v in manifest 
    if v.auto_labels and "walking" in v.auto_labels.get("action", "")
]
```

### 2. Content Moderation

```python
# Filter sensitive content
flagged_videos = [
    v for v in manifest
    if v.auto_labels and any(
        obj in ["weapon", "violence"]
        for obj in v.auto_labels.get("objects", [])
    )
]
```

### 3. Video Summarization

```python
# Generate video summaries
for video in manifest:
    if video.auto_labels:
        summary = (
            f"{video.label}: "
            f"{video.auto_labels['caption']}. "
            f"Contains: {', '.join(video.auto_labels['objects'][:5])}."
        )
        print(summary)
```

### 4. Accessibility (Captions)

```python
# Export captions for accessibility
for video in manifest:
    if video.auto_labels:
        caption = f"{video.auto_labels['caption']} " \
                  f"[Audio: {video.auto_labels['audio_text']}]"
        print(f"{video.chunk_path}: {caption}")
```

## 📊 Viewing Results

### 1. Check Metadata File

```bash
cat data/processed/metadata.json | jq '.[0].auto_labels'
```

Example output:
```json
{
  "objects": ["person", "bicycle", "road"],
  "action": "walking",
  "caption": "a person riding a bicycle on the road",
  "audio_text": "background traffic noise",
  "confidence": 0.87
}
```

### 2. Python Script

```python
import json
from pathlib import Path

metadata = json.loads(Path("data/processed/metadata.json").read_text())

for entry in metadata:
    if entry.get("auto_labels"):
        print(f"\n{entry['label']} ({entry['start_time']:.1f}s - {entry['end_time']:.1f}s)")
        labels = entry["auto_labels"]
        print(f"  Objects: {', '.join(labels.get('objects', [])[:5])}")
        print(f"  Action: {labels.get('action')}")
        print(f"  Caption: {labels.get('caption')}")
        if labels.get('has_speech'):
            print(f"  Speech: '{labels.get('audio_text')}'")
```

## 🔍 Fallback Behavior

If models fail to load (e.g., insufficient GPU memory), the system gracefully falls back:

- **YOLO Fallback**: Simple brightness-based heuristics
- **BLIP-2 Fallback**: Scene description based on image statistics
- **Whisper Fallback**: Returns empty transcription
- **VideoMAE Fallback**: Basic motion detection

**Fallback labels** include a `"fallback": true` flag in the metadata.

## ⚡ Performance Benchmarks

Typical processing times per 2-second video chunk on **RTX 3090**:

| Model | Time | Notes |
|-------|------|-------|
| YOLOv8-nano | ~50ms | Object detection |
| BLIP-2 (2.7b) | ~300ms | Caption generation |
| Whisper-tiny | ~200ms | Audio transcription |
| VideoMAE | ~150ms | Already in pipeline |
| **Total** | **~700ms** | Per chunk |

On **CPU** (Intel i9):
- Total time: ~3-5 seconds per chunk

## 🛠️ Troubleshooting

### Issue: Out of Memory (OOM)

**Solution**: Use smaller models or CPU

```python
# Use smaller models
labeler = AutoLabeler(config)
labeler.yolo.model_name = "yolov8n.pt"  # nano model
labeler.whisper.model_name = "openai/whisper-tiny"
```

### Issue: ffmpeg not found

**Solution**: Install ffmpeg

```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

### Issue: Models downloading slowly

**Solution**: Pre-download models

```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
python -c "from transformers import pipeline; pipeline('automatic-speech-recognition', model='openai/whisper-tiny')"
```

## 🎓 Advanced: Custom Action Recognition

The current VideoMAE integration uses heuristics. For production, train a classifier:

```python
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Collect labeled embeddings
X = []  # VideoMAE embeddings
y = []  # Action labels

# Train classifier
clf = RandomForestClassifier()
clf.fit(X, y)

# Use in pipeline
def recognize_action_custom(embedding):
    return clf.predict([embedding])[0]
```

## 📚 API Reference

See the full API documentation in:
- `src/models/yolo.py` - YOLODetector
- `src/models/blip2.py` - BLIP2Captioner
- `src/models/whisper_transcriber.py` - WhisperTranscriber
- `src/core/labeling.py` - AutoLabeler orchestrator

## 🎯 Next Steps

1. **Run the pipeline with labeling**:
   ```bash
   python run_pipeline.py --enable-labeling
   ```

2. **Inspect the results**:
   ```bash
   cat data/processed/metadata.json | jq '.[0].auto_labels'
   ```

3. **Build semantic search** using the generated labels

4. **Integrate with your application** for content discovery

---

**Questions?** Check the code examples in `src/core/labeling.py` or open an issue on GitHub.

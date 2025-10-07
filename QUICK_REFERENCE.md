# 🚀 Quick Reference - Zero-Copy AI Auto-Labeling

## 📦 Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install ffmpeg (required for audio extraction)
# Ubuntu/Debian: sudo apt-get install ffmpeg
# macOS: brew install ffmpeg
# Windows: Download from https://ffmpeg.org/
```

## 🎬 Basic Commands

### Run Pipeline

```bash
# Without auto-labeling (original functionality)
python run_pipeline.py

# With auto-labeling (new feature)
python run_pipeline.py --enable-labeling
```

### Test Auto-Labeling

```bash
# Test on single video
python test_auto_labeling.py --mode single

# Batch test
python test_auto_labeling.py --mode batch

# View labels in manifest
python test_auto_labeling.py --mode view

# Search by labels
python test_auto_labeling.py --mode search

# Run all tests
python test_auto_labeling.py --mode all
```

### View Examples

```bash
# All examples
python example_auto_labeling.py

# Specific example
python example_auto_labeling.py --example 1
```

## 💻 Python API

### Basic Usage

```python
from pathlib import Path
from src.config import load_config
from src.core.labeling import AutoLabeler

# Initialize
config = load_config()
labeler = AutoLabeler(config)
labeler.load()

# Label a video
labels = labeler.label_video_chunk(
    Path("video.mp4"),
    include_audio=True
)

print(labels['objects'])    # ['person', 'car']
print(labels['action'])     # 'walking'
print(labels['caption'])    # 'a person walking...'
print(labels['audio_text']) # 'hello there'
```

### Batch Processing

```python
# Process multiple videos
videos = list(Path("data/chunks").glob("*.mp4"))
results = labeler.batch_label(videos, include_audio=True)
```

### Search by Labels

```python
import json

manifest = json.loads(Path("data/processed/metadata.json").read_text())

# Find videos with people
person_videos = [
    v for v in manifest
    if v.get('auto_labels') and 'person' in v['auto_labels']['objects']
]
```

## 🌐 REST API

### Start Server

```bash
python run_api.py
# Or with uvicorn directly:
uvicorn src.api.server:build_app --factory --host 0.0.0.0 --port 8081
```

### Auto-Label Endpoint

```bash
# Label a video by manifest ID
curl -X POST http://localhost:8081/v1/label/auto \
  -H "Content-Type: application/json" \
  -H "x-api-key: changeme" \
  -d '{
    "manifest_id": "your-manifest-id",
    "include_audio": true
  }'
```

### Search Endpoint (now with labels)

```bash
curl -X POST http://localhost:8081/v1/search/similar \
  -H "Content-Type: application/json" \
  -H "x-api-key: changeme" \
  -d '{
    "query": "person waving",
    "options": {"top_k": 5}
  }'
```

## 📊 View Results

### Check Metadata

```bash
# View first entry with labels
cat data/processed/metadata.json | jq '.[0].auto_labels'

# Count labeled videos
cat data/processed/metadata.json | jq '[.[] | select(.auto_labels != null)] | length'

# Show all objects detected
cat data/processed/metadata.json | jq '[.[].auto_labels.objects[] // empty] | unique'
```

### Python Script

```python
import json
from pathlib import Path

manifest = json.loads(Path("data/processed/metadata.json").read_text())

for entry in manifest[:5]:
    if entry.get('auto_labels'):
        labels = entry['auto_labels']
        print(f"\n{entry['label']}")
        print(f"  Objects: {', '.join(labels['objects'][:5])}")
        print(f"  Caption: {labels['caption']}")
```

## ⚙️ Model Configuration

### Fast (Demo/Prototype)

```python
from src.models.yolo import YOLODetector
from src.models.whisper_transcriber import WhisperTranscriber

yolo = YOLODetector(model_name="yolov8n.pt")  # nano
whisper = WhisperTranscriber(model_name="openai/whisper-tiny")
```

### Accurate (Production)

```python
yolo = YOLODetector(model_name="yolov8x.pt")  # extra-large
whisper = WhisperTranscriber(model_name="openai/whisper-large")
```

### Device Selection

```yaml
# config/pipeline.yaml
models:
  device: cuda  # or 'cpu'
  precision: fp16  # FP16 for faster GPU inference
```

## 🔍 Common Queries

### Find Videos with Specific Objects

```python
# Find videos with cars
car_videos = [v for v in manifest 
              if v.get('auto_labels') 
              and 'car' in v['auto_labels']['objects']]
```

### Find Videos by Action

```python
# Find walking scenes
walking_videos = [v for v in manifest 
                  if v.get('auto_labels') 
                  and 'walk' in v['auto_labels']['action'].lower()]
```

### Search Captions

```python
# Search captions for keywords
outdoor_videos = [v for v in manifest 
                  if v.get('auto_labels') 
                  and 'outdoor' in v['auto_labels']['caption'].lower()]
```

### Find Videos with Speech

```python
# Videos with transcribed audio
speech_videos = [v for v in manifest 
                 if v.get('auto_labels') 
                 and v['auto_labels'].get('has_speech')]
```

## 🐛 Troubleshooting

### Out of Memory

```bash
# Use CPU instead of GPU
export RAG_MODELS__DEVICE=cpu
python run_pipeline.py --enable-labeling
```

### Models Not Loading

```bash
# Pre-download models
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
python -c "from transformers import pipeline; pipeline('automatic-speech-recognition', model='openai/whisper-tiny')"
```

### ffmpeg Not Found

```bash
# Install ffmpeg
sudo apt-get install ffmpeg  # Ubuntu/Debian
brew install ffmpeg          # macOS
```

### Check Logs

```bash
# View logs
tail -f logs/app.log

# Or check pipeline output
python run_pipeline.py --enable-labeling 2>&1 | tee pipeline.log
```

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [AUTO_LABELING_GUIDE.md](AUTO_LABELING_GUIDE.md) | Complete user guide |
| [ZERO_COPY_AI_IMPLEMENTATION.md](ZERO_COPY_AI_IMPLEMENTATION.md) | Implementation details |
| [example_auto_labeling.py](example_auto_labeling.py) | Code examples |
| [test_auto_labeling.py](test_auto_labeling.py) | Test suite |
| [README.md](README.md) | Main documentation |

## 🎯 Performance

### GPU (RTX 3090)

- ~700ms per 2-second chunk
- 8-9GB GPU RAM required

### CPU (Intel i9)

- ~3-5 seconds per 2-second chunk
- 4-8GB RAM required

## 🔗 Useful Links

- **API Docs**: http://localhost:8081/docs
- **Web UI**: http://localhost:8081/demo
- **Metrics**: http://localhost:8081/metrics
- **YOLOv8**: https://github.com/ultralytics/ultralytics
- **BLIP-2**: https://huggingface.co/Salesforce/blip2-opt-2.7b
- **Whisper**: https://github.com/openai/whisper

## 💡 Tips

1. **Start Small**: Test on a few videos first with `yolov8n.pt` (nano model)
2. **Use GPU**: 5-10x faster than CPU
3. **Batch Processing**: Process multiple videos together for efficiency
4. **Pre-download Models**: Save time by downloading models beforehand
5. **Check Logs**: Use logging to debug issues

## 🎉 Quick Start Checklist

- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Install ffmpeg
- [ ] Run pipeline: `python run_pipeline.py --enable-labeling`
- [ ] Test: `python test_auto_labeling.py --mode single`
- [ ] View results: `cat data/processed/metadata.json | jq '.[0].auto_labels'`
- [ ] Start API: `python run_api.py`
- [ ] Try examples: `python example_auto_labeling.py`

---

**Need more help?** See [AUTO_LABELING_GUIDE.md](AUTO_LABELING_GUIDE.md) for detailed documentation.

# RAG Video Search Pipeline

A production-ready Retrieval-Augmented Generation (RAG) system for video search.
It ingests labeled videos, chunks and encodes them with neural codecs, extracts
multi-modal embeddings, builds FAISS vector indices, and serves a conversational
search API with a lightweight web UI. The project ships with observability,
security, deployment manifests, and CI/CD automation.

## Features

- **Dataset ingestion** for folder-based labeled video corpora.
- **Neural chunking** with configurable duration, frame-rate, and motion-aware
  codec optimized for storage and search.
- **Multi-modal embeddings** using CLIP, VideoMAE, and Video Swin transformers.
- **Vector indexing** with FAISS for millisecond similarity search.
- **Conversational retrieval** with query expansion, history awareness, and
  streaming responses via FastAPI.
- **Security** via API key authentication and token bucket rate limiting.
- **Observability** with Prometheus metrics and structured JSON logging.
- **Deployment** ready through Docker, docker-compose, and Kubernetes manifests.
- **GPU support** with mixed-precision inference and optional quantization.
- **Comprehensive tests** and GitHub Actions CI pipeline.

## Architecture Blueprint

The platform follows a layered blueprint that is captured in detail inside
[`docs/architecture.md`](docs/architecture.md). At a glance:

- **Ingest & Control Plane** — API gateway, authentication, ingest, and manifest
  services coordinate uploads and enforce tenancy controls.
- **AI Codec Plane** — Encoder/decoder services backed by versioned model and
  codebook registries turn media into discrete latent tokens and reconstruct
  ROI payloads on demand.
- **Storage & Indexing Plane** — Object storage, Postgres manifests, vector
  indices, and Redis caches persist tokens and power low-latency retrieval.
- **Query & Analytics Plane** — Query planner, latent analytics, and ROI planner
  orchestrate search, anomaly detection, and minimum-byte decode plans.
- **Training & Operations Plane** — Offline training, data pipelines, evaluators,
  observability, billing, and metering keep the system continuously improving
  and production-ready.

## Repository Structure

```
├── config/
│   └── pipeline.yaml         # Primary runtime configuration
├── docker/
│   ├── Dockerfile            # Production container image
│   └── docker-compose.yaml   # Local multi-service setup
├── k8s/
│   ├── configmap.yaml        # Runtime configuration for Kubernetes
│   ├── deployment.yaml       # API + worker deployment spec
│   └── service.yaml          # Load-balanced service definition
├── scripts/
│   └── bootstrap_project.py  # Generates the repository skeleton
├── src/
│   ├── api/                  # FastAPI app, auth, rate limiting
│   ├── config/               # Pydantic-based configuration loader
│   ├── core/                 # Chunking, embeddings, codecs, indexing, retrieval
│   ├── models/               # Wrapper classes for ML models
│   └── utils/                # Logging, monitoring, helper utilities
├── tests/                    # Pytest-based unit and integration tests
├── web/static/               # Front-end assets
├── run_pipeline.py           # Orchestrates chunking → embeddings → index build
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.10+
- Optional: NVIDIA GPU with CUDA 12 for accelerated inference
- Docker (for containerized deployment)
- kubectl and Helm (for Kubernetes deployment)

### Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The repository ships with a `scripts/bootstrap_project.py` helper that can
re-create the layout if you start from an empty directory.

### Configuration

All runtime parameters are defined in `config/pipeline.yaml`. Key sections:

- `data`: locations for raw and processed videos plus chunking settings.
- `models`: names of CLIP, VideoMAE, and Video Swin checkpoints.
- `codec`: quantization and motion estimation parameters for neural compression.
- `index`: FAISS configuration and refresh intervals.
- `api`: server host/port, worker counts, and query expansion parameters.
- `security`: API keys and rate limits.
- `monitoring`: Prometheus exporter ports and intervals.
- `logging`: JSON logging options and destinations.

Override any value using environment variables with the `RAG_` prefix (see
`src/config/settings.py` for details).

### Preparing the Dataset

Organize your data as directories where the folder name is the label:

```
datasets/
├── sports/
│   ├── clip1.mp4
│   └── clip2.mp4
└── news/
    ├── bulletin1.mp4
    └── interview.mp4
```

Update `config/pipeline.yaml` with the `data.root_dir` path pointing to your
`datasets` folder. Processed chunks and embeddings will be stored under
`data.processed_dir`.

#### Example: Using a Windows dataset path

If your labeled videos live at `C:\Users\kvaru\Downloads\ground_clips`,
follow these exact steps:

1. Ensure the directory contains sub-folders for each label (create them if
   necessary) and move the respective videos inside each label folder.
2. Open `config/pipeline.yaml` and set
   ```yaml
   data:
     root_dir: "C:/Users/kvaru/Downloads/ground_clips"
     processed_dir: "C:/Users/kvaru/Downloads/ground_clips/processed"
   ```
   Use forward slashes in the YAML file so the path is interpreted correctly on
   all platforms.
3. From an activated virtual environment, run
   ```bash
   python run_pipeline.py --config config/pipeline.yaml
   ```
   The command will create the `processed` directory (if it does not already
   exist), generate manifests, tokens, and embeddings for every labeled video,
   and refresh the FAISS index.
4. Launch the API server with
   ```bash
   uvicorn src.api.server:build_app --factory --host 0.0.0.0 --port 8080
   ```
   Then use the API key defined in `config/pipeline.yaml` (under
   `security.api_keys`) when calling the endpoints or using the web UI.
5. To explore the results visually, open `web/static/index.html` in your
   browser, configure the API URL (e.g., `http://localhost:8080`) and API key in
   the UI, and issue search queries against your newly processed dataset.

These steps mirror the general instructions but provide concrete values for the
`C:\Users\kvaru\Downloads\ground_clips` dataset path.

### Running the Pipeline

The pipeline runner orchestrates chunking, embedding extraction, and index
construction:

```bash
python run_pipeline.py --config config/pipeline.yaml
```

This will:

1. Extract video chunks and persist manifests in `processed/`.
2. Encode chunks with the neural codec for efficient storage.
3. Generate embeddings via CLIP, VideoMAE, and Video Swin.
4. Build or refresh the FAISS index at `data/index/faiss.index`.

### Launching the API

```bash
uvicorn src.api.server:build_app --factory --host 0.0.0.0 --port 8080
```

Use the provided API key from `config/pipeline.yaml` to authenticate requests.
The `GET /health`, `POST /search`, and `POST /feedback` endpoints form the core
experience. The static UI can be served by any web server or directly via
`uvicorn` static files middleware (enabled in the FastAPI application).

### Front-end UI

Open `web/static/index.html` in your browser. It provides a lightweight
conversational interface with history, query expansion options, and playback of
matched video segments. Configure the API URL and key in `web/static/app.js`.

### Observability

The API exposes Prometheus metrics at `/metrics`. Structured JSON logs are
emitted to STDOUT and optionally to `logs/app.log`. Use tools like Grafana to
visualize metrics and Kibana or Loki to ingest logs.

### Research to Industry Roadmap

| Research Paper / Repo | What it Provides | MVP Feature You Can Build | Industry Disruption Angle |
| --- | --- | --- | --- |
| Ballé et al. (2018) “Variational Image Compression with a Scale Hyperprior” (arXiv:1802.01436) | Strong baseline for learned lossy compression | Base encoder–decoder for image/video streams with high compression ratios | Cuts storage bills (media, CCTV, cloud archiving) by 60–80% |
| VQ-VAE / VQ-VAE-2 (van den Oord 2017; Razavi 2019) | Discrete latent tokenization | Token store: compressed data as discrete embeddings in a database | Enables search and analytics on compressed tokens without full decompression |
| Mentzer et al. (2020) “High-Fidelity Generative Image Compression” (arXiv:2006.09965) | GAN-based perceptual reconstruction | Realistic region-of-interest (ROI) decode for visually sharp previews | Lets security, sports, or retail teams preview content fast without full downloads |
| Wu et al. (2018) “Video Compression through Image Interpolation” (arXiv:1804.06919) | Video compression via learned interpolation | Temporal transformer priors for video streams | Reduces bandwidth for CCTV/streaming while preserving analyzable structure |
| Ström et al. (2020) “Rate–Distortion–Accuracy Tradeoffs” (arXiv:2001.08758) | Multi-objective optimization for compression and task accuracy | Multi-task loss (reconstruction, perceptual metrics, downstream accuracy) | Guarantees downstream tasks (object detection, anomaly detection) aren’t harmed |
| Xie et al. (2021) “Task-Oriented Semantic Communication” (arXiv:2104.12508) | Compression optimized for downstream IoT/AI tasks | Task-aware encoders for IoT sensor streams | Enables real-time anomaly detection and forecasting with 5–10× less data |
| Choi et al. (2019) “Spatially Adaptive Image Compression using a Learned ROI Mask” (arXiv:1812.01891) | ROI-based selective compression | Partial/ROI decode APIs in your MVP | Customers only pay to decode what they need, driving major egress savings |
| Djelouah et al. (2018) “Compressed Domain Deep Learning” (arXiv:1806.00848) | Running ML directly on compressed codes | Analytics API (search, anomaly detection) operating on tokens | Enables “zero-copy AI” by keeping ML inside the storage layer |
| FAISS / Product Quantization (github.com/facebookresearch/faiss) | Efficient vector similarity search | Latent token database that supports queries without decompression | Turns raw archives into instantly searchable data lakes |
| Facebook NeuralCompression repo (github.com/facebookresearch/NeuralCompression) | Baseline implementations for neural compression | Reference code to accelerate prototyping | Saves 6–12 months of research by building on open-source foundations |

#### Pitch the End-to-End Flow

1. **Ingest** → Encode media into latent tokens with VQ-VAE and hyperprior codecs.
2. **Store** → Persist tokens in FAISS/Milvus/pgvector for fast vector search.
3. **Query** → Run search, analytics, or anomaly detection directly in latent space.
4. **Partial Decode** → Use ROI decoding to selectively reconstruct only what the user needs.
5. **Task-Aware** → Ensure compression preserves accuracy for downstream detection or forecasting tasks.

#### Industry Use-Cases

- **Video surveillance (CCTV/retail):** Store 10× less data, query for “red cars,” decode only five-second clips.
- **IoT / smart factories:** Compress time-series data 20× yet maintain equipment anomaly detection accuracy.
- **Healthcare imaging:** Maintain compressed MRI/CT archives and query for biomarkers without full retrieval.
- **Cloud archiving providers:** Offer “query without decompress” as a differentiated storage tier.

### Deployment

#### Docker

```bash
docker build -t rag-video-search -f docker/Dockerfile .
docker run --gpus all -p 8080:8080 -e RAG_CONFIG_PATH=/app/config/pipeline.yaml rag-video-search
```

For multi-service deployments (API + worker + redis for rate limiting), use
`docker/docker-compose.yaml`.

#### Kubernetes

```bash
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

Adjust replica counts, resource requests, and GPU scheduling based on your
cluster capabilities.

### CI/CD

GitHub Actions workflow in `.github/workflows/ci.yaml` runs tests, linting, and
builds Docker images on push. Customize secrets and registry endpoints before
activating.

### Testing

```bash
pytest
```

### Security

- Rotate API keys regularly and store them in secrets management solutions.
- Enable HTTPS termination via your ingress or reverse proxy.
- Configure rate limiting backends (e.g., Redis) for distributed deployments.

### Contributing

1. Fork the repo.
2. Create a feature branch.
3. Run `pytest` and ensure all checks pass.
4. Submit a PR describing your changes and testing steps.

### License

Distributed under the MIT License. See `LICENSE` for details.

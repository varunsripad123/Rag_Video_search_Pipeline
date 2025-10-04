# Architecture Blueprint

This document captures the end-to-end architecture, service contracts, and rollout
plan for the production RAG video search pipeline. It distills the platform into
operational layers, data primitives, and delivery milestones that align with the
multi-tenant, GPU-accelerated requirements of the product.

## 1. High-Level Blueprint

### Ingest & Control Plane
- **API Gateway:** REST/gRPC termination, request routing, and quota enforcement.
- **AuthN/Z:** OIDC/JWT validation plus organization/project scoped API keys.
- **Ingest Service:** Handles upload sessions, video/audio chunking, and metadata capture.
- **Manifest Service:** Source of truth for stored artifacts, lifecycle states, and ACLs.

### AI Codec Plane
- **Encoder Service:** Learned compression (motion-aware) that maps media to discrete tokens; supports batch streaming.
- **Decoder Service:** Full-frame or ROI decode routines that reconstruct bytes from stored tokens.
- **Codebook Registry:** Versioned codebooks per model and tenant with compatibility metadata.
- **Model Registry:** Tracks encoder/decoder builds, enables A/B testing, and supports rollbacks.

### Storage & Indexing Plane
- **Object Store:** S3/GCS/Azure Blob buckets for token blobs and side information.
- **Metadata DB:** Postgres tables for manifests, offsets, ACLs, billing counters.
- **Vector Index:** pgvector, Milvus, or FAISS-backed similarity search for latent embeddings.
- **Feature Cache:** Redis or Memcached tier caching hot embeddings and decoder tiles.

### Query & Analytics Plane
- **Query Planner:** Orchestrates latent search, filtering, and aggregation pipelines.
- **Latent Analytics:** Runs anomaly detection, similarity scoring, and clustering directly on tokens.
- **ROI Planner:** Maps query hits to minimal decode regions, minimizing egress.

### Training & Operations Plane
- **Offline Trainer:** PyTorch Lightning/Ray setup for continual encoder, decoder, and head training.
- **Data Pipes:** Kafka + Spark/Flink/Airflow ETL for ingest, labeling, and retraining data feeds.
- **Evaluator:** Generates rate–distortion–accuracy dashboards with regression checks.
- **Observability:** Prometheus/Grafana metrics, OpenTelemetry traces/logs.
- **Billing/Metering:** Usage event collection driving Stripe/Chargebee invoicing.

## 2. Data Model

### Manifest Row (per stored chunk/window)
```json
{
  "manifest_id": "uuid",
  "tenant_id": "uuid",
  "stream_id": "uuid",
  "t0": "2025-10-03T12:00:00Z",
  "t1": "2025-10-03T12:00:02Z",
  "codebook_id": "cb_v3_tenant42",
  "model_id": "video_encoder_v3",
  "token_uri": "s3://zcpy/tenant42/stream7/000123.tokens",
  "sideinfo_uri": "s3://.../000123.side",
  "byte_size": 18324,
  "ratio": 6.7,
  "hash": "blake3:...",
  "quality_stats": {"psnr": 34.1, "vmaf": 93.2}
}
```

### Token Blob (binary)
- **Header:** `{codebook_id, layout, dims, time_stride}`.
- **Payload:** Compressed discrete indices (e.g., uint16 per token) plus optional low-rank residuals.

### Index Row (latent search)
```json
{
  "manifest_id": "uuid",
  "t_mid": "2025-10-03T12:00:01Z",
  "embedding": [ /* float32 D */ ],
  "tags": ["warehouse_cam", "night_shift"]
}
```

## 3. Core Services

| Service | Responsibilities | Tech Choices (v1) |
| --- | --- | --- |
| API Gateway | Routing, auth, quota management | FastAPI + Envoy |
| Ingest | Chunk streams, write token/side info to object store, emit manifests | Python/Go workers, Kafka for back-pressure |
| Encoder | gRPC encode (bytes→tokens) with streaming batches | PyTorch/TensorRT on A10 or L4, Triton Inference Server |
| Decoder | Full/ROI decode by manifest/ROI | Mirrors encoder stack (PyTorch/TensorRT) |
| Manifest | Upsert/read manifests, enforce multi-tenant ACL | Postgres + FastAPI |
| Indexer | Pool embeddings from tokens and upsert vector DB | Python workers + pgvector/Milvus |
| Query Planner | Parse query, orchestrate latent search/analytics/ROI | Python/Go microservice |
| Analytics | Latent anomaly/similarity/clustering heads | PyTorch runtime, optional Ray |
| Trainer | Offline training, codebook updates, eval harness | PyTorch Lightning + Ray cluster |
| Registry | Model/codebook versioning, signatures, compatibility | MinIO/S3 + Postgres |
| Metering | Per-call counters for encode/search/decode | ClickHouse or Postgres + Kafka |
| Observability | Metrics, traces, logs fan-in | Prometheus/Grafana + OpenTelemetry |

## 4. Model Design Recipes

### Video Encoder (VQ-VAE-2 + Temporal Attention)
- **Encoder:** 3D convolutions feeding a windowed temporal transformer → latent feature map `H/16 × W/16 × D`.
- **Quantization:** Vector quantization with 4–8k entries; EMA updates for stability.
- **Auxiliary Heads:** Optional detector/segmentor enforcing task preservation.
- **Loss:** `L = a·Charbonnier + b·LPIPS + c·TemporalConsistency + d·VQCommit + e·TaskLoss`.
- **Rationale:** Discrete tokens are compact and indexable while temporal attention preserves motion fidelity.

### IoT Encoder (Temporal Transformer Autoencoder + PQ)
- **Encoder:** 1D convolutions → dilated transformer stack → latent representation `T' × D`.
- **Quantization:** Product quantization (multiple subspaces) or VQ.
- **Auxiliary Heads:** Next-window prediction and anomaly score head.
- **Loss:** `MSE + forecast_loss + contrastive_anomaly`.

## 5. Query Execution Flow

### A) Similarity / Semantic Search
1. **Client:** `POST /v1/search/similar` with text or embedding query (CLIP-like projection if text).
2. **Query Planner:** Queries vector DB to retrieve top-K manifest IDs and scores.
3. **Latent Filter (optional):** Run quick anomaly/semantic checks on tokens without decoding.
4. **ROI Planner:** Compute minimal token regions and time windows to decode.
5. **Decoder:** Return thumbnails/snippets by decoding selected tiles/seconds only.
6. **Response:** `{hits: [{manifest_id, t0, t1, roi, preview_uri}]}`.

### B) Analytics over a Time Range (No Decode)
1. **Client:** `POST /v1/aggregate/anomaly {stream_id, t_range}`.
2. **Planner:** Enumerates manifests intersecting the time range.
3. **Analytics:** Latent-domain model computes anomaly scores.
4. **Optional Decode:** Decode only segments whose scores exceed thresholds.
5. **Response:** Time series of scores and optional decode suggestions.

## 6. ROI Decode Strategy
- Tokens stored in tiled layout (`16×16` spatial tiles × temporal windows).
- ROI Planner maps `(x, y, w, h, t0, t1)` to tile sets and byte ranges.
- Decoder loads only referenced tiles plus minimal context frames.
- Motion side-info ensures temporal dependencies without full-frame fetches.

## 7. Public API (v1)
- `POST /v1/ingest/init` → `{stream_id, upload_url}` for pre-signed uploads.
- `POST /v1/ingest/chunk?stream_id=...&t0=...&t1=...` (binary payload) → `{manifest_id, ratio, codebook_id}`.
- `POST /v1/search/similar` → `[{manifest_id, score, t0, t1, roi_suggestion}]`.
- `POST /v1/aggregate/anomaly` → `[{t, score}]` per aggregation bucket.
- `POST /v1/decode` → media bytes (`mp4/png`) for requested ROI.
- `GET /v1/stats/rd_curve?stream_id=...` → RD curve samples for dashboards.
- **Authentication:** Bearer JWT with org/project claims; API keys supported for service accounts.

## 8. Tenancy, Security, and Privacy
- Org/project scoped tenancy with row-level ACL enforced in Manifest and Index queries.
- KMS-backed encryption (SSE-S3) and per-tenant envelope keys for token blobs.
- Optional federated fine-tuning via tenant adapters trained on-prem.
- Immutable audit logs capturing decode/search events with manifest IDs and requesters.

## 9. Latency & SLO Targets (MVP)
- **Ingest Encode:** ≤ 20 ms/frame at 720p on NVIDIA L4 (batch=8, mixed precision).
- **Search (top-K=50):** ≤ 80 ms p50 via pgvector HNSW with warm cache.
- **ROI Decode (1s @ 720p):** ≤ 150 ms p50 with tile payloads under 1 MB.
- **Index Upsert Lag:** < 3 seconds from ingest to searchable state.
- **Availability:** 99.9% control plane, 99.5% data plane.

## 10. Deployment Topology (Kubernetes)
- **GPU Pool:** `encoder-deploy`, `decoder-deploy` with HPA driven by QPS.
- **CPU Pool:** Ingest, manifest, query planner, indexer services.
- **Stateful Tier:** Postgres + pgvector, Redis cache, MinIO/S3 buckets.
- **Batch Nodes:** Ray cluster for training/evaluation workloads.
- **Event Bus:** Kafka topics (e.g., `ingested_chunk` → indexer, metering events).
- **Observability:** OpenTelemetry sidecars, Prometheus scraping, Grafana dashboards.

## 11. Build Order (90-Day Plan)
- **Weeks 1–3:** Implement encoder/decoder gRPC stubs, CLI for local compress/decompress, plot RD curves vs H.264/gzip.
- **Weeks 4–6:** Deploy Postgres/S3/pgvector, build indexer, ship `/search/similar`, add ROI tiling and decode path.
- **Weeks 7–9:** Add latent analytics endpoint, multi-tenant auth, metering, and baseline dashboards; harden retries/idempotency/hash checks.
- **Weeks 10–12:** Kubernetes pilot (10–50 streams, 50–200 sensors), gather storage/egress savings, accuracy deltas, and latency metrics.

## 12. Differentiators
- Search and analytics without full decompression (latent plane + ROI decode).
- Task-preserving compression via auxiliary heads and rate–distortion–accuracy optimization.
- Tenant-scoped codebooks/models enabling private adaptation.
- Open manifest specification supporting export and interoperability.

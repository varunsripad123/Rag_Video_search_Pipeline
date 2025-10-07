"""Check what dimension the actual video embeddings are."""

import numpy as np
from pathlib import Path
import json

print("Checking embedding dimensions...")
print("=" * 60)

# Load metadata
metadata_path = Path("data/processed/metadata.json")
if not metadata_path.exists():
    print("❌ Metadata not found. Run pipeline first.")
    exit(1)

with open(metadata_path) as f:
    metadata = json.load(f)

if not metadata:
    print("❌ No videos in metadata")
    exit(1)

# Check first few embeddings
print(f"\nFound {len(metadata)} videos")
print("\nChecking embedding dimensions:\n")

dims_found = set()
for i, entry in enumerate(metadata[:10]):
    emb_path = entry.get('embedding_path')
    if emb_path and Path(emb_path).exists():
        emb = np.load(emb_path)
        dims_found.add(emb.shape[0])
        print(f"  Video {i+1}: {Path(emb_path).name}")
        print(f"    Shape: {emb.shape}")
        print(f"    Norm: {np.linalg.norm(emb):.4f}")
        print(f"    Non-zero elements: {np.count_nonzero(emb)}/{emb.shape[0]}")
        print()

print("=" * 60)
print("\nSummary:")
print(f"  Unique dimensions found: {dims_found}")

if len(dims_found) == 1:
    dim = list(dims_found)[0]
    print(f"  ✅ All embeddings have dimension: {dim}")
    
    # Suggest fix
    print("\n" + "=" * 60)
    print("RECOMMENDED FIX:")
    print("=" * 60)
    
    if dim == 512:
        print("\n  Videos are CLIP-only (512 dims)")
        print("  Query embeddings should work fine.")
        print("  \n  The low scores suggest normalization issue.")
        print("  Run: python fix_search_normalization.py")
    elif dim == 1288:
        print("\n  Videos are multi-modal (1288 dims)")
        print("  Queries are CLIP-only (512 dims) + padding")
        print("  \n  This causes low similarity scores!")
        print("  \n  Options:")
        print("    1. Re-embed videos with CLIP only (RECOMMENDED)")
        print("       python re_embed_clip_only.py")
        print("    \n    2. Or: Fix query to use full multi-modal")
        print("       (Requires loading VideoMAE/VideoSwin for queries)")
    else:
        print(f"\n  Custom dimension: {dim}")
        print("  Check config/pipeline.yaml for model settings")
else:
    print(f"  ⚠️  Mixed dimensions: {dims_found}")
    print("  This will cause search problems!")
    print("  Re-run pipeline to get consistent embeddings.")

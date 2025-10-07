"""Rebuild FAISS index from existing embeddings."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import load_config
from src.core.indexing import build_index
from src.utils.io import read_manifest

def main():
    """Rebuild the index from existing embeddings."""
    config_path = Path("config/pipeline.yaml")
    config = load_config(config_path)
    
    # Read existing manifest
    manifest_path = config.data.processed_dir / "metadata.json"
    
    if not manifest_path.exists():
        print(f"❌ Manifest not found at {manifest_path}")
        print("Run the full pipeline first: python run_pipeline.py")
        return
    
    print(f"📖 Reading manifest from {manifest_path}")
    manifest = read_manifest(manifest_path)
    print(f"✅ Found {len(manifest)} entries")
    
    # Check embeddings exist
    missing = []
    for entry in manifest:
        emb_path = Path(entry.embedding_path) if isinstance(entry.embedding_path, str) else entry.embedding_path
        if not emb_path.exists():
            missing.append(emb_path)
    
    if missing:
        print(f"⚠️  Warning: {len(missing)} embeddings missing")
        print(f"First few missing: {missing[:5]}")
    
    print(f"\n🔨 Building FAISS index...")
    try:
        index_path = build_index(config, manifest)
        print(f"\n✅ Success! Index saved to: {index_path}")
        print(f"\n🎉 You can now start the API server:")
        print(f"   python run_api.py")
    except Exception as e:
        print(f"\n❌ Failed to build index: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

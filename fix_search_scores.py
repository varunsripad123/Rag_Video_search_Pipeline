"""Fix search scores by using only CLIP portion of embeddings."""

import numpy as np
from pathlib import Path
import json
from src.config import load_config

print("=" * 60)
print("Fixing Search Scores")
print("=" * 60)

config = load_config()

# Check current embedding dimension
metadata_path = config.data.processed_dir / "metadata.json"
if not metadata_path.exists():
    print("\n❌ No metadata found. Run pipeline first.")
    exit(1)

with open(metadata_path) as f:
    metadata = json.load(f)

# Check first embedding
first_emb_path = Path(metadata[0]['embedding_path'])
first_emb = np.load(first_emb_path)
video_dim = first_emb.shape[0]

print(f"\nCurrent Setup:")
print(f"  Video embeddings: {video_dim} dims")
print(f"  Query embeddings: 512 dims (CLIP) + padding")

if video_dim == 512:
    print("\n✅ Already using CLIP-only embeddings!")
    print("   The search should work. Low scores might be due to:")
    print("   1. No matching videos in dataset")
    print("   2. Embeddings not normalized")
    print("   3. FAISS index needs rebuilding")
    print("\nTry rebuilding index:")
    print("  python -c \"from src.core.indexing import build_index; from src.config import load_config; from src.utils.io import read_manifest; build_index(load_config(), read_manifest('data/processed/metadata.json'))\"")
    exit(0)

elif video_dim == 1288:
    print("\n⚠️  Videos use multi-modal embeddings (CLIP + VideoMAE + VideoSwin)")
    print("   Queries use CLIP-only")
    print("   → This causes low similarity scores!\n")
    
    print("=" * 60)
    print("FIX OPTIONS:")
    print("=" * 60)
    
    print("\n1. QUICK FIX: Use only CLIP portion (512 dims)")
    print("   - Extracts first 512 dims from video embeddings")
    print("   - Rebuilds FAISS index with 512 dims")
    print("   - Takes ~30 seconds")
    print("   - Search will work immediately")
    
    print("\n2. FULL FIX: Re-embed videos with CLIP-only")
    print("   - Re-runs pipeline with CLIP encoder only")
    print("   - Takes ~5-10 minutes for 272 videos")
    print("   - More accurate long-term")
    
    choice = input("\nChoose option (1 or 2): ").strip()
    
    if choice == "1":
        print("\n" + "=" * 60)
        print("Applying Quick Fix...")
        print("=" * 60)
        
        # Extract CLIP portion and rebuild index
        from src.core.indexing import FAISSIndex
        
        print("\nStep 1: Extracting CLIP portions from embeddings...")
        clip_embeddings = []
        for i, entry in enumerate(metadata):
            emb_path = Path(entry['embedding_path'])
            emb = np.load(emb_path)
            clip_emb = emb[:512]  # First 512 dims are CLIP
            clip_embeddings.append(clip_emb)
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(metadata)} videos...")
        
        clip_embeddings = np.array(clip_embeddings)
        print(f"  ✅ Extracted {len(clip_embeddings)} CLIP embeddings (512 dims)")
        
        print("\nStep 2: Building new FAISS index...")
        index = FAISSIndex(
            dim=512,
            nlist=config.index.nlist,
            nprobe=config.index.nprobe,
            use_gpu=config.index.use_gpu
        )
        index.train(clip_embeddings)
        index.add(clip_embeddings)
        
        print("\nStep 3: Saving index...")
        index_path = config.index.faiss_index_path
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index.save(index_path)
        
        print(f"  ✅ Saved to: {index_path}")
        
        print("\n" + "=" * 60)
        print("✅ QUICK FIX COMPLETE!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Update query embedder target_dim to 512")
        print("2. Restart API: python run_api.py")
        print("3. Test search - scores should be much higher!")
        print("\nExpected results:")
        print("  'person waving' on waving videos: 70-95%")
        print("  'random text' on any video: 10-30%")
        
        # Update config hint
        print("\n⚠️  IMPORTANT: Update src/api/server.py")
        print("   Change: target_dim: int = 1288")
        print("   To:     target_dim: int = 512")
        
    elif choice == "2":
        print("\n" + "=" * 60)
        print("Full Re-embedding")
        print("=" * 60)
        print("\nThis will:")
        print("1. Modify pipeline to use CLIP-only")
        print("2. Re-process all videos")
        print("3. Rebuild FAISS index")
        print("\nEstimated time: 5-10 minutes")
        
        confirm = input("\nProceed? (yes/no): ").strip().lower()
        if confirm == "yes":
            print("\nRun: python run_pipeline.py --clip-only")
            print("(I can add --clip-only flag if needed)")
        else:
            print("\nCancelled. Use option 1 for quick fix.")
    else:
        print("\n❌ Invalid option. Run script again.")

else:
    print(f"\n⚠️  Unexpected dimension: {video_dim}")
    print("   Check your pipeline configuration.")

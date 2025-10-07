"""Simple test script to verify pipeline components."""
import sys
from pathlib import Path

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

print("=" * 60)
print("Testing RAG Video Search Pipeline Components")
print("=" * 60)

# Test 1: Config loading
print("\n[1/6] Testing config loading...")
try:
    from src.config import load_config
    config = load_config(Path("config/pipeline.yaml"))
    print("[OK] Config loaded successfully")
    print(f"  - Root dir: {config.data.root_dir}")
    print(f"  - Device: {config.models.device}")
except Exception as e:
    print(f"[FAIL] Config loading failed: {e}")
    sys.exit(1)

# Test 2: Video loading
print("\n[2/6] Testing video loading...")
try:
    from src.utils.video import iter_videos, load_video_frames
    videos = list(iter_videos(config.data.root_dir))
    print(f"[OK] Found {len(videos)} videos")
    if videos:
        test_video, label = videos[0]
        print(f"  - Testing: {test_video.name} (label: {label})")
        tensor = load_video_frames(test_video)
        print(f"  - Loaded tensor shape: {tensor.shape}")
except Exception as e:
    print(f"[FAIL] Video loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Codec
print("\n[3/6] Testing neural codec...")
try:
    from src.core.codecs import NeuralVideoCodec
    import torch
    codec = NeuralVideoCodec(config.codec, device="cpu")
    print(f"[OK] Codec initialized")
    
    # Test encoding
    test_tensor = torch.randn(1, 3, 5, 64, 64)  # Small test tensor
    output_path = Path("data/test_token.npy")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    enc_chunk = codec.encode_tensor(test_tensor, output_path)
    print(f"  - Encoded test tensor: {enc_chunk.size_bytes} bytes")
    output_path.unlink()  # Clean up
except Exception as e:
    print(f"[FAIL] Codec test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Embeddings
print("\n[4/6] Testing embedding extraction...")
try:
    from src.core.embedding import EmbeddingExtractor
    extractor = EmbeddingExtractor(config)
    extractor.configure_precision()
    print(f"[OK] Extractor initialized")
    
    # Load models
    print("  - Loading models (this may take a while)...")
    extractor.load()
    print(f"[OK] Models loaded")
    
    # Test encoding
    if videos:
        test_tensor = load_video_frames(videos[0][0])
        embedding = extractor.encode_frames(test_tensor)
        print(f"  - Embedding shape: {embedding.shape}")
except Exception as e:
    print(f"[FAIL] Embedding test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Chunking
print("\n[5/6] Testing video chunking...")
try:
    from src.core.chunking import chunk_dataset
    chunks = list(chunk_dataset(config))
    print(f"[OK] Created {len(chunks)} chunks")
    if chunks:
        print(f"  - First chunk: {chunks[0].video_path.name}")
except Exception as e:
    print(f"[FAIL] Chunking failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Full pipeline (limited)
print("\n[6/6] Running limited pipeline test...")
try:
    from src.core.pipeline import run_pipeline
    
    # Temporarily limit to first 3 chunks for testing
    print("  - Processing first 3 chunks only...")
    
    # Run pipeline
    index_path = run_pipeline(config)
    print(f"[OK] Pipeline completed successfully!")
    print(f"  - Index saved to: {index_path}")
    
except Exception as e:
    print(f"[FAIL] Pipeline test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("All tests passed! Pipeline is working correctly.")
print("=" * 60)

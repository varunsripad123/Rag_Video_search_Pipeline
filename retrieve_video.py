"""Retrieve original video quality from storage."""
import sys
import json
import shutil
from pathlib import Path
from typing import List, Optional
import cv2
import numpy as np

def load_manifest():
    """Load the video metadata."""
    manifest_path = Path("data/processed/metadata.json")
    if not manifest_path.exists():
        print("❌ Manifest not found. Run the pipeline first.")
        sys.exit(1)
    return json.load(open(manifest_path))

def search_videos(query: str, manifest: List[dict]) -> List[dict]:
    """Search for videos matching a query."""
    results = []
    query_lower = query.lower()
    
    for entry in manifest:
        # Search in label, stream_id, and tags
        if (query_lower in entry['label'].lower() or 
            query_lower in entry['stream_id'].lower() or
            any(query_lower in tag.lower() for tag in entry.get('tags', []))):
            results.append(entry)
    
    return results

def get_video_by_id(manifest_id: str, manifest: List[dict]) -> Optional[dict]:
    """Get a specific video by manifest ID."""
    for entry in manifest:
        if entry['manifest_id'] == manifest_id:
            return entry
    return None

def retrieve_original_chunk(entry: dict, output_path: Path) -> bool:
    """Copy the original video chunk to output location."""
    source = Path(entry['chunk_path'])
    
    if not source.exists():
        print(f"❌ Original chunk not found: {source}")
        return False
    
    try:
        shutil.copy2(source, output_path)
        print(f"✅ Retrieved: {output_path}")
        print(f"   Duration: {entry['start_time']:.1f}s - {entry['end_time']:.1f}s")
        print(f"   Size: {source.stat().st_size / 1024:.1f} KB")
        return True
    except Exception as e:
        print(f"❌ Failed to copy: {e}")
        return False

def stitch_video_chunks(entries: List[dict], output_path: Path) -> bool:
    """Stitch multiple chunks from the same video stream together."""
    if not entries:
        return False
    
    # Sort by start time
    entries = sorted(entries, key=lambda x: x['start_time'])
    
    print(f"\n🎬 Stitching {len(entries)} chunks...")
    
    # Read all chunks
    all_frames = []
    fps = entries[0]['fps']
    
    for i, entry in enumerate(entries):
        chunk_path = Path(entry['chunk_path'])
        if not chunk_path.exists():
            print(f"⚠️  Chunk {i+1} not found: {chunk_path}")
            continue
        
        cap = cv2.VideoCapture(str(chunk_path))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            all_frames.append(frame)
        
        cap.release()
        print(f"   ✓ Chunk {i+1}/{len(entries)}: {len(all_frames)} frames")
    
    if not all_frames:
        print("❌ No frames to write")
        return False
    
    # Write stitched video
    h, w = all_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    
    for frame in all_frames:
        out.write(frame)
    
    out.release()
    
    print(f"✅ Stitched video saved: {output_path}")
    print(f"   Total frames: {len(all_frames)}")
    print(f"   Duration: {len(all_frames) / fps:.1f}s")
    print(f"   Size: {output_path.stat().st_size / 1024:.1f} KB")
    
    return True

def main():
    """Main retrieval interface."""
    print("\n" + "="*60)
    print("🎥 VIDEO RETRIEVAL TOOL - Original Quality")
    print("="*60 + "\n")
    
    manifest = load_manifest()
    print(f"📚 Loaded {len(manifest)} video chunks\n")
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python retrieve_video.py <query>              - Search and retrieve")
        print("  python retrieve_video.py --id <manifest_id>   - Retrieve by ID")
        print("  python retrieve_video.py --stitch <stream_id> - Stitch full video")
        print("\nExamples:")
        print("  python retrieve_video.py waving")
        print("  python retrieve_video.py --id cecc6ffb-b210-433d-b81d-e70f9877df61")
        print("  python retrieve_video.py --stitch person01_01_ground_waving")
        sys.exit(0)
    
    output_dir = Path("retrieved_videos")
    output_dir.mkdir(exist_ok=True)
    
    if sys.argv[1] == "--id" and len(sys.argv) > 2:
        # Retrieve by manifest ID
        manifest_id = sys.argv[2]
        entry = get_video_by_id(manifest_id, manifest)
        
        if not entry:
            print(f"❌ Video not found with ID: {manifest_id}")
            sys.exit(1)
        
        output_path = output_dir / Path(entry['chunk_path']).name
        retrieve_original_chunk(entry, output_path)
    
    elif sys.argv[1] == "--stitch" and len(sys.argv) > 2:
        # Stitch all chunks from a stream
        stream_id = sys.argv[2]
        chunks = [e for e in manifest if e['stream_id'] == stream_id]
        
        if not chunks:
            print(f"❌ No chunks found for stream: {stream_id}")
            sys.exit(1)
        
        output_path = output_dir / f"{stream_id}_full.mp4"
        stitch_video_chunks(chunks, output_path)
    
    else:
        # Search by query
        query = " ".join(sys.argv[1:])
        results = search_videos(query, manifest)
        
        if not results:
            print(f"❌ No videos found matching: '{query}'")
            sys.exit(1)
        
        print(f"🔍 Found {len(results)} matching chunks:\n")
        
        for i, entry in enumerate(results[:10], 1):  # Show first 10
            print(f"{i}. {Path(entry['chunk_path']).name}")
            print(f"   Label: {entry['label']}")
            print(f"   Stream: {entry['stream_id']}")
            print(f"   Time: {entry['start_time']:.1f}s - {entry['end_time']:.1f}s")
            print(f"   ID: {entry['manifest_id']}\n")
        
        # Ask which to retrieve
        if len(results) == 1:
            choice = 1
        else:
            try:
                choice = int(input(f"Which video to retrieve? (1-{min(len(results), 10)}, 0=all): "))
            except (ValueError, KeyboardInterrupt):
                print("\n❌ Cancelled")
                sys.exit(0)
        
        if choice == 0:
            # Retrieve all
            for entry in results:
                output_path = output_dir / Path(entry['chunk_path']).name
                retrieve_original_chunk(entry, output_path)
        else:
            # Retrieve specific one
            entry = results[choice - 1]
            output_path = output_dir / Path(entry['chunk_path']).name
            retrieve_original_chunk(entry, output_path)
    
    print(f"\n📁 Output directory: {output_dir.resolve()}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()

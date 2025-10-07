"""
Demo: Search using auto-generated labels

This script demonstrates the enhanced search functionality that uses
auto-generated labels (objects, actions, captions) for filtering and ranking.
"""

import json
from pathlib import Path
import requests


def demo_1_basic_search():
    """Demo 1: Basic search (returns auto-labels in results)"""
    print("=" * 80)
    print("DEMO 1: Basic Search with Auto-Labels")
    print("=" * 80)
    
    # Using the API
    response = requests.post(
        "http://localhost:8081/v1/search/similar",
        headers={"x-api-key": "changeme"},
        json={
            "query": "person waving",
            "options": {
                "top_k": 5
            }
        }
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"\nQuery: 'person waving'")
        print(f"Answer: {data['answer']}\n")
        
        for i, result in enumerate(data['results'], 1):
            print(f"{i}. {result['label']} ({result['start_time']:.1f}s - {result['end_time']:.1f}s)")
            print(f"   Score: {result['score']:.3f}")
            
            if result.get('auto_labels'):
                labels = result['auto_labels']
                print(f"   Objects: {', '.join(labels.get('objects', [])[:5])}")
                print(f"   Action: {labels.get('action', 'unknown')}")
                print(f"   Caption: {labels.get('caption', '')[:60]}...")
            print()
    else:
        print(f"Error: {response.status_code}")
        print("Make sure the API server is running: python run_api.py")


def demo_2_filter_by_objects():
    """Demo 2: Filter search results by detected objects"""
    print("\n" + "=" * 80)
    print("DEMO 2: Filter by Objects")
    print("=" * 80)
    
    # Search only for videos containing "person"
    response = requests.post(
        "http://localhost:8081/v1/search/similar",
        headers={"x-api-key": "changeme"},
        json={
            "query": "video content",
            "options": {
                "top_k": 5,
                "filter_objects": ["person"]  # Only return videos with people
            }
        }
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"\nQuery: 'video content' (filtered to only show videos with 'person')")
        print(f"Found: {len(data['results'])} results\n")
        
        for i, result in enumerate(data['results'], 1):
            labels = result.get('auto_labels', {})
            objects = labels.get('objects', [])
            print(f"{i}. {result['label']}")
            print(f"   Objects: {', '.join(objects)}")
            print(f"   ✓ Contains 'person': {'person' in objects}")
            print()


def demo_3_filter_by_action():
    """Demo 3: Filter search results by recognized action"""
    print("\n" + "=" * 80)
    print("DEMO 3: Filter by Action")
    print("=" * 80)
    
    # Search only for "walking" videos
    response = requests.post(
        "http://localhost:8081/v1/search/similar",
        headers={"x-api-key": "changeme"},
        json={
            "query": "movement",
            "options": {
                "top_k": 5,
                "filter_action": "walking"  # Only return walking videos
            }
        }
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"\nQuery: 'movement' (filtered to only show 'walking' action)")
        print(f"Found: {len(data['results'])} results\n")
        
        for i, result in enumerate(data['results'], 1):
            labels = result.get('auto_labels', {})
            action = labels.get('action', 'unknown')
            print(f"{i}. {result['label']}")
            print(f"   Action: {action}")
            print(f"   ✓ Contains 'walking': {'walk' in action.lower()}")
            print()


def demo_4_filter_by_confidence():
    """Demo 4: Filter by minimum confidence score"""
    print("\n" + "=" * 80)
    print("DEMO 4: Filter by Confidence Score")
    print("=" * 80)
    
    # Only return high-confidence results
    response = requests.post(
        "http://localhost:8081/v1/search/similar",
        headers={"x-api-key": "changeme"},
        json={
            "query": "any content",
            "options": {
                "top_k": 5,
                "min_confidence": 0.7  # Only return results with 70%+ confidence
            }
        }
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"\nQuery: 'any content' (filtered to confidence >= 0.7)")
        print(f"Found: {len(data['results'])} results\n")
        
        for i, result in enumerate(data['results'], 1):
            labels = result.get('auto_labels', {})
            confidence = labels.get('confidence', 0.0)
            print(f"{i}. {result['label']}")
            print(f"   Confidence: {confidence:.1%}")
            print(f"   ✓ High confidence: {confidence >= 0.7}")
            print()


def demo_5_combined_filters():
    """Demo 5: Combine multiple filters"""
    print("\n" + "=" * 80)
    print("DEMO 5: Combined Filters")
    print("=" * 80)
    
    # Search with multiple filters
    response = requests.post(
        "http://localhost:8081/v1/search/similar",
        headers={"x-api-key": "changeme"},
        json={
            "query": "person activity",
            "options": {
                "top_k": 5,
                "filter_objects": ["person"],  # Must contain person
                "filter_action": "walking",    # Must be walking
                "min_confidence": 0.6          # Must have 60%+ confidence
            }
        }
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"\nQuery: 'person activity'")
        print("Filters:")
        print("  • Must contain: person")
        print("  • Action: walking")
        print("  • Min confidence: 0.6")
        print(f"\nFound: {len(data['results'])} matching results\n")
        
        for i, result in enumerate(data['results'], 1):
            labels = result.get('auto_labels', {})
            print(f"{i}. {result['label']}")
            print(f"   Objects: {', '.join(labels.get('objects', []))}")
            print(f"   Action: {labels.get('action', 'unknown')}")
            print(f"   Confidence: {labels.get('confidence', 0):.1%}")
            print()


def demo_6_programmatic_search():
    """Demo 6: Programmatic search using Python API"""
    print("\n" + "=" * 80)
    print("DEMO 6: Programmatic Search")
    print("=" * 80)
    
    try:
        from src.config import load_config
        from src.core.retrieval import Retriever
        from src.api.server import QueryEmbedder
        from transformers import CLIPModel, CLIPTokenizer
        import torch
        
        # Load config and initialize
        config = load_config()
        metadata_path = config.data.processed_dir / "metadata.json"
        
        if not metadata_path.exists():
            print("⚠️  Metadata not found. Run pipeline first.")
            return
        
        # Initialize retriever
        retriever = Retriever(config, metadata_path)
        
        # Initialize query embedder
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            clip_model = CLIPModel.from_pretrained(config.models.clip_model_name).to(device)
            clip_tokenizer = CLIPTokenizer.from_pretrained(config.models.clip_model_name)
            embedder = QueryEmbedder(model=clip_model, tokenizer=clip_tokenizer, device=device)
        except Exception as e:
            print(f"⚠️  Failed to load CLIP model: {e}")
            print("   Using fallback embedding...")
            embedder = QueryEmbedder(model=None, tokenizer=None, device=device)
        
        # Search with filters
        query = "person waving"
        query_embedding = embedder.embed(query)
        
        results = retriever.search(
            query_embedding,
            top_k=5,
            filter_objects=["person"],
            min_confidence=0.3  # Lower threshold for fallback labels
        )
        
        print(f"\nQuery: '{query}'")
        print(f"Found: {len(results)} results\n")
        
        for i, result in enumerate(results, 1):
            print(f"{i}. {result.label} (score: {result.score:.3f})")
            if result.auto_labels:
                print(f"   Objects: {', '.join(result.auto_labels.get('objects', []))}")
                print(f"   Caption: {result.auto_labels.get('caption', '')[:50]}...")
            print()
    except Exception as e:
        print(f"⚠️  Error: {e}")
        print("   This demo requires CLIP model and metadata.")


def demo_7_direct_metadata_search():
    """Demo 7: Search metadata directly using Python"""
    print("\n" + "=" * 80)
    print("DEMO 7: Direct Metadata Search")
    print("=" * 80)
    
    # Load metadata directly
    metadata_path = Path("data/processed/metadata.json")
    
    if not metadata_path.exists():
        print("⚠️  Metadata not found")
        return
    
    manifest = json.loads(metadata_path.read_text())
    
    # Search for videos with specific objects
    print("\n🔍 Videos containing 'person':")
    person_videos = [
        v for v in manifest
        if v.get('auto_labels') and 'person' in v['auto_labels'].get('objects', [])
    ]
    print(f"   Found: {len(person_videos)} videos")
    
    # Search for videos with specific actions
    print("\n🔍 Videos with 'walking' action:")
    walking_videos = [
        v for v in manifest
        if v.get('auto_labels') and 'walk' in v['auto_labels'].get('action', '').lower()
    ]
    print(f"   Found: {len(walking_videos)} videos")
    
    # Search captions
    print("\n🔍 Videos with 'outdoor' in caption:")
    outdoor_videos = [
        v for v in manifest
        if v.get('auto_labels') and 'outdoor' in v['auto_labels'].get('caption', '').lower()
    ]
    print(f"   Found: {len(outdoor_videos)} videos")
    
    # Show statistics
    labeled = [v for v in manifest if v.get('auto_labels')]
    print(f"\n📊 Statistics:")
    print(f"   Total videos: {len(manifest)}")
    print(f"   With auto-labels: {len(labeled)}")
    print(f"   Coverage: {len(labeled)/len(manifest)*100:.1f}%")


def main():
    """Run all demos"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Demo search with auto-labels")
    parser.add_argument(
        "--demo",
        type=int,
        choices=[1, 2, 3, 4, 5, 6, 7],
        help="Run specific demo (1-7), or all if not specified"
    )
    parser.add_argument(
        "--api",
        action="store_true",
        help="Run API demos (requires API server running)"
    )
    args = parser.parse_args()
    
    api_demos = {
        1: demo_1_basic_search,
        2: demo_2_filter_by_objects,
        3: demo_3_filter_by_action,
        4: demo_4_filter_by_confidence,
        5: demo_5_combined_filters,
    }
    
    local_demos = {
        6: demo_6_programmatic_search,
        7: demo_7_direct_metadata_search,
    }
    
    try:
        if args.demo:
            # Run specific demo
            if args.demo in api_demos:
                api_demos[args.demo]()
            else:
                local_demos[args.demo]()
        elif args.api:
            # Run only API demos
            for demo_func in api_demos.values():
                demo_func()
        else:
            # Run all demos
            print("Running local demos (no API required)...\n")
            for demo_func in local_demos.values():
                demo_func()
            
            print("\n" + "=" * 80)
            print("To run API demos, start the server and use --api flag:")
            print("  python run_api.py")
            print("  python demo_search_with_labels.py --api")
            print("=" * 80)
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("  • For API demos: Make sure server is running (python run_api.py)")
        print("  • For local demos: Make sure metadata.json exists")
        print("  • Run pipeline first: python run_pipeline.py --enable-labeling")


if __name__ == "__main__":
    main()

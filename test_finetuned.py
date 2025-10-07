"""Test fine-tuned model accuracy vs baseline.

Usage:
    python test_finetuned.py --client acme
"""

import argparse
from pathlib import Path
import sys

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.finetuning.trainer import ClientFineTuner
from transformers import CLIPModel, CLIPProcessor
from src.config import load_config
from src.core.retrieval import Retriever


def test_model_accuracy(model, processor, test_queries, retriever, device="cuda"):
    """Test model on queries."""
    model.eval()
    results = []
    
    for query, expected_label in test_queries:
        # Get query embedding
        inputs = processor(text=query, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            text_embeds = model.get_text_features(**inputs)
            query_emb = text_embeds.cpu().numpy()
            query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        
        # Search
        search_results = retriever.search(query_emb, top_k=5)
        
        if search_results:
            top_result = search_results[0]
            score = top_result.score * 100
            correct = top_result.label == expected_label
            results.append({
                'query': query,
                'expected': expected_label,
                'got': top_result.label,
                'score': score,
                'correct': correct
            })
        else:
            results.append({
                'query': query,
                'expected': expected_label,
                'got': None,
                'score': 0,
                'correct': False
            })
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Test fine-tuned model")
    parser.add_argument("--client", required=True, help="Client name")
    parser.add_argument("--model-dir", default="models/finetuned", help="Model directory")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🧪 FINE-TUNED MODEL EVALUATION")
    print("=" * 70)
    print()
    
    model_path = Path(args.model_dir) / args.client
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print(f"   Run: python finetune_client.py --client {args.client}")
        return
    
    # Load config
    config = load_config()
    
    # Load retriever
    print("📊 Loading retriever...")
    retriever = Retriever(config, config.data.processed_dir / "metadata.json")
    print("✅ Retriever loaded")
    print()
    
    # Test queries (customize based on your dataset)
    test_queries = [
        ("basketball", "Basketball"),
        ("running", "Running"),
        ("cycling", "Biking"),
        ("playing guitar", "PlayingGuitar"),
        ("dancing", "IceDancing"),
        ("swimming", "BreastStroke"),
        ("archery", "Archery"),
        ("drumming", "Drumming"),
    ]
    
    # Test baseline model
    print("🔍 Testing BASELINE model...")
    print()
    
    baseline_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    baseline_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    baseline_model.to("cuda" if torch.cuda.is_available() else "cpu")
    
    baseline_results = test_model_accuracy(
        baseline_model, baseline_processor, test_queries, retriever
    )
    
    # Test fine-tuned model
    print("🔍 Testing FINE-TUNED model...")
    print()
    
    finetuned_model, finetuned_processor = ClientFineTuner.load_finetuned(model_path)
    
    finetuned_results = test_model_accuracy(
        finetuned_model, finetuned_processor, test_queries, retriever
    )
    
    # Compare results
    print("=" * 70)
    print("📊 COMPARISON RESULTS")
    print("=" * 70)
    print()
    
    for baseline, finetuned in zip(baseline_results, finetuned_results):
        query = baseline['query']
        baseline_score = baseline['score']
        finetuned_score = finetuned['score']
        improvement = finetuned_score - baseline_score
        
        print(f"📝 Query: '{query}'")
        print(f"   Baseline:   {baseline_score:.1f}% {'✅' if baseline['correct'] else '❌'}")
        print(f"   Fine-tuned: {finetuned_score:.1f}% {'✅' if finetuned['correct'] else '❌'}")
        print(f"   Change:     {improvement:+.1f}%")
        
        if improvement > 5:
            print(f"   Status:     🎯 Significant improvement!")
        elif improvement > 0:
            print(f"   Status:     ✅ Better")
        elif improvement < -5:
            print(f"   Status:     ⚠️  Worse")
        else:
            print(f"   Status:     ➡️  Similar")
        print()
    
    # Summary
    baseline_avg = np.mean([r['score'] for r in baseline_results])
    finetuned_avg = np.mean([r['score'] for r in finetuned_results])
    avg_improvement = finetuned_avg - baseline_avg
    
    baseline_correct = sum(r['correct'] for r in baseline_results)
    finetuned_correct = sum(r['correct'] for r in finetuned_results)
    
    print("=" * 70)
    print("📈 SUMMARY")
    print("=" * 70)
    print()
    print(f"Average accuracy:")
    print(f"  Baseline:   {baseline_avg:.1f}%")
    print(f"  Fine-tuned: {finetuned_avg:.1f}%")
    print(f"  Improvement: {avg_improvement:+.1f}%")
    print()
    print(f"Correct predictions:")
    print(f"  Baseline:   {baseline_correct}/{len(test_queries)}")
    print(f"  Fine-tuned: {finetuned_correct}/{len(test_queries)}")
    print()
    
    if avg_improvement > 10:
        print("🎉 EXCELLENT: Fine-tuning significantly improved accuracy!")
    elif avg_improvement > 5:
        print("✅ GOOD: Fine-tuning helped!")
    elif avg_improvement > 0:
        print("✅ SLIGHT: Small improvement")
    else:
        print("⚠️  WARNING: Fine-tuning didn't help. May need more data or epochs.")
    
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()

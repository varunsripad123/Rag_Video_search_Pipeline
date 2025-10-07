"""Quick test to see if CLIP can load without circular import."""

print("Testing CLIP loading...")
print("=" * 60)

# Test 1: Direct import from transformers.models.clip
print("\n1. Trying: from transformers.models.clip import CLIPModel")
try:
    from transformers.models.clip import CLIPModel, CLIPTokenizer
    print("   ✅ Direct import works!")
    direct_import = True
except Exception as e:
    print(f"   ❌ Failed: {e}")
    direct_import = False

# Test 2: Standard import
if not direct_import:
    print("\n2. Trying: from transformers import CLIPModel")
    try:
        from transformers import CLIPModel, CLIPTokenizer
        print("   ✅ Standard import works!")
        direct_import = True
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        print(f"\n   This is the transformers circular import bug.")
        print(f"   Fix: python fix_transformers.py")

if direct_import:
    # Test 3: Load model
    print("\n3. Loading CLIP model...")
    try:
        import torch
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        model.eval()
        print("   ✅ Model loaded!")
        
        # Test 4: Generate embeddings
        print("\n4. Testing embeddings...")
        queries = ["person waving", "walking", "kjhsd random text"]
        
        embeddings = []
        for query in queries:
            tokens = tokenizer(query, return_tensors="pt", padding=True)
            with torch.no_grad():
                features = model.get_text_features(**tokens)
            embedding = features.cpu().numpy().squeeze()
            embeddings.append(embedding)
            print(f"   Query: '{query}' → Embedding shape: {embedding.shape}")
        
        # Test 5: Check embeddings are different
        print("\n5. Checking embedding quality...")
        import numpy as np
        
        sim_1_2 = np.dot(embeddings[0], embeddings[1])  # person waving vs walking
        sim_1_3 = np.dot(embeddings[0], embeddings[2])  # person waving vs random
        sim_2_3 = np.dot(embeddings[1], embeddings[2])  # walking vs random
        
        print(f"   Similarity (person waving vs walking): {sim_1_2:.3f}")
        print(f"   Similarity (person waving vs random):  {sim_1_3:.3f}")
        print(f"   Similarity (walking vs random):        {sim_2_3:.3f}")
        
        if abs(sim_1_2 - sim_1_3) > 0.1:
            print("\n   ✅ Embeddings are distinct! Quality is good.")
        else:
            print("\n   ⚠️  Embeddings are too similar. Quality is poor.")
        
        print("\n" + "=" * 60)
        print("✅ CLIP WORKING! API will use proper embeddings.")
        print("=" * 60)
        print("\nStart API with: python run_api.py")
        print("It should show: '✅ CLIP model loaded successfully!'")
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
else:
    print("\n" + "=" * 60)
    print("❌ CLIP NOT WORKING")
    print("=" * 60)
    print("\nFix with:")
    print("  python fix_transformers.py")
    print("\nOr manually:")
    print("  pip uninstall transformers -y")
    print("  pip install transformers==4.36.0")

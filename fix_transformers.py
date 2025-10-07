"""Fix transformers installation to enable CLIP."""

import sys
import subprocess

print("=" * 60)
print("Fixing Transformers Installation")
print("=" * 60)

# Step 1: Check current transformers
print("\n1. Checking current transformers...")
try:
    import transformers
    print(f"   Current version: {transformers.__version__}")
    print(f"   Location: {transformers.__file__}")
except Exception as e:
    print(f"   Error: {e}")

# Step 2: Uninstall transformers
print("\n2. Uninstalling transformers...")
result = subprocess.run([sys.executable, "-m", "pip", "uninstall", "transformers", "-y"], 
                       capture_output=True, text=True)
print(f"   {result.stdout}")

# Step 3: Install specific version
print("\n3. Installing transformers 4.36.0...")
result = subprocess.run([sys.executable, "-m", "pip", "install", "transformers==4.36.0"], 
                       capture_output=True, text=True)
print(f"   {result.stdout}")

# Step 4: Test import
print("\n4. Testing CLIP import...")
try:
    from transformers import CLIPModel, CLIPTokenizer
    print("   ✅ CLIP imports successfully!")
    
    # Try loading model
    print("\n5. Testing CLIP model loading...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    print("   ✅ CLIP model loaded successfully!")
    print(f"   Model: {type(model)}")
    print(f"   Tokenizer: {type(tokenizer)}")
    
    # Test embedding
    print("\n6. Testing text embedding...")
    tokens = tokenizer("person waving", return_tensors="pt", padding=True)
    import torch
    with torch.no_grad():
        features = model.get_text_features(**tokens)
    print(f"   ✅ Embedding shape: {features.shape}")
    print(f"   ✅ All systems go!")
    
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()
    print("\n   Try alternative fix:")
    print("   pip install sentence-transformers")

print("\n" + "=" * 60)
print("Fix Complete!")
print("=" * 60)
print("\nNext steps:")
print("1. Restart the API: python run_api.py")
print("2. Check logs for: '✅ CLIP model loaded successfully!'")
print("3. Test search at: http://localhost:8081/static/index.html")
print("=" * 60)

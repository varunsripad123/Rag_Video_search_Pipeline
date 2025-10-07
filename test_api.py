"""Test the API endpoints."""
import requests
import json

# Test configuration
API_URL = "http://localhost:8081"
API_KEY = "changeme"

print("Testing API endpoints...\n")

# Test 1: Health check
print("1. Testing health endpoint...")
try:
    response = requests.get(f"{API_URL}/v1/health", headers={"x-api-key": API_KEY})
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    print("   ✅ Health check passed\n")
except Exception as e:
    print(f"   ❌ Health check failed: {e}\n")

# Test 2: Search endpoint
print("2. Testing search endpoint...")
try:
    payload = {
        "query": "person waving",
        "history": [],
        "options": {
            "expand": False,
            "top_k": 5
        }
    }
    
    response = requests.post(
        f"{API_URL}/v1/search/similar",
        headers={
            "Content-Type": "application/json",
            "x-api-key": API_KEY
        },
        json=payload
    )
    
    print(f"   Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"   Answer: {data.get('answer', 'N/A')}")
        print(f"   Results: {len(data.get('results', []))} videos found")
        if data.get('results'):
            print(f"   Top result: {data['results'][0]['label']} (score: {data['results'][0]['score']:.3f})")
        print("   ✅ Search passed\n")
    else:
        print(f"   Response: {response.text}")
        print("   ❌ Search failed\n")
        
except Exception as e:
    print(f"   ❌ Search failed: {e}\n")

print("\n" + "="*50)
print("If both tests passed, open your browser to:")
print(f"   {API_URL}/static/index.html")
print("="*50)
